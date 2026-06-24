#!/usr/bin/env python3
"""
SimAI Compute Time Cross-Validation
-------------------------------------
Compares AICB-generated compute times in a workload file against
independent analytical estimates from FLOP counting and roofline modeling.

This tool identifies systematic biases in AICB's compute time estimates
and suggests calibration values for compute_scale.

Methodology:
  1. Parse AICB workload file -> extract per-layer per-phase compute times
  2. Estimate per-layer FLOPs from model dimensions using published formulas
  3. Convert FLOPs to time using roofline model: T = FLOPs / (Peak_FLOPs * efficiency)
  4. Compare AICB estimates vs FLOP-based estimates per layer type
  5. Recommend compute_scale to correct systematic bias

Usage:
    python3 scripts/cross_validate_compute.py example/workload_analytical.txt
    python3 scripts/cross_validate_compute.py example/workload_analytical.txt --gpu H100 --efficiency 0.4
"""

import argparse
import math
import sys
import os
from collections import defaultdict


# GPU specifications (peak FP16/BF16 TFLOPS, HBM bandwidth GB/s)
GPU_SPECS = {
    'A100': {'fp16_tflops': 312.0, 'hbm_bw': 2039.0, 'l2_bw': 4100.0},
    'A800': {'fp16_tflops': 312.0, 'hbm_bw': 2039.0, 'l2_bw': 4100.0},
    'H100': {'fp16_tflops': 989.0, 'hbm_bw': 3352.0, 'l2_bw': 7200.0},
    'H800': {'fp16_tflops': 756.0, 'hbm_bw': 3352.0, 'l2_bw': 7200.0},
    'H20':  {'fp16_tflops': 148.0, 'hbm_bw': 4000.0, 'l2_bw': 6000.0},
    'V100': {'fp16_tflops': 125.0, 'hbm_bw': 900.0,  'l2_bw': 2200.0},
}


def parse_workload(filepath):
    """Parse a SimAI workload text file."""
    layers = []
    header = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if not lines:
        return layers, header

    header_line = lines[0].strip()
    parts = header_line.split()
    for i, part in enumerate(parts):
        if part.startswith('model_parallel_NPU_group:') and i + 1 < len(parts):
            header['tp'] = int(parts[i + 1])
        elif part.startswith('ep:') and i + 1 < len(parts):
            header['ep'] = int(parts[i + 1])
        elif part.startswith('pp:') and i + 1 < len(parts):
            header['pp'] = int(parts[i + 1])
        elif part.startswith('vpp:') and i + 1 < len(parts):
            header['vpp'] = int(parts[i + 1])
        elif part.startswith('ga:') and i + 1 < len(parts):
            header['ga'] = int(parts[i + 1])
        elif part.startswith('all_gpus:') and i + 1 < len(parts):
            header['all_gpus'] = int(parts[i + 1])

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 12:
            continue
        try:
            layer = {
                'name': parts[0],
                'fp_comp': int(parts[2]),
                'fp_comm_type': parts[3],
                'fp_comm_size': int(parts[4]),
                'ig_comp': int(parts[5]),
                'ig_comm_type': parts[6],
                'ig_comm_size': int(parts[7]),
                'wg_comp': int(parts[8]),
                'wg_comm_type': parts[9],
                'wg_comm_size': int(parts[10]),
            }
            layers.append(layer)
        except (ValueError, IndexError):
            continue

    return layers, header


def estimate_transformer_flops(hidden_size, num_heads, seq_len, vocab_size,
                                num_layers, ffn_multiplier=4.0):
    """Estimate FLOPs per layer for a standard transformer.

    Based on Kaplan et al. (2020) and llm-analysis formulas.
    Returns FLOPs for: attention fwd, attention bwd, ffn fwd, ffn bwd.
    All values are per-layer for a single sample.
    """
    # Attention FLOPs (forward): 4 * hidden_size^2 * seq_len  (Q,K,V,O projections)
    #                              + 2 * num_heads * seq_len^2 * (hidden_size/num_heads)
    #                              = 4 * h^2 * s + 2 * h * s^2
    attention_fwd_flops = 4.0 * hidden_size**2 * seq_len + 2.0 * hidden_size * seq_len**2
    attention_bwd_flops = attention_fwd_flops * 2.0  # backward ~2x forward

    # FFN FLOPs (forward): 2 * hidden_size * (ffn_multiplier * hidden_size) * seq_len
    #                       for both up-projection and down-projection
    ffn_hidden = int(ffn_multiplier * hidden_size)
    ffn_fwd_flops = 2.0 * hidden_size * ffn_hidden * seq_len * 2.0  # up + down
    ffn_bwd_flops = ffn_fwd_flops * 2.0

    # Embedding FLOPs (forward): hidden_size * vocab_size * seq_len  (lookup + projection)
    embed_fwd_flops = hidden_size * vocab_size * seq_len
    embed_bwd_flops = embed_fwd_flops * 2.0

    return {
        'attention_fwd': attention_fwd_flops,
        'attention_bwd': attention_bwd_flops,
        'ffn_fwd': ffn_fwd_flops,
        'ffn_bwd': ffn_bwd_flops,
        'embed_fwd': embed_fwd_flops,
        'embed_bwd': embed_bwd_flops,
    }


def roofline_time(flops, gpu_spec, efficiency=0.35):
    """Convert FLOPs to time using roofline model.

    T = max(FLOPs / Peak_FLOPs, Bytes / HBM_BW)
    For transformer layers, we only model the compute-bound case
    since arithmetic intensity is typically high for matrix multiplies.
    The memory-bound case is for element-wise ops like layernorm/activation.
    """
    peak_flops = gpu_spec['fp16_tflops'] * 1e12  # convert to FLOPs/s
    compute_time_s = flops / (peak_flops * efficiency)
    return compute_time_s * 1e9  # nanoseconds


def roofline_time_memory_bound(bytes_count, gpu_spec):
    """Estimate time for memory-bound operations."""
    hbm_bw = gpu_spec['hbm_bw'] * 1e9  # bytes/s
    return bytes_count / hbm_bw * 1e9  # nanoseconds


def classify_layer_type(name):
    """Classify a layer name into a canonical type."""
    name_lower = name.lower()
    if 'embedding' in name_lower:
        return 'embedding'
    elif 'attention' in name_lower and 'column' in name_lower:
        return 'attention_column'
    elif 'attention' in name_lower and 'row' in name_lower:
        return 'attention_row'
    elif 'mlp' in name_lower or 'moe' in name_lower:
        return 'mlp_moe'
    elif 'layernorm' in name_lower or 'norm' in name_lower:
        return 'layernorm'
    elif 'grad_param' in name_lower:
        return 'grad_param'
    elif 'final' in name_lower:
        return 'final'
    elif 'cross_entropy' in name_lower:
        return 'cross_entropy'
    else:
        return 'other'


def estimate_model_dimensions(layers, header):
    """Heuristically estimate model dimensions from workload characteristics.

    Uses the communication sizes and layer counts to reverse-engineer
    hidden_size, seq_len, etc. from the workload structure.
    """
    tp = header.get('tp', 1)
    pp = header.get('pp', 1)
    ep = header.get('ep', 1)

    # Count layer types
    type_counts = defaultdict(int)
    for layer in layers:
        ltype = classify_layer_type(layer['name'])
        type_counts[ltype] += 1

    num_attention_layers = type_counts.get('attention_column', 0)
    if num_attention_layers == 0:
        num_attention_layers = 32  # default

    # Estimate hidden_size from TP all-gather message sizes
    # TP all-gather message size = hidden_size * seq_len / tp * bytes_per_element
    # For attention_column: comm_size = 50331648 (50MB) typical at seq=2048
    # hidden_size ~ sqrt(comm_size * tp / seq_len / bytes_per_element)
    typical_comm = None
    for layer in layers:
        if classify_layer_type(layer['name']) == 'attention_column' and layer['fp_comm_size'] > 0:
            typical_comm = layer['fp_comm_size']
            break

    if typical_comm:
        # Reverse engineer: comm_size = h * s / tp * 2 (FP16) * layer_fraction
        # For attention_column QKV projection: 3*h elements * FP16 = 6*h bytes
        # Actually: comm_size = hidden_size * seq_len / tp * 2 bytes (FP16 activations)
        # Default seq_len = 2048
        seq_len = 2048
        hidden_size = int(math.sqrt(typical_comm * tp / (seq_len * 2)))
        # Refine: typical_comm for ColumnLinear AG is h * s
        hidden_size = typical_comm * tp // (seq_len * 2)
    else:
        hidden_size = 4096
        seq_len = 2048

    # Clamp to reasonable ranges
    hidden_size = max(1024, min(hidden_size, 16384))
    seq_len = 2048  # standard for GPT-3 family

    # Estimate vocab size from embedding comm
    vocab_size = 50257  # GPT-2/3 default
    for layer in layers:
        if classify_layer_type(layer['name']) == 'embedding' and layer['fp_comm_size'] > 0:
            # embedding allreduce = vocab * h / tp * 2 bytes
            vocab_size = max(10000, layer['fp_comm_size'] * tp // (hidden_size * 2))
            break

    # Estimate ffn_multiplier
    # MLP communication size reflects the expert hidden dimension
    ffn_multiplier = 4.0
    for layer in layers:
        if classify_layer_type(layer['name']) == 'mlp_moe':
            if layer['fp_comm_size'] > 0 and layer['fp_comm_size'] < 10_000_000:
                # MoE: comm_size for ALLGATHER reflects expert hidden dim
                # For SwiGLU: ffn_hidden = 8/3 * hidden_size roughly
                expert_dim = layer['fp_comm_size'] * tp // (seq_len * 2)
                ffn_multiplier = max(1.0, min(expert_dim / hidden_size, 8.0))
                break

    return {
        'hidden_size': hidden_size,
        'num_heads': max(8, hidden_size // 128),
        'seq_len': seq_len,
        'vocab_size': vocab_size,
        'num_layers': num_attention_layers,
        'ffn_multiplier': ffn_multiplier,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Cross-validate AICB compute times against FLOP-based estimates'
    )
    parser.add_argument('workload', help='Path to SimAI workload file')
    parser.add_argument('--gpu', default='A100',
                        choices=['A100', 'A800', 'H100', 'H800', 'H20', 'V100'],
                        help='Target GPU architecture (default: A100)')
    parser.add_argument('--efficiency', type=float, default=0.35,
                        help='FLOPs efficiency for training (default: 0.35 = 35%%)')
    parser.add_argument('--output', help='Output path for corrected workload file')

    args = parser.parse_args()

    if not os.path.exists(args.workload):
        print(f"Error: Workload file not found: {args.workload}")
        sys.exit(1)

    layers, header = parse_workload(args.workload)
    gpu_spec = GPU_SPECS[args.gpu]

    # Estimate model dimensions from workload data
    dims = estimate_model_dimensions(layers, header)
    flops = estimate_transformer_flops(
        dims['hidden_size'], dims['num_heads'], dims['seq_len'],
        dims['vocab_size'], dims['num_layers'], dims['ffn_multiplier']
    )

    print(f"{'='*85}")
    print(f"SimAI Compute Time Cross-Validation")
    print(f"{'='*85}")
    print(f"Workload: {args.workload}")
    print(f"GPU: {args.gpu} ({gpu_spec['fp16_tflops']} TFLOPS FP16)")
    print(f"FLOPs efficiency: {args.efficiency*100:.0f}%")
    print()
    print(f"Estimated model dimensions (from workload structure):")
    print(f"  hidden_size = {dims['hidden_size']}")
    print(f"  num_heads = {dims['num_heads']}")
    print(f"  seq_len = {dims['seq_len']}")
    print(f"  vocab_size = {dims['vocab_size']}")
    print(f"  num_layers = {dims['num_layers']}")
    print(f"  ffn_multiplier = {dims['ffn_multiplier']:.1f}")
    print()
    print(f"Estimated per-layer FLOPs:")
    for key, val in sorted(flops.items()):
        print(f"  {key}: {val/1e9:.2f} GFLOPs")
    print()

    # Compute roofline-based time estimates per layer type

    def tp_factor(dims, header):
        """TP reduces per-GPU compute proportionally."""
        return header.get('tp', 1)

    roofline = {
        'attention_column': (
            roofline_time(flops['attention_fwd'] / 3.0, gpu_spec, args.efficiency),
            roofline_time(flops['attention_bwd'] / 3.0, gpu_spec, args.efficiency),
            roofline_time(flops['attention_bwd'] / 3.0, gpu_spec, args.efficiency),
        ),
        'attention_row': (
            roofline_time(flops['attention_fwd'] / 3.0, gpu_spec, args.efficiency),
            roofline_time(flops['attention_bwd'] / 3.0, gpu_spec, args.efficiency),
            roofline_time(flops['attention_bwd'] / 3.0, gpu_spec, args.efficiency),
        ),
        'mlp_moe': (
            roofline_time(flops['ffn_fwd'] / tp_factor(dims, header), gpu_spec, args.efficiency),
            roofline_time(flops['ffn_fwd'] / tp_factor(dims, header), gpu_spec, args.efficiency),
            roofline_time(flops['ffn_bwd'] / tp_factor(dims, header), gpu_spec, args.efficiency),
        ),
        'embedding': (
            roofline_time(flops['embed_fwd'], gpu_spec, args.efficiency),
            0,  # ig_comp = 0 for embedding
            roofline_time(flops['embed_bwd'], gpu_spec, args.efficiency),
        ),
    }

    # Collect per-layer-type AICB vs FLOP comparison
    layer_stats = defaultdict(lambda: {
        'count': 0, 'aicb_fp': [], 'aicb_ig': [], 'aicb_wg': [],
        'roofline_fp': 0, 'roofline_ig': 0, 'roofline_wg': 0
    })

    for layer in layers:
        ltype = classify_layer_type(layer['name'])
        if ltype in roofline:
            rfp, rig, rwg = roofline[ltype]
        else:
            rfp = rig = rwg = 0

        stats = layer_stats[ltype]
        stats['count'] += 1
        if layer['fp_comp'] > 1:  # skip zero/placeholder values
            stats['aicb_fp'].append(layer['fp_comp'])
        if layer['ig_comp'] > 1:
            stats['aicb_ig'].append(layer['ig_comp'])
        if layer['wg_comp'] > 1:
            stats['aicb_wg'].append(layer['wg_comp'])
        stats['roofline_fp'] = rfp
        stats['roofline_ig'] = rig
        stats['roofline_wg'] = rwg

    # Print comparison table
    print(f"{'Layer Type':<20} {'Count':>6} {'AICB FP(us)':>12} {'Roofline FP(us)':>16} "
          f"{'AICB IG(us)':>12} {'Roofline IG(us)':>16} {'AICB WG(us)':>12} {'Roofline WG(us)':>16} {'Ratio':>8}")
    print(f"{'-'*20} {'-'*6} {'-'*12} {'-'*16} {'-'*12} {'-'*16} {'-'*12} {'-'*16} {'-'*8}")

    ratios = []
    for ltype in sorted(layer_stats.keys()):
        s = layer_stats[ltype]
        aicb_fp_avg = sum(s['aicb_fp']) / len(s['aicb_fp']) / 1000 if s['aicb_fp'] else 0
        aicb_ig_avg = sum(s['aicb_ig']) / len(s['aicb_ig']) / 1000 if s['aicb_ig'] else 0
        aicb_wg_avg = sum(s['aicb_wg']) / len(s['aicb_wg']) / 1000 if s['aicb_wg'] else 0
        rf_fp = s['roofline_fp'] / 1000
        rf_ig = s['roofline_ig'] / 1000
        rf_wg = s['roofline_wg'] / 1000

        # Compute average ratio (AICB / Roofline)
        ratio_values = []
        if rf_fp > 0 and aicb_fp_avg > 0:
            ratio_values.append(aicb_fp_avg / rf_fp)
        if rf_ig > 0 and aicb_ig_avg > 0:
            ratio_values.append(aicb_ig_avg / rf_ig)
        if rf_wg > 0 and aicb_wg_avg > 0:
            ratio_values.append(aicb_wg_avg / rf_wg)
        avg_ratio = sum(ratio_values) / len(ratio_values) if ratio_values else 0

        if avg_ratio > 0:
            ratios.append(avg_ratio)

        print(f"{ltype:<20} {s['count']:>6} {aicb_fp_avg:>12.1f} {rf_fp:>16.1f} "
              f"{aicb_ig_avg:>12.1f} {rf_ig:>16.1f} {aicb_wg_avg:>12.1f} {rf_wg:>16.1f} "
              f"{avg_ratio:>7.3f}x")

    print()
    if ratios:
        median_ratio = sorted(ratios)[len(ratios)//2]
        print(f"Median AICB/Roofline ratio: {median_ratio:.3f}x")
        print(f"AICB compute times are {'over' if median_ratio > 1 else 'under'}-estimated "
              f"by {abs(1 - median_ratio)*100:.0f}% vs FLOP-based roofline estimates")
        recommended_cs = 1.0 / median_ratio if median_ratio > 0 else 1.0
        print(f"Recommended compute_scale: {recommended_cs:.3f}")
        print(f"  (This scales AICB estimates to match FLOP-based predictions)")
    else:
        print("No comparable layer types found. Check workload format.")


if __name__ == '__main__':
    main()
