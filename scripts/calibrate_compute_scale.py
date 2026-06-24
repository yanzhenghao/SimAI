#!/usr/bin/env python3
"""
SimAI Compute Scale Calibration Tool
-------------------------------------
Analyzes a SimAI workload file to estimate the effect of compute_scale
on end-to-end iteration time and exposed communication ratio.

Usage:
    python3 scripts/calibrate_compute_scale.py example/workload_analytical.txt
    python3 scripts/calibrate_compute_scale.py example/workload_analytical.txt --target-exposed 0.10
"""

import argparse
import sys
import os


def parse_workload(filepath):
    """Parse a SimAI workload text file and extract per-layer compute and communication times.

    Returns:
        layers: list of dicts with keys:
            name, fp_comp, fp_comm_type, fp_comm_size,
            ig_comp, ig_comm_type, ig_comm_size,
            wg_comp, wg_comm_type, wg_comm_size, wg_update
        header: dict with model_parallel_npu_group, ep, pp, vpp, ga, all_gpus, checkpoints
    """
    layers = []
    header = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if not lines:
        return layers, header

    # Parse header line
    header_line = lines[0].strip()
    parts = header_line.split()
    for i, part in enumerate(parts):
        if part == 'model_parallel_NPU_group:' and i + 1 < len(parts):
            header['tp'] = int(parts[i + 1])
        elif part == 'ep:' and i + 1 < len(parts):
            header['ep'] = int(parts[i + 1])
        elif part == 'pp:' and i + 1 < len(parts):
            header['pp'] = int(parts[i + 1])
        elif part == 'vpp:' and i + 1 < len(parts):
            header['vpp'] = int(parts[i + 1])
        elif part == 'ga:' and i + 1 < len(parts):
            header['ga'] = int(parts[i + 1])
        elif part == 'all_gpus:' and i + 1 < len(parts):
            header['all_gpus'] = int(parts[i + 1])
        elif part == 'checkpoints:' and i + 1 < len(parts):
            header['checkpoints'] = int(parts[i + 1])

    # Parse layer lines
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
                'depen': int(parts[1]),
                'fp_comp': int(parts[2]),       # forward pass compute time (nanoseconds)
                'fp_comm_type': parts[3],        # e.g., ALLREDUCE, ALLGATHER, NONE
                'fp_comm_size': int(parts[4]),   # communication size (bytes)
                'ig_comp': int(parts[5]),        # input gradient compute time
                'ig_comm_type': parts[6],
                'ig_comm_size': int(parts[7]),
                'wg_comp': int(parts[8]),        # weight gradient compute time
                'wg_comm_type': parts[9],
                'wg_comm_size': int(parts[10]),
                'wg_update': int(parts[11]) if len(parts) > 11 else 0,
            }
            layers.append(layer)
        except (ValueError, IndexError):
            continue

    return layers, header


def estimate_comm_time(comm_size, comm_type, group_type, busbw_config, tp_size=1, dp_size=1):
    """Estimate communication time from message size and bus bandwidth.

    Uses the SimAI analytical model: time = data_size / busbw
    busbw values come from the busbw.yaml configuration.
    """
    if comm_type == 'NONE' or comm_size == 0:
        return 0.0

    # Convert bytes to GB for busbw calculation
    data_gb = comm_size / (1024.0 ** 3)

    # Look up bus bandwidth
    bw = None
    for key_prefix in [group_type, 'TP', 'DP', 'EP']:
        key = f"{key_prefix}:{comm_type.lower()}"
        if key in busbw_config:
            bw = busbw_config[key]
            break

    if bw is None or bw == 'null':
        # Fallback estimates per collective type
        fallback = {
            'ALLREDUCE': 300.0,
            'ALLGATHER': 280.0,
            'REDUCESCATTER': 280.0,
            'ALLTOALL': 230.0,
            'ALLREDUCE_TP': 300.0,
            'ALLGATHER_TP': 280.0,
            'REDUCESCATTER_TP': 280.0,
            'ALLGATHER_DP': 380.0,
            'REDUCESCATTER_DP': 380.0,
            'ALLTOALL_EP': 80.0,
        }
        bw = fallback.get(comm_type, 200.0)

    bw = float(bw)
    if bw <= 0:
        return 0.0

    # busbw is in GB/s, data_gb / bw gives seconds
    time_s = data_gb / bw
    return time_s * 1e9  # convert to nanoseconds


def load_busbw_config(filepath='example/busbw.yaml'):
    """Load bus bandwidth configuration from YAML-like file."""
    config = {}
    if not os.path.exists(filepath):
        # Use defaults if file not found
        return config

    with open(filepath, 'r') as f:
        group = 'TP'
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.endswith(':'):
                group = line[:-1]
                continue
            parts = line.split(':')
            if len(parts) >= 2:
                coll_name = parts[0].strip().rstrip(',')
                bw_value = parts[1].strip()
                key = f"{group}:{coll_name}"
                config[key] = bw_value

    return config


def simulate_iteration(layers, header, compute_scale, busbw_config):
    """Simulate one training iteration with the given compute_scale.

    Models SimAI's event-driven dispatch (Workload.cc iterate_data_parallel):
    - Forward: fp_comp -> fp_comm. fp_comm hidden by same layer's ig_comp.
    - Input grad: ig_comp -> ig_comm. ig_comm hidden by same layer's wg_comp.
    - Weight grad: wg_comp -> wg_comm. wg_comm hidden by NEXT layer's fp_comp.
    """
    total_compute = 0.0
    total_comm = 0.0
    exposed_comm = 0.0

    for i, layer in enumerate(layers):
        fp_comp = layer['fp_comp'] * compute_scale
        ig_comp = layer['ig_comp'] * compute_scale
        wg_comp = layer['wg_comp'] * compute_scale

        total_compute += fp_comp + ig_comp + wg_comp

        fp_comm = estimate_comm_time(
            layer['fp_comm_size'], layer['fp_comm_type'],
            'TP', busbw_config,
            tp_size=header.get('tp', 1)
        )
        ig_comm = estimate_comm_time(
            layer['ig_comm_size'], layer['ig_comm_type'],
            'TP', busbw_config,
            tp_size=header.get('tp', 1)
        )
        wg_comm = estimate_comm_time(
            layer['wg_comm_size'], layer['wg_comm_type'],
            'TP', busbw_config,
            tp_size=header.get('tp', 1)
        )
        total_comm += fp_comm + ig_comm + wg_comm

        # fp_comm hidden by ig_comp (same layer)
        if fp_comm > ig_comp:
            exposed_comm += (fp_comm - ig_comp)
        # ig_comm hidden by wg_comp (same layer)
        if ig_comm > wg_comp:
            exposed_comm += (ig_comm - wg_comp)
        # wg_comm hidden by NEXT layer's fp_comp (cross-layer overlap)
        if i + 1 < len(layers):
            next_layer = layers[i + 1]
            next_fp_comp = next_layer['fp_comp'] * compute_scale
            if wg_comm > next_fp_comp:
                exposed_comm += (wg_comm - next_fp_comp)
        else:
            exposed_comm += wg_comm

    total_time = total_compute + exposed_comm
    return total_compute, total_comm, exposed_comm, total_time


def main():
    parser = argparse.ArgumentParser(
        description='SimAI Compute Scale Calibration Tool'
    )
    parser.add_argument(
        'workload', help='Path to SimAI workload file (e.g., example/workload_analytical.txt)'
    )
    parser.add_argument(
        '--target-exposed', type=float, default=None,
        help='Target exposed communication ratio (e.g., 0.10 for 10%%). '
             'If specified, finds the compute_scale that achieves this ratio.'
    )
    parser.add_argument(
        '--busbw', default='example/busbw.yaml',
        help='Path to busbw.yaml configuration file'
    )
    parser.add_argument(
        '--scales', type=str, default='1.0,0.5,0.3,0.2,0.1,0.05',
        help='Comma-separated list of compute_scale values to test'
    )

    args = parser.parse_args()

    if not os.path.exists(args.workload):
        print(f"Error: Workload file not found: {args.workload}")
        sys.exit(1)

    layers, header = parse_workload(args.workload)
    busbw_config = load_busbw_config(args.busbw)

    if not layers:
        print("Error: No layers parsed from workload file")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"SimAI Compute Scale Calibration")
    print(f"{'='*70}")
    print(f"Workload: {args.workload}")
    print(f"Layers: {len(layers)}")
    print(f"Configuration: TP={header.get('tp',1)}, EP={header.get('ep',1)}, "
          f"PP={header.get('pp',1)}, GPUs={header.get('all_gpus','?')}")
    print()

    scales = [float(s) for s in args.scales.split(',')]

    if args.target_exposed is not None:
        print(f"Target exposed communication ratio: {args.target_exposed*100:.1f}%")
        print()

    # Run simulation at each compute_scale
    print(f"{'compute_scale':>14} {'total_comp':>12} {'total_comm':>12} "
          f"{'exposed_comm':>12} {'exposed_ratio':>12} {'total_time':>14}")
    print(f"{'-'*14} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*14}")

    results = []
    for cs in scales:
        total_comp, total_comm, exposed_comm, total_time = simulate_iteration(
            layers, header, cs, busbw_config
        )
        total_comp_ms = total_comp / 1e6
        total_comm_ms = total_comm / 1e6
        exposed_ms = exposed_comm / 1e6
        total_ms = total_time / 1e6
        exposed_ratio = exposed_comm / total_time if total_time > 0 else 0.0

        results.append((cs, total_comp_ms, total_comm_ms, exposed_ms, exposed_ratio, total_ms))

        print(f"{cs:>14.2f} {total_comp_ms:>10.2f}ms {total_comm_ms:>10.2f}ms "
              f"{exposed_ms:>10.2f}ms {exposed_ratio:>10.1%} {total_ms:>12.2f}ms")

    print()

    # Find compute_scale for target exposed ratio
    if args.target_exposed is not None and len(results) >= 2:
        target = args.target_exposed
        # Find scales that bracket the target
        for i in range(len(results) - 1):
            r_lo = results[i][4]
            r_hi = results[i + 1][4]
            cs_lo = results[i][0]
            cs_hi = results[i + 1][0]
            if (r_lo >= target >= r_hi) or (r_hi >= target >= r_lo):
                # Linear interpolation
                if abs(r_hi - r_lo) > 1e-9:
                    frac = (target - r_lo) / (r_hi - r_lo)
                    target_cs = cs_lo + frac * (cs_hi - cs_lo)
                else:
                    target_cs = (cs_lo + cs_hi) / 2
                print(f"*** Recommended compute_scale for {target*100:.1f}% exposed: "
                      f"{target_cs:.3f} ***")
                print()
                break

    # Calibration guidance
    print("Calibration guidance:")
    print("  compute_scale=1.0 : AICB analytical estimates (optimistic)")
    print("  compute_scale=0.3 : Typical training FLOPs efficiency (~30%)")
    print("  compute_scale=0.1 : Conservative estimate")
    print()
    print("Real hardware typically shows 5-15% exposed communication.")
    print("Adjust compute_scale until the exposed_ratio matches your measurements.")
    print()
    print("To apply: ./bin/SimAI_analytical -w workload.txt --compute_scale <value>")


if __name__ == '__main__':
    main()
