#!/usr/bin/env python3
"""
SimAI Per-Layer Compute/Communication Analysis
-----------------------------------------------
Breaks down exposed communication by layer type to identify
which layers are most sensitive to compute_scale calibration.

Usage:
    python3 scripts/analyze_layer_gaps.py example/workload_analytical.txt
"""

import sys
import os
from collections import defaultdict


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
        if part == 'model_parallel_NPU_group:' and i + 1 < len(parts):
            header['tp'] = int(parts[i + 1])
        elif part == 'ep:' and i + 1 < len(parts):
            header['ep'] = int(parts[i + 1])
        elif part == 'pp:' and i + 1 < len(parts):
            header['pp'] = int(parts[i + 1])

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


def comm_time_ns(comm_size, comm_type):
    """Simple busbw-based communication time estimate in nanoseconds.

    Uses TP busbw defaults: allreduce 300 GB/s, allgather 280 GB/s,
    reducescatter 280 GB/s, alltoall 230 GB/s.
    """
    if comm_type == 'NONE' or comm_size == 0:
        return 0.0

    busbw = {
        'ALLREDUCE': 300.0,
        'ALLGATHER': 280.0,
        'REDUCESCATTER': 280.0,
        'ALLTOALL': 230.0,
        'ALLGATHER_DP': 380.0,
        'REDUCESCATTER_DP': 380.0,
        'ALLTOALL_EP': 80.0,
        'ALLGATHER_DP_EP': 200.0,
        'REDUCESCATTER_DP_EP': 200.0,
    }.get(comm_type, 200.0)

    data_gb = comm_size / (1024.0 ** 3)
    return data_gb / busbw * 1e9


def analyze_gaps(layers, header, compute_scale):
    """Return per-layer-type gap analysis.

    Models SimAI's actual event-driven dispatch pattern (Workload.cc iterate_data_parallel):
    - Forward: fp_comp -> fp_comm. fp_comm hidden by same layer's ig_comp.
    - Input grad: ig_comp -> ig_comm. ig_comm hidden by same layer's wg_comp.
    - Weight grad: wg_comp -> wg_comm. wg_comm hidden by NEXT layer's fp_comp.
    """
    layer_groups = defaultdict(lambda: {
        'count': 0, 'total_comp': 0, 'total_comm': 0, 'exposed_comm': 0,
        'max_exposed': 0, 'max_layer': '', 'has_comm': False
    })

    for i, layer in enumerate(layers):
        name = layer['name']
        base = name.rstrip('0123456789_')

        fp_comp = layer['fp_comp'] * compute_scale
        ig_comp = layer['ig_comp'] * compute_scale
        wg_comp = layer['wg_comp'] * compute_scale

        fp_comm = comm_time_ns(layer['fp_comm_size'], layer['fp_comm_type'])
        ig_comm = comm_time_ns(layer['ig_comm_size'], layer['ig_comm_type'])
        wg_comm = comm_time_ns(layer['wg_comm_size'], layer['wg_comm_type'])

        # fp_comm hidden by ig_comp (same layer)
        fp_exposed = max(fp_comm - ig_comp, 0)
        # ig_comm hidden by wg_comp (same layer)
        ig_exposed = max(ig_comm - wg_comp, 0)
        # wg_comm hidden by NEXT layer's fp_comp (cross-layer overlap)
        if i + 1 < len(layers):
            next_layer = layers[i + 1]
            next_fp_comp = next_layer['fp_comp'] * compute_scale
            wg_exposed = max(wg_comm - next_fp_comp, 0)
        else:
            # Last layer - no next layer to hide wg_comm
            wg_exposed = wg_comm

        total_comp = fp_comp + ig_comp + wg_comp
        total_comm = fp_comm + ig_comm + wg_comm
        total_exposed = fp_exposed + ig_exposed + wg_exposed

        g = layer_groups[base]
        g['count'] += 1
        g['total_comp'] += total_comp
        g['total_comm'] += total_comm
        g['exposed_comm'] += total_exposed
        g['has_comm'] = g['has_comm'] or total_comm > 0

        if total_exposed > g['max_exposed']:
            g['max_exposed'] = total_exposed
            g['max_layer'] = name

    return layer_groups


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/analyze_layer_gaps.py <workload_file> [compute_scale]")
        sys.exit(1)

    filepath = sys.argv[1]
    compute_scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    layers, header = parse_workload(filepath)
    layer_groups = analyze_gaps(layers, header, compute_scale)

    # Sort by exposed communication (descending)
    sorted_groups = sorted(
        layer_groups.items(),
        key=lambda x: x[1]['exposed_comm'],
        reverse=True
    )

    print(f"{'='*90}")
    print(f"Per-Layer Compute/Communication Gap Analysis")
    print(f"compute_scale = {compute_scale}")
    print(f"{'='*90}")
    print(f"{'Layer Type':<25} {'Count':>6} {'Comp(ms)':>10} {'Comm(ms)':>10} "
          f"{'Exposed(ms)':>12} {'Exposed%':>8} {'Max Layer':>20}")
    print(f"{'-'*25} {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*8} {'-'*20}")

    grand_comp = 0
    grand_comm = 0
    grand_exposed = 0

    for name, g in sorted_groups:
        if g['exposed_comm'] < 0.001 and g['total_comm'] < 0.001:
            continue  # skip layers with no significant time

        comp_ms = g['total_comp'] / 1e6
        comm_ms = g['total_comm'] / 1e6
        exp_ms = g['exposed_comm'] / 1e6
        ratio = g['exposed_comm'] / g['total_comm'] * 100 if g['total_comm'] > 0 else 0

        grand_comp += g['total_comp']
        grand_comm += g['total_comm']
        grand_exposed += g['exposed_comm']

        print(f"{name:<25} {g['count']:>6} {comp_ms:>10.2f} {comm_ms:>10.2f} "
              f"{exp_ms:>12.2f} {ratio:>7.1f}% {g['max_layer']:>20}")

    total_ms = (grand_comp + grand_exposed) / 1e6
    grand_comp_ms = grand_comp / 1e6
    grand_comm_ms = grand_comm / 1e6
    grand_exp_ms = grand_exposed / 1e6
    overall = grand_exposed / grand_comm * 100 if grand_comm > 0 else 0

    print(f"{'-'*90}")
    print(f"{'TOTAL':<25} {len(layers):>6} {grand_comp_ms:>10.2f} {grand_comm_ms:>10.2f} "
          f"{grand_exp_ms:>12.2f} {overall:>7.1f}%")
    print(f"{'='*90}")
    print(f"End-to-end iteration time (est.): {total_ms:.2f} ms")
    print(f"Overall exposed communication: {overall:.1f}%")

    # Calibration recommendations
    print(f"\nCalibration diagnostics:")
    if overall < 5:
        print(f"  [LOW] Exposed comm <5%. Compute times may be overestimated.")
        print(f"  Recommend: decrease compute_scale below {compute_scale}")
    elif overall > 40:
        print(f"  [HIGH] Exposed comm >40%. Communication-dominated regime.")
        print(f"  Recommend: validate against real hardware; compute_scale may be too low")
    else:
        print(f"  [OK] Exposed comm in 5-40% range (plausible for distributed training)")


if __name__ == '__main__':
    main()
