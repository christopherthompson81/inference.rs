#!/usr/bin/env python3
"""Per-kernel hardware counters for the Vulkan backend from Nsight Graphics' GPU Trace.

Runs a binary under `ngfx --activity "GPU Trace Profiler"` with INFERENCE_VULKAN_LABELS set, so each dispatch is a
labelled range, then reads the trace's auto-exported tables (one row a range) and prints a table a kernel: dispatches,
time, and the counters weighted by each dispatch's time. With --export it reads an existing export instead.

    scripts/vulkan_gputrace.py --exe target/debug/deps/inference_vulkan-<hash> \\
        --args "--ignored --exact kernels::tests::expert_matmul_throughput" --start-after-submits 3
    scripts/vulkan_gputrace.py --export /tmp/trace
"""

import argparse
import collections
import csv
import glob
import os
import subprocess
import sys
import tempfile

NGFX_GLOB = "/opt/nvidia/nsight-graphics-for-linux/nsight-graphics-for-linux-*/host/linux-desktop-nomad-x64/ngfx"
LABELS_ENV = "INFERENCE_VULKAN_LABELS"
TRACE_TIMEOUT_S = 600
DEFAULT_DURATION_MS = 200
# the export's tables, a row a labelled range in the same order
REGIMES = "GPUTRACE_REGIMES.xls"
EVENTS = "D3DPERF_EVENTS.xls"
NAME_COLUMN = "flattened_event_name"
# (heading, metric name suffix): the counters worth a column, matched by suffix since units prefix them
COLUMNS = [
    ("sm%", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    (
        "warps%",
        "tpc__warps_active_shader_cs_realtime.avg.pct_of_peak_sustained_elapsed",
    ),
    ("ctas%", "tpc__ctas_active_queue_sync_realtime.avg.pct_of_peak_sustained_elapsed"),
    (
        "regs%",
        "tpc__sm_rf_registers_allocated_shader_cs_queue_sync_realtime.avg.pct_of_peak_sustained_elapsed",
    ),
    (
        "smem%",
        "tpc__l1tex_mem_shared_data_compute_bytes_allocated_queue_sync_realtime.avg.pct_of_peak_sustained_elapsed",
    ),
    ("l1%", "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("l1hit", "l1tex__t_sector_hit_rate.pct"),
    ("l2%", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("l2hit", "lts__average_t_sector_hit_rate_realtime.pct"),
    ("dram%", "dramc__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("alu%", "sm__inst_executed_pipe_alu_realtime.avg.pct_of_peak_sustained_elapsed"),
    ("fma%", "smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed"),
    ("xu%", "smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed"),
    ("lsu%", "l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed"),
    (
        "shm%",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_realtime.avg.pct_of_peak_sustained_elapsed",
    ),
]


def ngfx_path():
    found = sorted(glob.glob(NGFX_GLOB))
    if not found:
        sys.exit(f"no ngfx under {NGFX_GLOB}: install Nsight Graphics")
    return found[-1]


def run_trace(args, out):
    cmd = [
        ngfx_path(),
        "--activity",
        "GPU Trace Profiler",
        "--no-timeout",
        "--exe",
        os.path.abspath(args.exe),
        "--dir",
        os.path.abspath(args.dir),
        "--env",
        f"{LABELS_ENV}=1;{args.env}",
        "--args",
        args.args,
        "--max-duration-ms",
        str(args.duration_ms),
        "--auto-export",
        "--output-dir",
        out,
    ]
    if args.start_after_ms is not None:
        cmd += ["--start-after-ms", str(args.start_after_ms)]
    else:
        cmd += ["--start-after-submits", str(args.start_after_submits)]
    if args.multi_pass:
        cmd.append("--multi-pass-metrics")
    # Qt would read a --platform for itself; offscreen keeps ngfx headless
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    # ngfx can exit nonzero on a dropped connection after it exported; the export decides
    done = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=TRACE_TIMEOUT_S,
        check=False,
    )
    if not glob.glob(os.path.join(out, "*", REGIMES)):
        sys.exit(
            f"ngfx exported no {REGIMES}:\n{done.stdout[-3000:]}\n{done.stderr[-2000:]}"
        )


def read_rows(export):
    [regimes] = glob.glob(os.path.join(export, "*", REGIMES)) or sys.exit(
        f"no {REGIMES} under {export}"
    )
    with open(regimes) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    times = []
    events = os.path.join(os.path.dirname(regimes), EVENTS)
    if os.path.exists(events):
        with open(events) as f:
            times = [float(r["time_ms"]) for r in csv.DictReader(f, delimiter="\t")]
    return rows, times


def column_keys(header):
    keys = {}
    for heading, suffix in COLUMNS:
        match = [h for h in header if h.endswith(suffix)]
        if match:
            keys[heading] = match[0]
    return keys


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def summarize(rows, times):
    keys = column_keys(rows[0].keys()) if rows else {}
    kernels = collections.OrderedDict()
    for i, row in enumerate(rows):
        name = row[NAME_COLUMN]
        ms = times[i] if i < len(times) else 0.0
        k = kernels.setdefault(
            name,
            {
                "n": 0,
                "ms": 0.0,
                "sums": collections.defaultdict(float),
                "weights": collections.defaultdict(float),
            },
        )
        k["n"] += 1
        k["ms"] += ms
        for heading, key in keys.items():
            v = number(row.get(key))
            if v is not None:
                # time-weighted, so long dispatches speak for a kernel; unweighted where no times exported
                w = ms if times else 1.0
                k["sums"][heading] += v * w
                k["weights"][heading] += w
    total = sum(k["ms"] for k in kernels.values()) or 1.0
    headings = list(keys)
    print(
        f"{'kernel':28s} {'n':>5s} {'ms':>9s} {'%':>5s} {'us each':>9s} "
        + " ".join(f"{h:>6s}" for h in headings)
    )
    for name, k in sorted(kernels.items(), key=lambda kv: -kv[1]["ms"]):
        cells = []
        for h in headings:
            w = k["weights"][h]
            cells.append(f"{k['sums'][h] / w:6.1f}" if w else f"{'-':>6s}")
        each = 1e3 * k["ms"] / k["n"] if k["n"] else 0.0
        print(
            f"{name[:28]:28s} {k['n']:5d} {k['ms']:9.3f} {100 * k['ms'] / total:5.1f} {each:9.1f} "
            + " ".join(cells)
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--export", help="an existing auto-export directory to read instead of tracing"
    )
    p.add_argument("--exe", help="the binary to trace")
    p.add_argument("--args", default="", help="its arguments, one string")
    p.add_argument("--dir", default=".", help="its working directory")
    p.add_argument("--env", default="", help='more environment for it, as "A=1;B=2;"')
    p.add_argument("--start-after-submits", type=int, default=3)
    p.add_argument(
        "--start-after-ms", type=int, help="start on a timer instead of a submit count"
    )
    p.add_argument("--duration-ms", type=int, default=DEFAULT_DURATION_MS)
    p.add_argument(
        "--multi-pass",
        action="store_true",
        help="collect ngfx's multi-pass metrics as well",
    )
    p.add_argument(
        "--out",
        help="where the trace and its export go (default: a temporary directory)",
    )
    args = p.parse_args()
    export = args.export
    if export is None:
        if not args.exe:
            p.error("--exe or --export")
        export = args.out or tempfile.mkdtemp(prefix="vulkan-gputrace-")
        # ngfx wants its output directory to exist
        os.makedirs(export, exist_ok=True)
        run_trace(args, export)
        print(f"trace and export in {export}")
    summarize(*read_rows(export))


if __name__ == "__main__":
    main()
