"""The slowest tests and test modules of each nextest run in a log (`local_ci.sh ... 2>&1 | tee ci.log`).

A module is the binary plus the test's module path without a trailing `tests` (`inference-nn cuda::gdn`); its time is
the sum of its tests', so a module of many medium tests shows up even when no single test does.
"""

import argparse
import collections
import re
import sys
from pathlib import Path

TOP = 15
STARTING = re.compile(r"^\s*Starting \d+ tests? across")
PROFILE = re.compile(r"via profile\.(\w+)\.")
RESULT = re.compile(
    r"^\s*(PASS|FAIL|SLOW|LEAK|TIMEOUT|SIGSEGV|SIGABRT) \[\s*([\d.]+)s\] (?:\(\s*\d+/\d+\) )?(\S+) (\S+)"
)
SUMMARY = re.compile(r"^\s*Summary \[\s*([\d.]+)s\]")


def parse(lines):
    """One dict per nextest run: profile, wall seconds, and (binary, test, seconds) per finished test."""
    runs = []
    for line in lines:
        if STARTING.match(line):
            m = PROFILE.search(line)
            runs.append(
                {"profile": m.group(1) if m else "default", "wall": None, "tests": []}
            )
        elif runs and (m := RESULT.match(line)) and m.group(1) != "SLOW":
            runs[-1]["tests"].append((m.group(3), m.group(4), float(m.group(2))))
        elif runs and (m := SUMMARY.match(line)):
            runs[-1]["wall"] = float(m.group(1))
    return runs


def modules(tests):
    total = collections.Counter()
    count = collections.Counter()
    for binary, name, secs in tests:
        path = name.split("::")[:-1]
        if path and path[-1] == "tests":
            path.pop()
        key = f"{binary} {'::'.join(path) or name}"
        total[key] += secs
        count[key] += 1
    return [(k, total[k], count[k]) for k, _ in total.most_common()]


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("log")
    ap.add_argument("--top", type=int, default=TOP)
    args = ap.parse_args(argv)
    for run in parse(Path(args.log).read_text(errors="replace").splitlines()):
        if not run["tests"]:
            continue
        cpu = sum(t for *_, t in run["tests"])
        print(
            f"profile {run['profile']}: {len(run['tests'])} tests, wall {run['wall']} s, test time {cpu:.1f} s"
        )
        print("  slowest tests:")
        for binary, name, secs in sorted(run["tests"], key=lambda t: -t[2])[: args.top]:
            print(f"    {secs:7.2f}  {binary} {name}")
        print("  slowest modules (sum, tests):")
        for key, secs, n in modules(run["tests"])[: args.top]:
            print(f"    {secs:7.2f}  {n:4d}  {key}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
