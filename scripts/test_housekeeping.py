import unittest

from dup_scan import normalize, scan, shared_spans
from slow_tests import modules, parse

BODY = "\n".join(f"let v{i} = f({i});" for i in range(14))


class DupScanTest(unittest.TestCase):
    def test_normalize_drops_comments_blank_and_brace_only_lines(self):
        lines = normalize("fn a() {\n    // note\n\n    let  x =   1;\n}\n")
        self.assertEqual(lines, [(1, "fn a() {"), (4, "let x = 1;")])

    def test_a_block_in_two_files_is_counted_once_per_pair(self):
        a = normalize("fn a() {}\n" + BODY)
        b = normalize(BODY + "\nfn b() {}")
        dup, pairs = scan({"a": a, "b": b, "c": normalize("fn c() {}")}, 12)
        self.assertEqual(len(dup["a"]), 14)
        self.assertNotIn("c", dup)
        self.assertEqual(pairs, {("a", "b"): 3})

    def test_shared_spans_merge_consecutive_windows_into_line_ranges(self):
        a = normalize("fn a() {}\n" + BODY)
        b = normalize(BODY)
        self.assertEqual(shared_spans(a, b, 12), [((2, 15), (1, 14))])


LOG = """\
    Starting 3 tests across 2 binaries (1 test skipped)
        PASS [   0.500s] (1/3) inference-nn cuda::gdn::tests::decode
        SLOW [> 60.000s] (2/3) inference::integration kokoro_tiny::speaks
        PASS [  70.000s] (2/3) inference::integration kokoro_tiny::speaks
        FAIL [   1.250s] (3/3) inference-nn cuda::gdn::tests::prefill
     Summary [  71.000s] 3 tests run: 2 passed, 1 failed
    Starting 1 test across 1 binary (2 tests skipped, including 2 tests via profile.cuda.default-filter)
        PASS [   2.000s] (1/1) inference-ffi::integration engine_abi::loads
     Summary [   2.000s] 1 test run: 1 passed
"""


class SlowTestsTest(unittest.TestCase):
    def test_each_run_keeps_its_profile_wall_and_finished_tests(self):
        runs = parse(LOG.splitlines())
        self.assertEqual(
            [(r["profile"], r["wall"], len(r["tests"])) for r in runs],
            [("default", 71.0, 3), ("cuda", 2.0, 1)],
        )

    def test_modules_sum_their_tests_without_the_tests_segment(self):
        runs = parse(LOG.splitlines())
        self.assertEqual(
            modules(runs[0]["tests"]),
            [
                ("inference::integration kokoro_tiny", 70.0, 1),
                ("inference-nn cuda::gdn", 1.75, 2),
            ],
        )


if __name__ == "__main__":
    unittest.main()
