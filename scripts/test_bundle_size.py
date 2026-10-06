import unittest

from bundle_size import CHANGE_BYTES, TRACKED_MIN, build_key, regressions, sections, shrinks

SIZE_OUTPUT = """target/bundle/libinference_ffi.so  :
section                size      addr
.text              51000000   1000000
.rodata             9000000  60000000
.nv_fatbin         55000000  70000000
.comment                 80         0
Total             115000080
"""


class BundleSizeTest(unittest.TestCase):
    def test_size_output_parses_into_sections(self):
        self.assertEqual(
            sections(SIZE_OUTPUT),
            {".text": 51000000, ".rodata": 9000000, ".nv_fatbin": 55000000, ".comment": 80},
        )

    def test_growth_fails_only_past_both_limits(self):
        base = {".text": 100_000_000, ".tiny": 2_000_000}
        # 1% of .text is 1 MB: under it passes, over it fails
        self.assertEqual(regressions(base, {".text": 100_900_000, ".tiny": 2_000_000}), [])
        self.assertEqual(regressions(base, {".text": 101_100_000, ".tiny": 2_000_000}), [".text"])
        # 10% of a small section, but under the byte limit
        self.assertEqual(regressions(base, {".text": 100_000_000, ".tiny": 2_000_000 + CHANGE_BYTES}), [])

    def test_a_new_large_section_is_growth(self):
        self.assertEqual(regressions({".text": 100}, {".text": 100, ".nv_fatbin": TRACKED_MIN}), [".nv_fatbin"])

    def test_a_section_crossing_the_tracking_minimum_counts_from_its_size(self):
        base = {".data": TRACKED_MIN - 10_000}
        self.assertEqual(regressions(base, {".data": TRACKED_MIN + 10_000}), [])

    def test_shrinking_passes_and_is_reported_past_the_limits(self):
        base = {".text": 100_000_000, ".gone": 5_000_000}
        current = {".text": 90_000_000}
        self.assertEqual(regressions(base, current), [])
        self.assertEqual(shrinks(base, current), [".text", ".gone"])

    def test_compute_caps_compare_with_or_without_the_dot(self):
        self.assertEqual(build_key("8.6", "13.0"), build_key("86", "13.0"))

    def test_an_arch_list_keys_the_same_in_any_order(self):
        self.assertEqual(build_key("9.0,80, 86", "13.0"), build_key("80,86,90", "13.0"))
        self.assertEqual(build_key("80,86", "13.0")["compute_cap"], "80,86")
        self.assertEqual(build_key("sm_90a;8.6 80,", "13.0"), build_key("80,86,90", "13.0"))


if __name__ == "__main__":
    unittest.main()
