import unittest

from stockroom.units import COUNT, MASS, parse_quantity


class ParseQuantity(unittest.TestCase):
    def test_bare_numbers_are_items(self):
        self.assertEqual(parse_quantity("3"), (COUNT, 3))

    def test_dozens(self):
        self.assertEqual(parse_quantity("2 dozen"), (COUNT, 24))

    def test_kilograms_are_stored_as_grams(self):
        self.assertEqual(parse_quantity("2.5kg"), (MASS, 2500))

    def test_grams(self):
        self.assertEqual(parse_quantity("500 g"), (MASS, 500))

    def test_rejects_unknown_units(self):
        with self.assertRaises(ValueError):
            parse_quantity("3 crates")
