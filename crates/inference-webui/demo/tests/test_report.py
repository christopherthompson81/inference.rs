import unittest

from stockroom import Inventory
from stockroom.report import format_quantity, low_stock
from stockroom.units import COUNT, MASS


class LowStock(unittest.TestCase):
    def test_lists_skus_below_the_threshold_sorted_by_sku(self):
        inventory = Inventory()
        inventory.receive("washers", "3")
        inventory.receive("anchors", "8")
        inventory.receive("bolts", "1")
        inventory.receive("nuts", "5")
        self.assertEqual(low_stock(inventory, 5), ["bolts", "washers"])


class FormatQuantity(unittest.TestCase):
    def test_grams_render_as_kilograms(self):
        self.assertEqual(format_quantity(MASS, 1250), "1.250 kg")

    def test_items(self):
        self.assertEqual(format_quantity(COUNT, 12), "12 items")

    def test_one_item(self):
        self.assertEqual(format_quantity(COUNT, 1), "1 item")
