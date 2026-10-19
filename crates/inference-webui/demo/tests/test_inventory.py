import unittest

from stockroom import InsufficientStock, Inventory, NothingToUndo


class Shipping(unittest.TestCase):
    def setUp(self):
        self.inventory = Inventory()
        self.inventory.receive("bolts", "2 dozen")
        self.inventory.receive("flour", "1.5kg")

    def test_receive_adds_up(self):
        self.inventory.receive("bolts", "6")
        self.assertEqual(self.inventory.quantity("bolts"), 30)

    def test_ship_removes_stock(self):
        self.inventory.ship("flour", "500g")
        self.assertEqual(self.inventory.quantity("flour"), 1000)

    def test_shipping_everything_held_is_allowed(self):
        self.inventory.ship("bolts", "24")
        self.assertEqual(self.inventory.quantity("bolts"), 0)

    def test_shipping_too_much_fails_and_changes_nothing(self):
        with self.assertRaises(InsufficientStock):
            self.inventory.ship("bolts", "25")
        self.assertEqual(self.inventory.quantity("bolts"), 24)

    def test_unknown_skus_hold_nothing(self):
        with self.assertRaises(InsufficientStock):
            self.inventory.ship("nails", "1")
        self.assertEqual(self.inventory.quantity("nails"), 0)

    def test_units_cannot_mix(self):
        with self.assertRaises(ValueError):
            self.inventory.receive("bolts", "1kg")


class Undo(unittest.TestCase):
    def setUp(self):
        self.inventory = Inventory()
        self.inventory.receive("bolts", "10")
        self.inventory.ship("bolts", "4")

    def test_undo_reverses_the_last_ship(self):
        self.inventory.undo()
        self.assertEqual(self.inventory.quantity("bolts"), 10)

    def test_undo_walks_back_through_history(self):
        self.inventory.undo()
        self.inventory.undo()
        self.assertEqual(self.inventory.quantity("bolts"), 0)

    def test_undo_with_nothing_left_raises(self):
        self.inventory.undo()
        self.inventory.undo()
        with self.assertRaises(NothingToUndo):
            self.inventory.undo()

    def test_a_failed_ship_is_not_undone(self):
        with self.assertRaises(InsufficientStock):
            self.inventory.ship("bolts", "100")
        self.inventory.undo()
        self.assertEqual(self.inventory.quantity("bolts"), 10)
