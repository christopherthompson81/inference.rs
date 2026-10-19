# stockroom

A tiny warehouse stock tracker.

- Quantities are strings with an optional unit: `"3"` (items), `"2 dozen"` (24 items), `"2.5kg"`, `"500g"`. Mass is
  stored in grams, counts in items. A SKU holds either mass or items, never both.
- `Inventory.receive(sku, qty)` adds stock; `Inventory.ship(sku, qty)` removes it, raising `InsufficientStock` when
  the SKU doesn't hold enough (an unknown SKU holds nothing). Shipping exactly what is held is allowed.
- `Inventory.undo()` reverses the most recent receive or ship that hasn't been undone, and raises `NothingToUndo`
  when there is none.
- `report.low_stock(inventory, threshold)` lists the SKUs holding less than `threshold` (same units as the SKU),
  sorted by SKU.
- `report.format_quantity(sku_kind, amount)` renders grams as kilograms with 3 decimals (`"1.250 kg"`) and items as
  an integer (`"12 items"`, `"1 item"`).
