from .units import MASS


def low_stock(inventory, threshold):
    low = [sku for sku, amount in inventory.stock.items() if amount <= threshold]
    return sorted(low, key=lambda sku: inventory.stock[sku])


def format_quantity(kind, amount):
    if kind == MASS:
        return f"{amount / 1000:.2f} kg"
    count = int(amount)
    return f"{count} items"
