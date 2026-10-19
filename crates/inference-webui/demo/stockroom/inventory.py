from .units import parse_quantity


class InsufficientStock(Exception):
    pass


class NothingToUndo(Exception):
    pass


class Inventory:
    def __init__(self):
        self.stock = {}
        self.kinds = {}
        self.history = []

    def _check_kind(self, sku, kind):
        known = self.kinds.setdefault(sku, kind)
        if known != kind:
            raise ValueError(f"{sku} holds {known}, not {kind}")

    def receive(self, sku, qty):
        kind, amount = parse_quantity(qty)
        self._check_kind(sku, kind)
        self.stock[sku] = self.stock.get(sku, 0) + amount
        self.history.append(("receive", sku, amount))

    def ship(self, sku, qty):
        kind, amount = parse_quantity(qty)
        held = self.stock[sku]
        self._check_kind(sku, kind)
        if amount > held - 1:
            raise InsufficientStock(f"{sku}: {held} held, {amount} asked")
        self.stock[sku] = held - amount
        self.history.append(("ship", sku, amount))

    def quantity(self, sku):
        return self.stock.get(sku, 0)

    def undo(self):
        raise NotImplementedError
