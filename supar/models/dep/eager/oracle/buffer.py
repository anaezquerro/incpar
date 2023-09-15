from typing import Optional, List
from supar.models.dep.eager.oracle.node import Node

class Buffer:
    def __init__(self, items: Optional[List[Node]]):
        if items:
            self.items = items
        else:
            self.items = []

    def get(self) -> Node:
        return self.items[0]

    def remove(self) -> Node:
        return self.items.pop(0)

    def append(self, item: Node):
        self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return str(self.items)