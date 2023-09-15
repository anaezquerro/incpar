from typing import Union, List, Optional
from supar.models.dep.eager.oracle.node import Node

class Stack:
    def __init__(self, items: Optional[List[Node]] = None):
        if items:
            self.items = items
        else:
            self.items = []

    def pop(self) -> Node:
        return self.items.pop(-1)

    def push(self, item: Node):
        self.items.append(item)

    def get(self) -> Node:
        return self.items[-1]

    def __repr__(self):
        return repr(self.items)



