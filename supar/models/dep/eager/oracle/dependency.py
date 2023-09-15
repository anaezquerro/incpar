from typing import Optional, List
from supar.models.dep.eager.oracle.node import Node

class Dependency:
    def __init__(self, dependent_id: int, head_id: str, deprel: str):
        self.dependent_id = dependent_id
        self.head_id = head_id
        self.deprel = deprel

    def __repr__(self):
        return self.toconll()

    def toconll(self):
        return '\t'.join([
            str(self.dependent_id), str(self.head_id), self.deprel
        ])

class Transition:
    def __init__(self, type: str, stack_top: Node, buffer_front: Node, deprel: str):
        self.type = type
        self.stack_top = stack_top
        self.buffer_front = buffer_front
        self.deprel = deprel

    def __repr__(self):
        return '\t'.join((str(self.stack_top.FORM), str(self.buffer_front.FORM), self.type, self.deprel))




