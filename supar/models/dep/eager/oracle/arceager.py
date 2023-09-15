from supar.models.dep.eager.oracle.stack import Stack
from supar.models.dep.eager.oracle.buffer import Buffer
from supar.models.dep.eager.oracle.node import Node
from supar.models.dep.eager.oracle.dependency import Transition
from typing import List, Dict

class ArcEagerEncoder:
    def __init__(self, bos: str, eos: str):
        self.stack = None
        self.buffer = None
        self.transitions = []
        self.dependencies = []
        self.nodes_assigned = []
        self.bos, self.eos = bos, eos
        self.shift_token = '<shift>'
        self.reduce_token = '<reduce>'
        self.n = 0

    def left_arc(self):
        # head = buffer_front, dependent = stack_top (stack_top <- buffer_front)
        stack_top = self.stack.pop()
        buffer_front = self.buffer.get()
        self.transitions.append(
            Transition(type='left-arc', stack_top=stack_top, buffer_front=buffer_front, deprel=stack_top.DEPREL)
        )
        self.dependencies.append(
            Node(ID=stack_top.ID, FORM=stack_top.FORM, UPOS=stack_top.UPOS, HEAD=buffer_front.ID, DEPREL=stack_top.DEPREL)
        )
        self.nodes_assigned.append(stack_top.ID)
        return self.transitions

    def right_arc(self):
        # head = stack_top, dependent = buffer_front (stack_top -> buffer_front)
        stack_top = self.stack.get()
        buffer_front = self.buffer.remove()
        self.stack.push(buffer_front)
        self.transitions.append(
            Transition(type='right-arc', stack_top=stack_top, buffer_front=buffer_front, deprel=buffer_front.DEPREL)
        )
        self.dependencies.append(
            Node(ID=buffer_front.ID, FORM=buffer_front.FORM, UPOS=buffer_front.UPOS, HEAD=stack_top.ID, DEPREL=buffer_front.DEPREL)
        )
        self.nodes_assigned.append(buffer_front.ID)
        return self.transitions

    def shift(self):
        front_item = self.buffer.remove()
        stack_top = self.stack.get()
        self.stack.push(front_item)
        self.transitions.append(
            Transition(type='shift', stack_top=stack_top, buffer_front=front_item, deprel=front_item.DEPREL))
        return self.transitions

    def reduce(self):
        stack_top = self.stack.get()
        buffer_front = self.buffer.get() if len(self.buffer) > 0 else Node.create_eos(self.n, self.eos)
        self.stack.pop()
        self.transitions.append(
            Transition(type='reduce', stack_top=stack_top, buffer_front=buffer_front, deprel=stack_top.DEPREL)
        )
        return self.transitions

    def next_action(self):
        stack_top = self.stack.get()
        try:
            buffer_front = self.buffer.get()
        except IndexError:
            if stack_top.ID in self.nodes_assigned:
                return self.reduce()
            return None

        if buffer_front.ID == stack_top.HEAD:
            return self.left_arc()
        elif (buffer_front.ID not in self.nodes_assigned) and (stack_top.ID == buffer_front.HEAD):
            return self.right_arc()
        elif (stack_top.ID in self.nodes_assigned) and (buffer_front.HEAD in [node.ID for node in self.stack.items]):
            return self.reduce()
        elif (stack_top.ID in self.nodes_assigned) and (buffer_front.ID in [node.HEAD for node in self.stack.items]):
            return self.reduce()
        else:
            return self.shift()

    def encode(self, sentence: List[Node]):
        # create stack and buffer
        self.stack = Stack([Node.create_root(self.bos)])
        self.buffer = Buffer(sentence.copy())
        self.n = len(sentence)

        # reset
        self.transitions, self.dependencies = [], []

        next_action = self.next_action()
        while next_action:
            next_action = self.next_action()

        # remove values
        self.dependencies = sorted(self.dependencies, key=lambda dep: dep.ID)
        return self.transitions



class ArcEagerDecoder:
    def __init__(self, sentence: List[Node], bos: str, eos: str, unk: str):
        self.sentence = sentence.copy()
        self.decoded_nodes = [Node(ID=node.ID, FORM=node.FORM, UPOS=node.UPOS, HEAD=0, DEPREL=unk) for node in sentence]
        self.transitions = list()
        self.nodes_assigned = list()
        self.stack = Stack([Node.create_root(bos)])
        self.buffer = Buffer(sentence.copy())
        self.bos, self.eos, self.unk = bos, eos, unk
        self.shift_token, self.reduce_token = '<shift>',  '<reduce>'

    def left_arc(self, deprel: str):
        # head = buffer_front, dependent = stack_top (stack_top <- buffer_front)
        stack_top = self.stack.pop()
        buffer_front = self.buffer.get()
        self.nodes_assigned.append(stack_top.ID)
        self.transitions.append(
            Transition(type='left-arc', stack_top=stack_top, buffer_front=buffer_front, deprel=deprel)
        )
        self.decoded_nodes[stack_top.ID - 1].HEAD = buffer_front.ID
        self.decoded_nodes[stack_top.ID - 1].DEPREL = deprel
        # get next states
        stack_top = self.stack.get()
        buffer_front = self.buffer.get() if len(self.buffer) > 0 else Node.create_eos(len(self.sentence), self.eos)
        return stack_top, buffer_front

    def right_arc(self, deprel: str):
        # head = stack_top, dependent = buffer_front (stack_top -> buffer_front)
        stack_top = self.stack.get()
        buffer_front = self.buffer.remove()
        self.stack.push(buffer_front)
        self.transitions.append(
            Transition(type='right-arc', stack_top=stack_top, buffer_front=buffer_front, deprel=deprel)
        )
        self.nodes_assigned.append(buffer_front.ID)
        self.decoded_nodes[buffer_front.ID - 1].HEAD = stack_top.ID
        self.decoded_nodes[buffer_front.ID - 1].DEPREL = deprel

        # get next states
        stack_top = self.stack.get()
        buffer_front = self.buffer.get() if len(self.buffer) > 0 else Node.create_eos(len(self.sentence), self.eos)
        return stack_top, buffer_front


    def shift(self, deprel):
        front_item = self.buffer.remove()
        stack_top = self.stack.get()
        self.stack.push(front_item)
        self.transitions.append(
            Transition(type='shift', stack_top=stack_top, buffer_front=front_item, deprel=deprel))
        # get next states
        stack_top = self.stack.get()
        buffer_front = self.buffer.get() if len(self.buffer) > 0 else Node.create_eos(len(self.sentence), self.eos)
        return stack_top, buffer_front

    def reduce(self, deprel):
        stack_top = self.stack.get()
        try:
            buffer_front = self.buffer.get()
        except IndexError:
            buffer_front = Node.create_eos(len(self.sentence), self.eos)
        self.stack.pop()
        self.transitions.append(
            Transition(type='reduce', stack_top=stack_top, buffer_front=buffer_front, deprel=deprel)
        )
        # get next states
        stack_top = self.stack.get()
        buffer_front = self.buffer.get() if len(self.buffer) > 0 else Node.create_eos(len(self.sentence), self.eos)
        return stack_top, buffer_front

    def apply_transition(self, transitions: List[str], deprel: str):
        stack_top = self.stack.get()
        try:
            buffer_front = self.buffer.get()
        except IndexError:
            if stack_top.ID in self.nodes_assigned:
                self.reduce(deprel)
            return None

        for transition in transitions:
            if (transition == 'left-arc') and (stack_top.ID not in self.nodes_assigned) and (not stack_top.is_root):
                return self.left_arc(deprel)
            if (transition == 'right-arc') and (buffer_front.ID not in self.nodes_assigned):
                return self.right_arc(deprel)
            if (transition == 'reduce') and (stack_top.ID in self.nodes_assigned):
                return self.reduce(deprel)
            return self.shift(deprel)

    def apply(self, transition: str, deprel: str):
        if transition == 'left-arc':
            return self.left_arc(deprel)
        if transition == 'right-arc':
            return self.right_arc(deprel)
        if transition == 'reduce':
            return self.reduce(deprel)
        if transition == 'shift':
            return self.shift(deprel)

    def decode_sentence(self, transitions: List[List[str]], deprels: List[str]):
        for transition_ops, deprel in zip(transitions, deprels):
            info = self.apply_transition(transition_ops, deprel)
            if info is None:
                break
        self.postprocess()
        return self.decoded_nodes

    def postprocess(self):
        # check if there are more than one node with root head
        roots = sum([node.HEAD == 0 for node in self.decoded_nodes])
        if roots > 1:
            # get leftmost root
            for node in self.decoded_nodes:
                if node.HEAD == 0:
                    root = node.ID
            for node in self.decoded_nodes[root:]:
                if node.HEAD == 0:
                    node.HEAD = root
