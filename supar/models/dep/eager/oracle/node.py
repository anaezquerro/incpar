from typing import List

class Node:

    def __init__(self, ID: int, FORM: str, UPOS: str, HEAD: int, DEPREL: str, is_root: bool = False):
        self.ID = ID
        self.FORM = FORM
        self.UPOS = UPOS
        self.HEAD = HEAD
        self.DEPREL = DEPREL
        self.is_root = is_root

    def __str__(self):
        return f'Node(ID={self.ID}, FORM={self.FORM}, UPOS={self.UPOS}, HEAD={self.HEAD})'

    def __repr__(self):
        return f'Node(ID={self.ID}, FORM={self.FORM}, UPOS={self.UPOS}, HEAD={self.HEAD})'

    @classmethod
    def from_conllu(cls, conll: str):
        ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = conll.split('\t')
        return Node(int(ID), FORM, UPOS, int(HEAD), DEPREL)

    @classmethod
    def create_root(cls, token: str):
        return Node(0, token, token, 0, token, is_root=True)

    @classmethod
    def create_eos(cls, position: int, token: str):
        return Node(position, token, token, 0, token, is_root=False)

    def coverage(self) -> range:
        limits = sorted([self.ID, self.HEAD])
        return range(*limits)

    def isprojective(heads: List[int]):
        pairs = [(h, d) for d, h in enumerate(heads, 1) if h >= 0]
        for i, (hi, di) in enumerate(pairs):
            for hj, dj in pairs[i + 1:]:
                (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
                if li <= hj <= ri and hi == dj:
                    print('1')
                    return False
                if lj <= hi <= rj and hj == di:
                    print('2')
                    return False
                if (li < lj < ri or li < rj < ri) and (li - lj) * (ri - rj) > 0:
                    print('3')
                    print(di, hi, dj, hj)
                    return False
        return True

def isprojective(heads: List[int]):
    pairs = [(h, d) for d, h in enumerate(heads, 1) if h >= 0]
    for i, (hi, di) in enumerate(pairs):
        for hj, dj in pairs[i + 1:]:
            (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
            if li <= hj <= ri and hi == dj:
                print('1')
                return False
            if lj <= hi <= rj and hj == di:
                print('2')
                return False
            if (li < lj < ri or li < rj < ri) and (li - lj) * (ri - rj) > 0:
                print('3')
                print(di, hi, dj, hj)
                return False
    return True



