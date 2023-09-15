from typing import Union, Iterable, Optional, Set
from supar.utils.field import Field
from supar.utils.transform import Sentence, Transform
from supar.codelin import C_Tree, UNARY_JOINER, LABEL_SEPARATOR
import nltk, re


class SLConstituent(Transform):
    fields = ['WORD', 'POS', 'COMMON', 'ANCESTOR', 'TREE']

    def __init__(
        self,
        encoder,
        WORD: Union[Field, Iterable[Field]],
        POS: Union[Field, Iterable[Field]],
        COMMON: Union[Field, Iterable[Field]],
        ANCESTOR: Union[Field, Iterable[Field]],
        TREE: Union[Field, Iterable[Field]]
    ):
        super().__init__()

        self.WORD = WORD
        self.POS = POS
        self.COMMON = COMMON
        self.ANCESTOR = ANCESTOR
        self.TREE = TREE
        self.encoder = encoder


    @property
    def src(self):
        return self.WORD, self.POS

    @property
    def tgt(self):
        return self.COMMON, self.ANCESTOR, self.TREE
    def load(
        self,
        data: Union[str, Iterable],
        **kwargs
    ) -> Iterable[Sentence]:
        lines = open(data)
        index = 0
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                sentence = SLConstituentSentence(self, line, self.encoder, index)
                yield sentence
                index += 1


def get_nodes(tree: nltk.Tree):
    nodes = {tree.label()}
    for subtree in tree:
        if isinstance(subtree[0], nltk.Tree):
            nodes = {*nodes, *get_nodes(subtree)}
    return nodes

def remove_doubles(tree: nltk.Tree):
    for i, subtree in enumerate(tree):
        if isinstance(subtree, str) and len(tree) > 1:
            tree.pop(i)
        elif isinstance(subtree, nltk.Tree):
            remove_doubles(subtree)

class SLConstituentSentence(Sentence):

    def __init__(self, transform: SLConstituent, line: str, encoder, index: Optional[int] = None):
        super().__init__(transform, index)

        # get nodes of the tree
        gold = nltk.Tree.fromstring(line)
        nodes = ''.join(get_nodes(gold))
        assert (encoder.separator not in nodes) and (encoder.unary_joiner not in nodes)

        # create linearized tree
        tree = C_Tree.from_string(line)
        linearized_tree = encoder.encode(tree)
        self.annotations = []
        commons = list(map(lambda x: repr(x).split(encoder.separator)[0], linearized_tree.labels))
        ancestors = list(map(lambda x: encoder.separator.join(repr(x).split(encoder.separator)[1:]), linearized_tree.labels))

        _, postags = zip(*gold.pos())
        self.values = [
            linearized_tree.words,
            postags,
            commons, ancestors
        ]
        self.values.append(
            nltk.Tree.fromstring(line)
        )
    def __repr__(self):
        remove_doubles(self.values[-1])
        return re.sub(' +', ' ', str(self.values[-1]).replace('\n', '').replace('\t', ''))
