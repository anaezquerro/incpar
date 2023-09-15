from supar.utils.transform import Transform, Sentence
from supar.utils.field import Field
from typing import Iterable, List, Optional, Union
from supar.codelin import D_Tree

class SLDependency(Transform):

    fields = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC', 'LABEL']

    def __init__(
        self,
        encoder,
        ID: Optional[Union[Field, Iterable[Field]]] = None,
        FORM: Optional[Union[Field, Iterable[Field]]] = None,
        LEMMA: Optional[Union[Field, Iterable[Field]]] = None,
        UPOS: Optional[Union[Field, Iterable[Field]]] = None,
        XPOS: Optional[Union[Field, Iterable[Field]]] = None,
        FEATS: Optional[Union[Field, Iterable[Field]]] = None,
        HEAD: Optional[Union[Field, Iterable[Field]]] = None,
        DEPREL: Optional[Union[Field, Iterable[Field]]] = None,
        DEPS: Optional[Union[Field, Iterable[Field]]] = None,
        MISC: Optional[Union[Field, Iterable[Field]]] = None,
        LABEL: Optional[Union[Field, Iterable[Field]]] = None
    ):
        super().__init__()

        self.encoder = encoder
        self.ID = ID
        self.FORM = FORM
        self.LEMMA = LEMMA
        self.UPOS = UPOS
        self.XPOS = XPOS
        self.FEATS = FEATS
        self.HEAD = HEAD
        self.DEPREL = DEPREL
        self.DEPS = DEPS
        self.MISC = MISC
        self.LABEL = LABEL

    @property
    def src(self):
        return self.FORM, self.LEMMA, self.UPOS, self.XPOS, self.FEATS

    @property
    def tgt(self):
        return self.HEAD, self.DEPREL, self.DEPS, self.MISC, self.LABEL

    def load(
        self,
        data: str,
        **kwargs
    ):
        lines = open(data)
        index, sentence = 0, []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                sentence = SLDependencySentence(self, sentence, self.encoder, index)
                if sentence.values:
                    yield sentence
                    index += 1
                sentence = []
            else:
                sentence.append(line)



class SLDependencySentence(Sentence):
    def __init__(self, transform: SLDependency, lines: List[str], encoder, index: Optional[int] = None):
        super().__init__(transform, index)
        self.annotations = dict()
        self.values = list()

        for i, line in enumerate(lines):
            value = line.split('\t')
            if value[0].startswith('#') or not value[0].isdigit():
                self.annotations[-i-1] = line
            else:
                self.annotations[len(self.values)] = line
                self.values.append(value)

        # convert values into nodes
        tree = D_Tree.from_string('\n'.join(['\t'.join(value) for value in self.values]))

        # linearize tree
        linearized_tree = encoder.encode(tree)

        self.values = list(zip(*self.values))
        self.values[6] = tuple(map(int, self.values[6]))

        # add labels
        labels = tuple(str(label.xi) for label in linearized_tree.labels)
        self.values.append(labels)



    def __repr__(self):
        # cover the raw lines
        merged = {**self.annotations,
                  **{i: '\t'.join(map(str, line))
                     for i, line in enumerate(zip(*self.values[:-1]))}}
        return '\n'.join(merged.values()) + '\n'
