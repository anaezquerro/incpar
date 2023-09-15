from supar.utils.transform import Transform, Sentence
from supar.utils.field import Field
from typing import Iterable, Union, Optional, List
from supar.models.dep.eager.oracle.arceager import ArcEagerEncoder
from supar.models.dep.eager.oracle.node import Node

class ArcEagerTransform(Transform):

    fields = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC', 'STACK_TOP', 'BUFFER_FRONT', 'TRANSITION', 'TREL']

    def __init__(
        self,
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
        STACK_TOP: Optional[Union[Field, Iterable[Field]]] = None,
        BUFFER_FRONT: Optional[Union[Field, Iterable[Field]]] = None,
        TRANSITION: Optional[Union[Field, Iterable[Field]]] = None,
        TREL: Optional[Union[Field, Iterable[Field]]] = None
    ):
        super().__init__()

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
        self.STACK_TOP = STACK_TOP
        self.BUFFER_FRONT = BUFFER_FRONT
        self.TRANSITION = TRANSITION
        self.TREL = TREL

    @property
    def src(self):
        return self.FORM, self.LEMMA, self.UPOS, self.XPOS, self.FEATS

    @property
    def tgt(self):
        return self.HEAD, self.DEPREL, self.DEPS, self.MISC, self.STACK_TOP, self.BUFFER_FRONT, self.TRANSITION, self.TREL

    def load(
        self,
        data: Union[str, Iterable],
        **kwargs
    ):
        lines = open(data)
        index, sentence = 0, []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                sentence = ArcEagerSentence(self, sentence, index)
                yield sentence
                index += 1
                sentence = []
            else:
                sentence.append(line)



class ArcEagerSentence(Sentence):
    def __init__(self, transform: ArcEagerEncoder, lines: List[str], index: Optional[int] = None):
        super().__init__(transform, index)
        self.values = list()
        self.annotations = dict()

        for i, line in enumerate(lines):
            value = line.split('\t')
            if value[0].startswith('#') or not value[0].isdigit():
                self.annotations[-i-1] = line
            else:
                self.annotations[len(self.values)] = line
                self.values.append(value)

        nodes = [Node.from_conllu('\t'.join(value)) for value in self.values]

        algorithm = ArcEagerEncoder(bos=transform.FORM[0].bos, eos=transform.FORM[0].eos)
        transitions = algorithm.encode(nodes.copy())


        stack_top, buffer_front, transition, trel = zip(
            *[(transition.stack_top.ID, transition.buffer_front.ID, transition.type, transition.deprel)
              for transition in transitions])

        self.values = list(zip(*self.values))
        self.values[6] = tuple(map(int, self.values[6]))
        self.values += [stack_top, buffer_front, transition, trel]

    def __repr__(self):
        # cover the raw lines
        merged = {**self.annotations,
                  **{i: '\t'.join(map(str, line))
                     for i, line in enumerate(zip(*self.values[:-4]))}}
        return '\n'.join(merged.values()) + '\n'