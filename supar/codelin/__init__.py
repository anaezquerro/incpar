from .encs.enc_deps import D_NaiveAbsoluteEncoding, D_NaiveRelativeEncoding, D_PosBasedEncoding, D_BrkBasedEncoding, D_Brk2PBasedEncoding
from .encs.enc_const import C_NaiveAbsoluteEncoding, C_NaiveRelativeEncoding
from .utils.constants import D_2P_GREED, D_2P_PROP

# import structures for encoding/decoding
from .models.const_label import C_Label
from .models.const_tree import C_Tree
from .models.linearized_tree import LinearizedTree
from .models.deps_label import D_Label
from .models.deps_tree import D_Tree

LABEL_SEPARATOR = '€'
UNARY_JOINER = '@'

def get_con_encoder(encoding: str, sep: str = LABEL_SEPARATOR, unary_joiner: str = UNARY_JOINER):
    if encoding == 'abs':
        return C_NaiveAbsoluteEncoding(sep, unary_joiner)
    elif encoding == 'rel':
        return C_NaiveRelativeEncoding(sep, unary_joiner)
    return NotImplementedError

def get_dep_encoder(encoding: str, sep: str, displacement: bool = False):
    if encoding == 'abs':
        return D_NaiveAbsoluteEncoding(sep)
    elif encoding == 'rel':
        return D_NaiveRelativeEncoding(sep, hang_from_root=False)
    elif encoding == 'pos':
        return D_PosBasedEncoding(sep)
    elif encoding == '1p':
        return D_BrkBasedEncoding(sep, displacement)
    elif encoding == '2p':
        return D_Brk2PBasedEncoding(sep, displacement, D_2P_PROP)
    return NotImplementedError

