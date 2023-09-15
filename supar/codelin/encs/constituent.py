from src.models.linearized_tree import LinearizedTree
from src.encs.enc_const import *
from src.utils.extract_feats import extract_features_const
from src.utils.constants import C_INCREMENTAL_ENCODING, C_ABSOLUTE_ENCODING, C_RELATIVE_ENCODING, C_DYNAMIC_ENCODING

import stanza.pipeline
from supar.codelin.models.const_tree import C_Tree


## Encoding and decoding

def encode_constituent(in_path, out_path, encoding_type, separator, unary_joiner, features):
    '''
    Encodes the selected file according to the specified parameters:
    :param in_path: Path of the file to be encoded
    :param out_path: Path where to write the encoded labels
    :param encoding_type: Encoding used
    :param separator: string used to separate label fields
    :param unary_joiner: string used to separate nodes from unary chains
    :param features: features to add as columns to the labels file
    '''

    if encoding_type == C_ABSOLUTE_ENCODING:
            encoder = C_NaiveAbsoluteEncoding(separator, unary_joiner)
    elif encoding_type == C_RELATIVE_ENCODING:
            encoder = C_NaiveRelativeEncoding(separator, unary_joiner)
    elif encoding_type == C_DYNAMIC_ENCODING:
            encoder = C_NaiveDynamicEncoding(separator, unary_joiner)
    elif encoding_type == C_INCREMENTAL_ENCODING:
            encoder = C_NaiveIncrementalEncoding(separator, unary_joiner)
    else:
        raise Exception("Unknown encoding type")

    # build feature index dictionary
    f_idx_dict = {}
    if features:
        if features == ["ALL"]:
            features = extract_features_const(in_path)
        i=0
        for f in features:
            f_idx_dict[f]=i
            i+=1

    file_out = open(out_path, "w")
    file_in = open(in_path, "r")

    tree_counter = 0
    labels_counter = 0
    label_set = set()

    for line in file_in:
        line = line.rstrip()
        tree = C_Tree.from_string(line)
        linearized_tree = encoder.encode(tree)
        file_out.write(linearized_tree.to_string(f_idx_dict))
        file_out.write("\n")
        tree_counter += 1
        labels_counter += len(linearized_tree)
        for lbl in linearized_tree.get_labels():
            label_set.add(str(lbl))   
    
    return labels_counter, tree_counter, len(label_set)

def decode_constituent(in_path, out_path, encoding_type, separator, unary_joiner, conflicts, nulls, postags, lang):
    '''
    Decodes the selected file according to the specified parameters:
    :param in_path: Path of the labels file to be decoded
    :param out_path: Path where to write the decoded tree
    :param encoding_type: Encoding used
    :param separator: string used to separate label fields
    :param unary_joiner: string used to separate nodes from unary chains
    :param conflicts: conflict resolution heuristics to apply
    '''

    if encoding_type == C_ABSOLUTE_ENCODING:
            decoder = C_NaiveAbsoluteEncoding(separator, unary_joiner)
    elif encoding_type == C_RELATIVE_ENCODING:
            decoder = C_NaiveRelativeEncoding(separator, unary_joiner)
    elif encoding_type == C_DYNAMIC_ENCODING:
            decoder = C_NaiveDynamicEncoding(separator, unary_joiner)
    elif encoding_type == C_INCREMENTAL_ENCODING:
            decoder = C_NaiveIncrementalEncoding(separator, unary_joiner)
    else:
        raise Exception("Unknown encoding type")

    if postags:
        stanza.download(lang=lang)
        nlp = stanza.Pipeline(lang=lang, processors='tokenize, pos')

    f_in = open(in_path)
    f_out = open(out_path,"w+")
    
    tree_string   = ""
    labels_counter = 0
    tree_counter = 0

    for line in f_in:
        if line == "\n":
            tree_string = tree_string.rstrip()
            current_tree = LinearizedTree.from_string(tree_string, mode="CONST", separator=separator, unary_joiner=unary_joiner)
            
            if postags:
                c_tags = nlp(current_tree.get_sentence())
                current_tree.set_postags([word.pos for word in c_tags])

            decoded_tree = decoder.decode(current_tree)
            decoded_tree = decoded_tree.postprocess_tree(conflicts, nulls)

            f_out.write(str(decoded_tree).replace('\n','')+'\n')
            tree_string   = ""
            tree_counter+=1
        tree_string += line
        labels_counter += 1
    
    return tree_counter, labels_counter