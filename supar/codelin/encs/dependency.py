import stanza
from supar.codelin.models.linearized_tree import LinearizedTree
from supar.codelin.models.deps_label import D_Label
from supar.codelin.utils.extract_feats import extract_features_deps
from supar.codelin.encs.enc_deps import *
from supar.codelin.utils.constants import *
from supar.codelin.models.deps_tree import D_Tree

# Encoding
def encode_dependencies(in_path, out_path, encoding_type, separator, displacement, planar_alg, root_enc, features):
    '''
    Encodes the selected file according to the specified parameters:
    :param in_path: Path of the file to be encoded
    :param out_path: Path where to write the encoded labels
    :param encoding_type: Encoding used
    :param separator: string used to separate label fields
    :param displacement: boolean to indicate if use displacement in bracket based encodings
    :param planar_alg: string used to choose the plane separation algorithm
    :param features: features to add as columns to the labels file
    '''

    # Create the encoder
    if encoding_type == D_ABSOLUTE_ENCODING:
            encoder = D_NaiveAbsoluteEncoding(separator)
    elif encoding_type == D_RELATIVE_ENCODING:
            encoder = D_NaiveRelativeEncoding(separator, root_enc)
    elif encoding_type == D_POS_ENCODING:
            encoder = D_PosBasedEncoding(separator)
    elif encoding_type == D_BRACKET_ENCODING:
            encoder = D_BrkBasedEncoding(separator, displacement)
    elif encoding_type == D_BRACKET_ENCODING_2P:
            encoder = D_Brk2PBasedEncoding(separator, displacement, planar_alg)
    else:
        raise Exception("Unknown encoding type")
    
    f_idx_dict = {}
    if features:
        if features == ["ALL"]:
            features = extract_features_deps(in_path)
        i=0
        for f in features:
            f_idx_dict[f]=i
            i+=1

    file_out = open(out_path,"w+")
    label_set = set()
    tree_counter = 0
    label_counter = 0
    trees = D_Tree.read_conllu_file(in_path, filter_projective=False)
    
    for t in trees:
        # encode labels
        linearized_tree = encoder.encode(t)        
        file_out.write(linearized_tree.to_string(f_idx_dict))
        file_out.write("\n")
        
        tree_counter+=1
        label_counter+=len(linearized_tree)
        
        for lbl in linearized_tree.get_labels():
            label_set.add(str(lbl))      
    
    return tree_counter, label_counter, len(label_set)

# Decoding

def decode_dependencies(in_path, out_path, encoding_type, separator, displacement, multiroot, root_search, root_enc, postags, lang):
    '''
    Decodes the selected file according to the specified parameters:
    :param in_path: Path of the file to be encoded
    :param out_path: Path where to write the encoded labels
    :param encoding_type: Encoding used
    :param separator: string used to separate label fields
    :param displacement: boolean to indicate if use displacement in bracket based encodings
    :param multiroot: boolean to indicate if multiroot conll trees are allowed
    :param root_search: strategy to select how to search the root if no root found in decoded tree
    '''

    if encoding_type == D_ABSOLUTE_ENCODING:
        decoder = D_NaiveAbsoluteEncoding(separator)
    elif encoding_type == D_RELATIVE_ENCODING:
        decoder = D_NaiveRelativeEncoding(separator, root_enc)
    elif encoding_type == D_POS_ENCODING:
        decoder = D_PosBasedEncoding(separator)
    elif encoding_type == D_BRACKET_ENCODING:
        decoder = D_BrkBasedEncoding(separator, displacement)
    elif encoding_type == D_BRACKET_ENCODING_2P:
        decoder = D_Brk2PBasedEncoding(separator, displacement, None)
    else:
        raise Exception("Unknown encoding type")

    f_in=open(in_path)
    f_out=open(out_path,"w+")

    tree_counter=0
    labels_counter=0

    tree_string = ""

    if postags:
        stanza.download(lang=lang)
        nlp = stanza.Pipeline(lang=lang, processors='tokenize,pos')

    for line in f_in:
        if line == "\n":
            tree_string = tree_string.rstrip()
            current_tree = LinearizedTree.from_string(tree_string, mode="DEPS", separator=separator)
            
            if postags:
                c_tags = nlp(current_tree.get_sentence())
                current_tree.set_postags([word.pos for word in c_tags])

            decoded_tree = decoder.decode(current_tree)
            decoded_tree.postprocess_tree(root_search, multiroot)
            f_out.write("# text = "+decoded_tree.get_sentence()+"\n")
            f_out.write(str(decoded_tree))
            
            tree_string = ""
            tree_counter+=1
        
        tree_string += line
        labels_counter += 1


    return tree_counter, labels_counter
