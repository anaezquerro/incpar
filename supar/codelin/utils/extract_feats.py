import argparse
from supar.codelin.models.const_tree import C_Tree
from supar.codelin.models.deps_tree import D_Tree

def extract_features_const(in_path):
    file_in = open(in_path, "r")
    feats_set = set()
    for line in file_in:
        line = line.rstrip()
        tree = C_Tree.from_string(line)
        tree.extract_features()
        feats = tree.get_feature_names()
        
        feats_set = feats_set.union(feats)

    return sorted(feats_set)

def extract_features_deps(in_path):
    feats_list=set()
    trees = D_Tree.read_conllu_file(in_path, filter_projective=False)
    for t in trees:
        for node in t:
            if node.feats != "_":
                feats_list = feats_list.union(a for a in (node.feats.keys()))

    return sorted(feats_list)
    

'''
Python script that returns a ordered list of the features
included in a conll tree or a constituent tree
'''

# parser = argparse.ArgumentParser(description='Prints all features in a constituent treebank')
# parser.add_argument('form', metavar='formalism', type=str, choices=['CONST','DEPS'], help='Grammar encoding the file to extract features')
# parser.add_argument('input', metavar='in file', type=str, help='Path of the file to clean (.trees file).')
# args = parser.parse_args()
# if args.form=='CONST':
#     feats = extract_features_const(args.input)
# elif args.form=='DEPS':
#     feats = extract_features_conll(args.input)
# print(" ".join(feats))
