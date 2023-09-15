from supar.codelin.encs.abstract_encoding import ADEncoding
from supar.codelin.models.deps_label import D_Label
from supar.codelin.models.linearized_tree import LinearizedTree
from supar.codelin.models.deps_tree import D_Tree
from supar.codelin.utils.constants import D_NONE_LABEL

class D_NaiveRelativeEncoding(ADEncoding):
    def __init__(self, separator, hang_from_root):
        super().__init__(separator)
        self.hfr = hang_from_root

    def __str__(self):
        return "Dependency Naive Relative Encoding"

    def encode(self, dep_tree):
        encoded_labels = []
        dep_tree.remove_dummy()
        for node in dep_tree:
            li = node.relation 
            xi = node.delta_head()

            if node.relation == 'root' and self.hfr:
                xi = D_NONE_LABEL
            
            current = D_Label(xi, li, self.separator)
            encoded_labels.append(current)

        return LinearizedTree(dep_tree.get_words(), dep_tree.get_postags(), dep_tree.get_feats(), encoded_labels, len(encoded_labels))

    def decode(self, lin_tree):
        dep_tree = D_Tree.empty_tree(len(lin_tree)+1)

        i = 1
        for word, postag, features, label in lin_tree.iterrows():
            if label.xi == D_NONE_LABEL:
                # set as root
                dep_tree.update_head(i, 0)
            else:
                dep_tree.update_head(i, int(label.xi)+(i))
                
            
            dep_tree.update_word(i, word)
            dep_tree.update_upos(i, postag)
            dep_tree.update_relation(i, label.li)
            i+=1

        dep_tree.remove_dummy()
        return dep_tree