from supar.codelin.encs.abstract_encoding import ADEncoding
from supar.codelin.models.deps_label import D_Label
from supar.codelin.models.deps_tree import D_Tree
from supar.codelin.models.linearized_tree import LinearizedTree
from supar.codelin.utils.constants import D_POSROOT, D_NONE_LABEL

POS_ROOT_LABEL = "0--ROOT"

class D_PosBasedEncoding(ADEncoding):
    def __init__(self, separator):
        super().__init__(separator)

    def __str__(self) -> str:
        return "Dependency Part-of-Speech Based Encoding"
        
    def encode(self, dep_tree):
        encoded_labels = []
        
        for node in dep_tree:
            if node.id == 0:
                # skip dummy root
                continue

            li = node.relation     
            pi = dep_tree[node.head].upos           
            oi = 0
            
            # move left or right depending if the node 
            # dependency edge is to the left or to the right

            step = 1 if node.id < node.head else -1
            for i in range(node.id + step, node.head + step, step):
                if pi == dep_tree[i].upos:
                    oi += step

            xi = str(oi)+"--"+pi

            current = D_Label(xi, li, self.separator)
            encoded_labels.append(current)

        dep_tree.remove_dummy()
        return LinearizedTree(dep_tree.get_words(), dep_tree.get_postags(), dep_tree.get_feats(), encoded_labels, len(encoded_labels))

    def decode(self, lin_tree):
        dep_tree = D_Tree.empty_tree(len(lin_tree)+1)

        i = 1
        postags = lin_tree.postags
        for word, postag, features, label in lin_tree.iterrows():
            node_id = i
            if label.xi == D_NONE_LABEL:
                label.xi = POS_ROOT_LABEL
            
            dep_tree.update_word(node_id, word)
            dep_tree.update_upos(node_id, postag)
            dep_tree.update_relation(node_id, label.li)
            
            oi, pi = label.xi.split('--')
            oi = int(oi)

            # Set head for root
            if (pi==D_POSROOT or oi==0):
                dep_tree.update_head(node_id, 0)
                i+=1
                continue

            # Compute head position
            target_oi = oi

            step = 1 if oi > 0 else -1
            stop_point = (len(postags)+1) if oi > 0 else 0

            for j in range(node_id+step, stop_point, step):
                if (pi == postags[j-1]):
                    target_oi -= step
                
                if (target_oi==0):
                    break
            
            head_id = j
            dep_tree.update_head(node_id, head_id)
            
            i+=1

        
        dep_tree.remove_dummy()
        return dep_tree