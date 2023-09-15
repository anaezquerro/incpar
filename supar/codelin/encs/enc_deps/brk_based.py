from supar.codelin.encs.abstract_encoding import ADEncoding
from supar.codelin.models.deps_label import D_Label
from supar.codelin.models.deps_tree import D_Tree
from supar.codelin.utils.constants import D_NONE_LABEL
from supar.codelin.models.linearized_tree import LinearizedTree

class D_BrkBasedEncoding(ADEncoding):
    
    def __init__(self, separator, displacement):
        super().__init__(separator)
        self.displacement = displacement

    def __str__(self):
        return "Dependency Bracketing Based Encoding"


    def encode(self, dep_tree):
        n_nodes = len(dep_tree)
        labels_brk     = [""] * (n_nodes + 1)
        encoded_labels = []
        
        # compute brackets array
        # brackets array should be sorted ?
        dep_tree.remove_dummy()
        for node in dep_tree:
            # skip root relations (optional?)
            if node.head == 0:
                continue
            
            if node.is_left_arc():
                labels_brk[node.id + (1 if self.displacement else 0)]+='<'
                labels_brk[node.head]+='\\'
            
            else:
                labels_brk[node.head + (1 if self.displacement else 0)]+='/'
                labels_brk[node.id]+='>'
        
        # encode labels
        for node in dep_tree:
            li = node.relation
            xi = labels_brk[node.id]

            current = D_Label(xi, li, self.separator)
            encoded_labels.append(current)

        return LinearizedTree(dep_tree.get_words(), dep_tree.get_postags(), dep_tree.get_feats(), encoded_labels, len(encoded_labels))

    def decode(self, lin_tree):
        # Create an empty tree with n labels
        decoded_tree = D_Tree.empty_tree(len(lin_tree)+1)
        
        l_stack = []
        r_stack = []
        
        current_node = 1
        for word, postag, features, label in lin_tree.iterrows():
            
            # get the brackets
            brks = list(label.xi) if label.xi != D_NONE_LABEL else []
                       
            # set parameters to the node
            decoded_tree.update_word(current_node, word)
            decoded_tree.update_upos(current_node, postag)
            decoded_tree.update_relation(current_node, label.li)

            # fill the relation using brks
            for char in brks:
                if char == "<":
                    node_id = current_node + (-1 if self.displacement else 0)
                    r_stack.append(node_id)

                if char == "\\":
                    head_id = r_stack.pop() if len(r_stack) > 0 else 0
                    decoded_tree.update_head(head_id, current_node)
                
                if char =="/":
                    node_id = current_node + (-1 if self.displacement else 0)
                    l_stack.append(node_id)

                if char == ">":
                    head_id = l_stack.pop() if len(l_stack) > 0 else 0
                    decoded_tree.update_head(current_node, head_id)
            
            current_node+=1

        decoded_tree.remove_dummy()
        return decoded_tree