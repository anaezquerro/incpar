from supar.codelin.encs.abstract_encoding import ADEncoding
from supar.codelin.utils.constants import D_2P_GREED, D_2P_PROP, D_NONE_LABEL
from supar.codelin.models.deps_label import D_Label
from supar.codelin.models.deps_tree import D_Tree
from supar.codelin.models.linearized_tree import LinearizedTree


class D_Brk2PBasedEncoding(ADEncoding):
    def __init__(self, separator, displacement, planar_alg):
        if planar_alg and planar_alg not in [D_2P_GREED, D_2P_PROP]:
            print("[*] Error: Unknown planar separation algorithm")
            exit(1)
        super().__init__(separator)
        self.displacement = displacement
        self.planar_alg = planar_alg

    def __str__(self):
        return "Dependency 2-Planar Bracketing Based Encoding"

    def get_next_edge(self, dep_tree, idx_l, idx_r):
        next_arc=None

        if dep_tree[idx_l].head==idx_r:
            next_arc = dep_tree[idx_l]
        
        elif dep_tree[idx_r].head==idx_l:
            next_arc = dep_tree[idx_r]
        
        return next_arc

    def two_planar_propagate(self, nodes):
        p1=[]
        p2=[]
        fp1=[]
        fp2=[]

        for i in range(0, (len(nodes))):
            for j in range(i, -1, -1):
                # if the node in position 'i' has an arc to 'j' 
                # or node in position 'j' has an arc to 'i'
                next_arc=self.get_next_edge(nodes, i, j)
                if next_arc is None:
                    continue
                else:
                    # check restrictions
                    if next_arc not in fp1:
                        p1.append(next_arc)
                        fp1, fp2 = self.propagate(nodes, fp1, fp2, next_arc, 2)
                    
                    elif next_arc not in fp2:
                        p2.append(next_arc)
                        fp1, fp2 = self.propagate(nodes, fp1, fp2, next_arc, 1)
        return p1, p2
    def propagate(self, nodes, fp1, fp2, current_edge, i):
        # add the current edge to the forbidden plane opposite to the plane
        # where the node has already been added
        fpi  = None
        fp3mi= None
        if i==1:
            fpi  = fp1
            fp3mi= fp2
        if i==2:
            fpi  = fp2
            fp3mi= fp1

        fpi.append(current_edge)
        
        # add all nodes from the dependency graph that crosses the current edge
        # to the corresponding forbidden plane
        for node in nodes:
            if current_edge.check_cross(node):
                if node not in fp3mi:
                    (fp1, fp2)=self.propagate(nodes, fp1, fp2, node, 3-i)
        
        return fp1, fp2

    def two_planar_greedy(self, dep_tree):    
        plane_1 = []
        plane_2 = []

        for i in range(len(dep_tree)):
            for j in range(i, -1, -1):
                # if the node in position 'i' has an arc to 'j' 
                # or node in position 'j' has an arc to 'i'
                next_arc = self.get_next_edge(dep_tree, i, j)
                if next_arc is None:
                    continue

                else:
                    cross_plane_1 = False
                    cross_plane_2 = False
                    for node in plane_1:                
                        cross_plane_1 = cross_plane_1 or next_arc.check_cross(node)
                    for node in plane_2:        
                        cross_plane_2 = cross_plane_2 or next_arc.check_cross(node)
                    
                    if not cross_plane_1:
                        plane_1.append(next_arc)
                    elif not cross_plane_2:
                        plane_2.append(next_arc)

        # processs them separately
        return plane_1,plane_2


    def encode(self, dep_tree):
        # create brackets array
        n_nodes = len(dep_tree)
        labels_brk     = [""] * (n_nodes + 1)

        # separate the planes
        if self.planar_alg==D_2P_GREED:
            p1_nodes, p2_nodes = self.two_planar_greedy(dep_tree)
        elif self.planar_alg==D_2P_PROP:
            p1_nodes, p2_nodes = self.two_planar_propagate(dep_tree)
            
        # get brackets separatelly
        labels_brk = self.encode_step(p1_nodes, labels_brk, ['>','/','\\','<'])
        labels_brk = self.encode_step(p2_nodes, labels_brk, ['>*','/*','\\*','<*'])
        
        # merge and obtain labels
        lbls=[]
        dep_tree.remove_dummy()
        for node in dep_tree:
            current = D_Label(labels_brk[node.id], node.relation, self.separator)
            lbls.append(current)
        return LinearizedTree(dep_tree.get_words(), dep_tree.get_postags(), dep_tree.get_feats(), lbls, len(lbls))

    def encode_step(self, p, lbl_brk, brk_chars):
        for node in p:
            # skip root relations (optional?)
            if node.head==0:
                continue
            if node.id < node.head:
                if self.displacement:
                    lbl_brk[node.id+1]+=brk_chars[3]
                else:
                    lbl_brk[node.id]+=brk_chars[3]

                lbl_brk[node.head]+=brk_chars[2]
            else:
                if self.displacement:
                    lbl_brk[node.head+1]+=brk_chars[1]
                else:
                    lbl_brk[node.head]+=brk_chars[1]

                lbl_brk[node.id]+=brk_chars[0]
        return lbl_brk

    def decode(self, lin_tree):
        decoded_tree = D_Tree.empty_tree(len(lin_tree)+1)
        
        # create plane stacks
        l_stack_p1=[]
        l_stack_p2=[]
        r_stack_p1=[]
        r_stack_p2=[]
        
        current_node=1

        for word, postag, features, label in lin_tree.iterrows():
            brks = list(label.xi) if label.xi != D_NONE_LABEL else []
            temp_brks=[]
            
            for i in range(0, len(brks)):
                current_char=brks[i]
                if brks[i]=="*":
                    current_char=temp_brks.pop()+brks[i]
                temp_brks.append(current_char)
                    
            brks=temp_brks
            
            # set parameters to the node
            decoded_tree.update_word(current_node, word)
            decoded_tree.update_upos(current_node, postag)
            decoded_tree.update_relation(current_node, label.li)
            
            # fill the relation using brks
            for char in brks:
                if char == "<":
                    node_id=current_node + (-1 if self.displacement else 0)
                    r_stack_p1.append((node_id,char))
                
                if char == "\\":
                    head_id = r_stack_p1.pop()[0] if len(r_stack_p1)>0 else 0
                    decoded_tree.update_head(head_id, current_node)
                
                if char =="/":
                    node_id=current_node + (-1 if self.displacement else 0)
                    l_stack_p1.append((node_id,char))

                if char == ">":
                    head_id = l_stack_p1.pop()[0] if len(l_stack_p1)>0 else 0
                    decoded_tree.update_head(current_node, head_id)

                if char == "<*":
                    node_id=current_node + (-1 if self.displacement else 0)
                    r_stack_p2.append((node_id,char))
                
                if char == "\\*":
                    head_id = r_stack_p2.pop()[0] if len(r_stack_p2)>0 else 0
                    decoded_tree.update_head(head_id, current_node)
                
                if char =="/*":
                    node_id=current_node + (-1 if self.displacement else 0)
                    l_stack_p2.append((node_id,char))

                if char == ">*":
                    head_id = l_stack_p2.pop()[0] if len(l_stack_p2)>0 else 0
                    decoded_tree.update_head(current_node, head_id)
            
            current_node+=1

        decoded_tree.remove_dummy()
        return decoded_tree   
