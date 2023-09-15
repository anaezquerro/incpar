from supar.codelin.utils.constants import D_ROOT_HEAD, D_NULLHEAD, D_ROOT_REL, D_POSROOT, D_EMPTYREL

class D_Node:
    def __init__(self, wid, form, lemma=None, upos=None, xpos=None, feats=None, head=None, deprel=None, deps=None, misc=None):
        self.id = int(wid)                      # word id
        
        self.form = form if form else "_"       # word 
        self.lemma = lemma if lemma else "_"    # word lemma/stem
        self.upos = upos if upos else "_"       # universal postag
        self.xpos = xpos if xpos else "_"       # language_specific postag
        self.feats = self.parse_feats(feats) if feats else "_"    # morphological features
        
        self.head = int(head)                   # id of the word that depends on
        self.relation = deprel                  # type of relation with head

        self.deps = deps if deps else "_"       # enhanced dependency graph
        self.misc = misc if misc else "_"       # miscelaneous data
    
    def is_left_arc(self):
        return self.head > self.id

    def delta_head(self):
        return self.head - self.id
    
    def parse_feats(self, feats):
        if feats == '_':
            return [None]
        else:
            return [x for x in feats.split('|')]

    def check_cross(self, other):
        if ((self.head == other.head) or (self.head==other.id)):
                return False

        r_id_inside = (other.head < self.id < other.id)
        l_id_inside = (other.id < self.id < other.head)

        id_inside = r_id_inside or l_id_inside

        r_head_inside = (other.head < self.head < other.id)
        l_head_inside = (other.id < self.head < other.head)

        head_inside = r_head_inside or l_head_inside

        return head_inside^id_inside
    
    def __repr__(self):
        return '\t'.join(str(e) for e in list(self.__dict__.values()))+'\n'

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    

    @staticmethod
    def from_string(conll_str):
        wid,form,lemma,upos,xpos,feats,head,deprel,deps,misc = conll_str.split('\t')
        return D_Node(int(wid), form, lemma, upos, xpos, feats, int(head), deprel, deps, misc)

    @staticmethod
    def dummy_root():
        return D_Node(0, D_POSROOT, None, D_POSROOT, None, None, 0, D_EMPTYREL, None, None)
    
    @staticmethod
    def empty_node():
        return D_Node(0, None, None, None, None, None, 0, None, None, None)

class D_Tree:
    def __init__(self, nodes):
        self.nodes = nodes

# getters    
    def get_node(self, id):
        return self.nodes[id-1]

    def get_edges(self):
        '''
        Return sentence dependency edges as a tuple 
        shaped as ((d,h),r) where d is the dependant of the relation,
        h the head of the relation and r the relationship type
        '''
        return list(map((lambda x :((x.id, x.head), x.relation)), self.nodes))
    
    def get_arcs(self):
        '''
        Return sentence dependency edges as a tuple 
        shaped as (d,h) where d is the dependant of the relation,
        and h the head of the relation.
        '''
        return list(map((lambda x :(x.id, x.head)), self.nodes))

    def get_relations(self):
        '''
        Return a list of relationships betwee nodes
        '''
        return list(map((lambda x :x.relation), self.nodes))
    
    def get_sentence(self):
        '''
        Return the sentence as a string
        '''
        return " ".join(list(map((lambda x :x.form), self.nodes)))

    def get_words(self):
        '''
        Returns the words of the sentence as a list
        '''
        return list(map((lambda x :x.form), self.nodes))

    def get_indexes(self):
        '''
        Returns a list of integers representing the words of the 
        dependency tree
        '''
        return list(map((lambda x :x.id), self.nodes))

    def get_postags(self):
        '''
        Returns the part of speech tags of the tree
        '''
        return list(map((lambda x :x.upos), self.nodes))

    def get_lemmas(self):
        '''
        Returns the lemmas of the tree
        '''
        return list(map((lambda x :x.lemma), self.nodes))

    def get_heads(self):
        '''
        Returns the heads of the tree
        '''
        return list(map((lambda x :x.head), self.nodes))
    
    def get_feats(self):
        '''
        Returns the morphological features of the tree
        '''
        return list(map((lambda x :x.feats), self.nodes))

# update functions
    def append_node(self, node):
        '''
        Append a node to the tree and sorts the nodes by id
        '''
        self.nodes.append(node)
        self.nodes.sort(key=lambda x: x.id)

    def update_head(self, node_id, head_value):
        '''
        Update the head of a node indicated by its id
        '''
        for node in self.nodes:
            if node.id == node_id:
                node.head = head_value
                break
    
    def update_relation(self, node_id, relation_value):
        '''
        Update the relation of a node indicated by its id
        '''
        for node in self.nodes:
            if node.id == node_id:
                node.relation = relation_value
                break
    
    def update_word(self, node_id, word):
        '''
        Update the word of a node indicated by its id
        '''
        for node in self.nodes:
            if node.id == node_id:
                node.form = word
                break

    def update_upos(self, node_id, postag):
        '''
        Update the upos field of a node indicated by its id
        '''
        for node in self.nodes:
            if node.id == node_id:
                node.upos = postag
                break

# properties functions
    def is_projective(self):
        '''
        Returns a boolean indicating if the dependency tree
        is projective (i.e. no edges are crossing)
        '''
        arcs = self.get_arcs()
        for (i,j) in arcs:
            for (k,l) in arcs:
                if (i,j) != (k,l) and min(i,j) < min(k,l) < max(i,j) < max(k,l):
                    return False
        return True
    
# postprocessing
    def remove_dummy(self):
        self.nodes = self.nodes[1:]

    def postprocess_tree(self, search_root_strat, allow_multi_roots=False):
        '''
        Postprocess the tree by finding the root according to the selected 
        strategy and fixing cycles and out of bounds heads
        '''
        # 1) Find the root
        root = self.root_search(search_root_strat)
        
        # 2) Fix oob heads
        self.fix_oob_heads()
        
        # 3) Fix cycles
        self.fix_cycles(root)
        
        # 4) Set all null heads to root and remove other root candidates
        for node in self.nodes:
            if node.id == root:
                node.head = 0
                continue
            if node.head == D_NULLHEAD:
                node.head = root
            if not allow_multi_roots and node.head == 0:
                node.head = root

    def root_search(self, search_root_strat):
        '''
        Search for the root of the tree using the method indicated
        '''
        root = 1 # Default root
        for node in self.nodes:    
            if search_root_strat == D_ROOT_HEAD:
                if node.head == 0:
                    root = node.id
                    break
            
            elif search_root_strat == D_ROOT_REL:
                if node.rel == 'root' or node.rel == 'ROOT':
                    root = node.id
                    break
        return root

    def fix_oob_heads(self):
        '''
        Fixes heads of the tree (if they dont exist, if they are out of bounds, etc)
        If a head is out of bounds set it to nullhead
        '''
        for node in self.nodes:
            if node.head==D_NULLHEAD:
                continue
            if int(node.head) < 0:
                node.head = D_NULLHEAD
            elif int(node.head) > len(self.nodes):
                node.head = D_NULLHEAD
    
    def fix_cycles(self, root):
        '''
        Breaks cycles in the tree by setting the head of the node to root_id
        '''
        for node in self.nodes:
            visited = []
            
            while (node.id != root) and (node.head !=D_NULLHEAD):
                if node in visited:
                    node.head = D_NULLHEAD
                else:
                    visited.append(node)
                    next_node = min(max(node.head-1, 0), len(self.nodes)-1)
                    node = self.nodes[next_node]
        
# python related functions
    def __repr__(self):
        return "".join(str(e) for e in self.nodes)+"\n"
    
    def __iter__(self):
        for n in self.nodes:
            yield n

    def __getitem__(self, key):
        return self.nodes[key]  
    
    def __len__(self):
        return self.nodes.__len__()

# base tree
    @staticmethod
    def empty_tree(l=1):
        ''' 
        Creates an empty dependency tree with l nodes
        '''
        t = D_Tree([])
        for i in range(l):
            n = D_Node.empty_node()
            n.id = i
            t.append_node(n)
        return t

# reader and writter
    @staticmethod
    def from_string(conll_str, dummy_root=True, clean_contractions=True, clean_omisions=True):
        '''
        Create a ConllTree from a dependency tree conll-u string.
        '''
        data = conll_str.split('\n')
        dependency_tree_start_index = 0
        for line in data:
            if len(line)>0 and line[0]!="#":
                break
            dependency_tree_start_index+=1
        data = data[dependency_tree_start_index:]
        nodes = []
        if dummy_root:
            nodes.append(D_Node.dummy_root())
        
        for line in data:
            # check if not valid line (empty or not enough fields)
            if (len(line)<=1) or len(line.split('\t'))<10:
                continue 
            
            wid = line.split('\t')[0]

            # check if node is a comment (comments are marked with #)
            if "#" in wid:
                continue
            
            # check if node is a contraction (multiexp lines are marked with .)
            if clean_contractions and "-" in wid:    
                continue
            
            # check if node is an omited word (empty nodes are marked with .)
            if clean_omisions and "." in wid:
                continue

            conll_node = D_Node.from_string(line)
            nodes.append(conll_node)
        
        return D_Tree(nodes)
    
    @staticmethod
    def read_conllu_file(file_path, filter_projective = True):
        '''
        Read a conllu file and return a list of ConllTree objects.
        '''
        with open(file_path, 'r') as f:
            data = f.read()
        data = data.split('\n\n')
        # remove last empty line
        data = data[:-1]
        
        trees = []
        for x in data:
            t = D_Tree.from_string(x)
            if not filter_projective or t.is_projective():
                trees.append(t)
        return trees    

    @staticmethod
    def write_conllu_file(file_path, trees):
        '''
        Write a list of ConllTree objects to a conllu file.
        '''
        with open(file_path, 'w') as f:
            f.write("".join(str(e) for e in trees))

    @staticmethod
    def write_conllu(file_io, tree):
        '''
        Write a single ConllTree to a already open file.
        Includes the # text = ... line
        '''
        file_io.write("# text = "+tree.get_sentence()+"\n")
        file_io.write("".join(str(e) for e in tree)+"\n")