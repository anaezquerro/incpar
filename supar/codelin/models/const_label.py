from supar.codelin.utils.constants import C_ABSOLUTE_ENCODING, C_RELATIVE_ENCODING, C_NONE_LABEL

class C_Label:
    def __init__(self, nc, lc, uc, et, sp, uj):
        self.encoding_type = et

        self.n_commons = int(nc)
        self.last_common = lc
        self.unary_chain = uc if uc != C_NONE_LABEL else None
        self.separator = sp
        self.unary_joiner = uj
    
    def __repr__(self):
        unary_str = self.unary_joiner.join([self.unary_chain]) if self.unary_chain else ""
        return (str(self.n_commons) + ("*" if self.encoding_type==C_RELATIVE_ENCODING else "")
        + self.separator + self.last_common + (self.separator + unary_str if self.unary_chain else ""))
    
    def to_absolute(self, last_label):
        self.n_commons+=last_label.n_commons
        if self.n_commons<=0:
            self.n_commons = 1
        
        self.encoding_type=C_ABSOLUTE_ENCODING
    
    @staticmethod
    def from_string(l, sep, uj):
        label_components = l.split(sep)
        
        if len(label_components)== 2:
            nc, lc = label_components
            uc = None
        else:
            nc, lc, uc = label_components
        
        et = C_RELATIVE_ENCODING if '*' in nc else C_ABSOLUTE_ENCODING
        nc = nc.replace("*","")
        return C_Label(nc, lc, uc, et, sep, uj)
    