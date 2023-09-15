from abc import ABC, abstractmethod

class ADEncoding(ABC):
    '''
    Abstract class for Dependency Encodings
    Sets the main constructor method and defines the methods
        - Encode
        - Decode
    When adding a new Dependency Encoding it must extend this class
    and implement those methods
    '''
    def __init__(self, separator):
        self.separator = separator
    
    def encode(self, nodes):
        pass
    def decode(self, labels, postags, words):
        pass

class ACEncoding(ABC):
    '''
    Abstract class for Constituent Encodings
    Sets the main constructor method and defines the abstract methods
        - Encode
        - Decode
    When adding a new Constituent Encoding it must extend this class
    and implement those methods
    '''
    def __init__(self, separator, ujoiner):
        self.separator = separator
        self.unary_joiner = ujoiner
    
    def encode(self, constituent_tree):
        pass
    def decode(self, linearized_tree):
        pass

