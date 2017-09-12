import dldac
import numpy as np
import pdb
    
class DLDA(object):
    """LDA Class"""
    def __init__(self, this):
        self.this = this
        self.ext = None

    def __del__(self):
        print("Destructor called!")
        dldac.delete(self.this, self.ext)
        
    def __reduce__(self):
        buff = self.serialize()
        return (DLDA.from_string, (buff,))

    @classmethod
    def init(cls, k, iters, vocab, data=None):
        if isinstance(vocab, int):
            if vocab < 0:
                raise ValueError('Vocab size must be non-negative!')
            vocab = [''.join(['w',i]) for i in range(vocab)]
        elif isinstance(vocab, list):
            if len(vocab) > 0:
                if not isinstance(vocab[0], str):
                    raise ValueError('Vocab must be list of stringss!')
        else:
            raise NotImplementedError('This type of vocab is not implemented')
        
        ptr = dldac.new(k, iters, vocab)
        return cls(ptr)
        
    @classmethod
    def from_string(cls, buff):
        ptr = dldac.deserialize(buff)
        return cls(ptr)
        
    def fit(self, trngdata, testdata):
        return dldac.fit(self.this, trngdata, testdata)
        
    def evaluate(self, data):
        return dldac.evaluate(self.this, data)

    def predict(self, data):
        pass

    def get_topic_matrix(self):
        if self.ext is None:
            self.ext = dldac.topic_matrix(self.this)
        return self.ext
        
    def get_top_words(self, number=15):
        return dldac.top_words(self.this, number)
        
    def serialize(self):
        return dldac.serialize(self.this)