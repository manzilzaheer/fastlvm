import ldac
import numpy as np
import pdb
    
class LDA(object):
    """LDA Class"""
    def __init__(self, this):
        self.this = this
        self.ext = None

    def __del__(self):
        print("Destructor called!")
        ldac.delete(self.this, self.ext)
        
    def __reduce__(self):
        buff = self.serialize()
        return (LDA.from_string, (buff,))

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
        
        ptr = ldac.new(k, iters, vocab)
        return cls(ptr)
        
    @classmethod
    def from_string(cls, buff):
        ptr = ldac.deserialize(buff)
        return cls(ptr)
        
    def fit(self, trngdata, testdata):
        return ldac.fit(self.this, trngdata, testdata)
        
    def evaluate(self, data):
        return ldac.evaluate(self.this, data)

    def predict(self, data):
        pass

    def get_topic_matrix(self):
        if self.ext is None:
            self.ext = ldac.topic_matrix(self.this)
        return self.ext
        
    def get_top_words(self, number=15):
        return ldac.top_words(self.this, number)
        
    def serialize(self):
        return ldac.serialize(self.this)