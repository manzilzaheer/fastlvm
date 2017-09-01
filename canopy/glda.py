import gldac
import numpy as np
import pdb
    
class GLDA(object):
    """GLDA Class"""
    def __init__(self, this):
        self.this = this
        self.ext = None
        self.word_vec = None

    def __del__(self):
        print("Destructor called!")
        gldac.delete(self.this, self.ext)
        
    def __reduce__(self):
        buff = self.serialize()
        return (GLDA.from_string, (buff,))

    @classmethod
    def init(cls, k, iters, vocab, vectors, data=None):
        print(id(vectors))
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
        if isinstance(vectors, np.ndarray):
            pass
            #vectors = np.require(vectors, requirements=['A', 'C', 'O', 'W'])
            #print(id(vectors))
        elif isinstance(vectors, list):
            vectors = np.array(vectors)
            vectors = np.require(vectors, requirements=['A', 'C', 'O', 'W'])
        else:
            raise NotImplementedError('This type of vectors is not implemented')
        
        ptr = gldac.new(k, iters, vocab, vectors)
        obj = cls(ptr)
        obj.word_vec = vectors
        return obj
        
    @classmethod
    def from_string(cls, buff):
        ptr = gldac.deserialize(buff)
        obj = cls(ptr)
        obj.word_vec = gldac.word_vec(ptr)
        return obj
        
    def fit(self, trngdata, testdata):
        return gldac.fit(self.this, trngdata, testdata)
        
    def evaluate(self, data):
        return gldac.evaluate(self.this, data)

    def predict(self, data):
        pass

    def get_topic_matrix(self):
        if self.ext is None:
            self.ext = gldac.topic_matrix(self.this)
        return self.ext
        
    def get_top_words(self, number=15):
        return gldac.top_words(self.this, number)
        
    def serialize(self):
        return gldac.serialize(self.this)