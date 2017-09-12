import kmeansc
import numpy as np
import pdb
    
def init_covertree(k, points):
    import covertreec
    trunc = 3
    ptr = covertreec.new(points, trunc)
    #covertreec.display(ptr)
    seeds = covertreec.spreadout(ptr, k)
    covertreec.delete(ptr)
    return seeds
    
def init_kmeanspp(k, points):
    import utilsc
    seed_idx = utilsc.kmeanspp(k, points)
    seeds = points[seed_idx]
    return seeds
    

class KMeans(object):
    """KMeans Class"""
    def __init__(self, this):
        self.this = this

    def __del__(self):
        print("Destructor called!")
        kmeansc.delete(self.this)
        
    def __reduce__(self):
        buff = self.serialize()
        return (KMeans.from_string, (buff,))

    @classmethod
    def init(cls, k, iters, initial_centres, data=None):
        if initial_centres == 'random':
            if data is None:
                raise ValueError('Must provide data when using random')
            idx = np.random.choice(data.shape[0], k, replace=False)
            initial_centres = data[idx]
        elif initial_centres == 'firstk':
            if data is None:
                raise ValueError('Must provide data when using firstk')
            initial_centres = data[:k]
        elif initial_centres == 'kmeanspp':
            if data is None:
                raise ValueError('Must provide data when using kmeanspp')
            initial_centres = init_kmeanspp(k, data)
        elif initial_centres == 'covertree':
            if data is None:
                raise ValueError('Must provide data when using covertree')
            initial_centres = init_covertree(k, data)
        elif isinstance(initial_centres, np.ndarray):
            if initial_centres.shape[0] != k:
                raise ValueError('Must provide ', k, ' initial centres when providing numpy arrays!')
        else:
            raise NotImplementedError('This type of initial centres is not implemented')
        
        ptr = kmeansc.new(k, iters, initial_centres)
        return cls(ptr)
        
    @classmethod
    def from_string(cls, buff):
        ptr = kmeansc.deserialize(buff)
        return cls(ptr)
        
    def fit(self, trngpoints, testpoints):
        return kmeansc.fit(self.this, trngpoints, testpoints)
        
    def evaluate(self, points):
        return kmeansc.evaluate(self.this, points)

    def predict(self, points):
        return kmeansc.predict(self.this, points)

    def get_centers(self):
        return kmeansc.centers(self.this)
        
    def serialize(self):
        return kmeansc.serialize(self.this)
