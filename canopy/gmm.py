import gmmc
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
    

class GMM(object):
    """CoverTree Class"""
    def __init__(self, this):
        self.this = this

    def __del__(self):
        print("Destructor called!")
        gmmc.delete(self.this)
        
    def __reduce__(self):
        buff = self.serialize()
        return (GMM.from_string, (buff,))

    @classmethod
    def init(cls, k, iters, initial_centres, data=None, initial_vars=None):
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
                raise ValueError('Must provide ', k, ' initial means when providing numpy arrays!')
        else:
            raise NotImplementedError('This type of initial means is not implemented')

        if initial_vars is None:
            if data is None:
                raise ValueError('Must provide data when not providing init var')
            initial_vars = 0.5/np.var(data, axis=0)
            #initial_vars = np.tile(initial_vars.ravel(), (k,1))
        elif isinstance(initial_vars, np.ndarray):
            if sum(initial_vars.shape) == initial_centres.shape[1]:
                initial_vars = np.tile(initial_vars.ravel(), (k,1))
            elif initial_vars.shape[0] != k:
                raise ValueError('Must provide ', k, ' initial vars when providing numpy arrays!')
        else:
            raise NotImplementedError('This type of initial vars is not implemented')

        ptr = gmmc.new(k, iters, initial_centres, initial_vars)
        return cls(ptr)
        
    @classmethod
    def from_string(cls, buff):
        ptr = gmmc.deserialize(buff)
        return cls(ptr)
        
    def fit(self, trngpoints, testpoints):
        return gmmc.fit(self.this, trngpoints, testpoints)
        
    def evaluate(self, points):
        return gmmc.evaluate(self.this, points)

    def predict(self, points):
        return gmmc.predict(self.this, points)

    def get_centers(self):
        return gmmc.centers(self.this)
        
    def serialize(self):
        return gmmc.serialize(self.this)