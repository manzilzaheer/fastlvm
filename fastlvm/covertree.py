import covertreec

class CoverTree(object):
    """CoverTree Class"""
    def __init__(self, this):
        self.this = this

    def __del__(self):
        #print("Destructor called!")
        covertreec.delete(self.this)
        
    def __reduce__(self):
        buff = self.serialize()
        return (CoverTree.from_string, (buff,))

    @classmethod
    def from_matrix(cls, points, trunc=-1, use_multi_core=True):
        ptr = covertreec.new(points, trunc, use_multi_core)
        return cls(ptr)
        
    @classmethod
    def from_string(cls, buff):
        ptr = covertreec.deserialize(buff)
        return cls(ptr)

    def insert(self, point):
        return covertreec.insert(self.this, point)

    def remove(self, point):
        return covertreec.remove(self.this, point)

    def NearestNeighbour(self, points, use_multi_core=True, return_points=False):
        return covertreec.NearestNeighbour(self.this, points, use_multi_core, return_points)

    def kNearestNeighbours(self, points, k=10, use_multi_core=True, return_points=False):
        return covertreec.kNearestNeighbours(self.this, points, k, use_multi_core, return_points)
        
    def RangeSearch(self, points, r=1.0, use_multi_core=True, return_points=False):
        return covertreec.RangeSearch(self.this, points, r, use_multi_core, return_points)
        
    def serialize(self):
        return covertreec.serialize(self.this)

    def display(self):
        return covertreec.display(self.this)

    def test_covering(self):
        return covertreec.test_covering(self.this)
