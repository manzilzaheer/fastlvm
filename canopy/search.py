import covertreec

class Search(object):
    """CoverTree Class"""
    def __init__(self, this):
        self.this = this

    def __del__(self):
        #print("Destructor called!")
        covertreec.delete(self.this)
        
    def __reduce__(self):
        buff = self.serialize()
        return (Search.from_string, (buff,))

    @classmethod
    def from_matrix(cls, points, trunc=-1):
        ptr = covertreec.new(points, trunc)
        return cls(ptr)
        
    @classmethod
    def from_string(cls, buff):
        ptr = covertreec.deserialize(buff)
        return cls(ptr)

    def insert(self, point):
        return covertreec.insert(self.this, point)

    def remove(self, point):
        return covertreec.remove(self.this, point)

    def NearestNeighbour(self, points):
        return covertreec.NearestNeighbour(self.this, points)

    def kNearestNeighbours(self, points, k=10):
        return covertreec.kNearestNeighbours(self.this, points, k)
        
    def range(self, points, r=1.0):
        print("Sorry not implemented yet!")
        return None
        
    def serialize(self):
        return covertreec.serialize(self.this)

    def display(self):
        return covertreec.display(self.this)

    def test_covering(self):
        return covertreec.test_covering(self.this)
