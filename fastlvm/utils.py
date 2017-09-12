import utilsc

_stirling = utilsc.new_stirling()

def read_corpus(fname, vocab=[], stopwords=[]):
    return utilsc.read_corpus(fname, vocab, stopwords)
    
def get_ref_count(var):
    return utilsc.ref_count(var)

def kmeanspp(k, points):
    seed_idx = utilsc.kmeanspp(k, points)
    seeds = points[seed_idx]
    return seeds

def log_stirling_num(n, m):
    return utilsc.log_stirling_num(_stirling, n, m)

def uratio(n, m):
    return utilsc.uratio(_stirling, n, m)

def vratio(n, m):
    return utilsc.vratio(_stirling, n, m)

def wratio(n, m):
    return utilsc.wratio(_stirling, n, m)
