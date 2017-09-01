import time
import pickle
import numpy as np
import scipy as sc
from tqdm import trange
from fastlvm import Search, Clustering, GMM, LDA, GLDA
from fastlvm import read_corpus
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

gt = time.time
np.random.seed(seed=3)

print('Generate random points')
N=100
K=100
D=1000
means = 20*np.random.rand(K,D) - 10
x = np.vstack([np.random.randn(N,D) + means[i] for i in range(K)])
np.random.shuffle(x)
with open('train_data.bin', 'wb') as f:
    np.array(x.shape, dtype='int32').tofile(f)
    x.tofile(f)
x = np.require(x, requirements=['A', 'C', 'O', 'W'])
#print(x[0,0], x[0,1], x[1,0])
y = np.vstack([np.random.randn(N//10,D) + means[i] for i in range(K)])
with open('test_data.bin', 'wb') as f:
    np.array(y.shape, dtype='int32').tofile(f)
    y.tofile(f)
y = np.require(y, requirements=['A', 'C', 'O', 'W'])

print('======== Checks for Search ==========')

t = gt()
ct = Search.from_matrix(x)
b_t = gt() - t
#ct.display()
print("Building time:", b_t, "seconds")
    
print("Test covering: ", ct.test_covering())

print('Test Nearest Neighbour: ')
t = gt()
a = ct.NearestNeighbour(y)
b_t = gt() - t
print("Query time:", b_t, "seconds")
nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(x)
distances, indices = nbrs.kneighbors(y)
b = np.squeeze(x[indices])
if np.all(a==b):
    print("Test for Nearest Neighbour passed")
else:
    print("Test for Nearest Neighbour failed")
print()

print('Test k-Nearest Neighbours (k=3): ')
t = gt()
a = ct.kNearestNeighbours(y,3)
b_t = gt() - t
print("Query time:", b_t, "seconds")
nbrs = NearestNeighbors(n_neighbors=3, algorithm='brute').fit(x)
distances, indices = nbrs.kneighbors(y)
if np.all(a==x[indices]):
    print("Test for k-Nearest Neighbours passed")
else:
    print("Test for k-Nearest Neighbours failed")
print()

print('Test pickling: ')
#s = ct.serialize()
#f_string = pickle.dumps(ct, protocol=4)
#print('string got', len(f_string))
with open('tree.dat', 'wb') as f:
    pickle.dump(ct, f, protocol=4)
ct = None
#ct_new = Search.from_string(s)
#ct_new = pickle.loads(f_string)
with open('tree.dat', 'rb') as f:
   ct_new =  pickle.load(f)
#ct_new.display()
print("Tree reconstructed")
a = ct_new.kNearestNeighbours(y,3)
if np.all(a==x[indices]):
    print("Test for pickling passed")
else:
    print("Test for pickling failed")
    
print('======== Checks for Clustering ==========')
print('Building clustering data structures')
skm = KMeans(K, 'k-means++', 1, 10, verbose=0)
ctm = Clustering.init(K, 10, 'covertree', x)

t = gt()
ct = ctm.fit(x,y)
b_t = gt() - t
#ct.display()
print("Training time:", b_t, "seconds")
skm.fit(x,y)

a = ctm.evaluate(y)
b = skm.score(y)
print('Canopy score: ', a)
print('Sklearn score: ', b)
print('Difference: ', a-b)
print()

print('Cluster centres')
cc = ctm.get_centers()

print('Test pickling: ')
#s = ctm.serialize()
f_string = pickle.dumps(ctm)
#print('string got', len(f_string), len(ctm.serialize()))
ctm = None
#ct_new = Clustering.from_string(s)
ct_new = pickle.loads(f_string)
#ct_new.display()
print("Tree reconstructed")
a_new = ct_new.evaluate(y)
if np.abs(a_new - a) < 1e-9*np.abs(a):
    print("Test for pickling passed")
else:
    print("Test for pickling failed")

print('======== Checks for GMM ==========')
print('Building clustering data structures')
skm = GaussianMixture(K, covariance_type='diag', max_iter=10, init_params='kmeans', verbose=0)
ctm = GMM.init(K, 10, 'covertree', x)

t = gt()
ct = ctm.fit(x,y)
b_t = gt() - t
#ct.display()
print("Training time:", b_t, "seconds")
skm.fit(x,y)

a = ctm.evaluate(y)
b = skm.score(y)
print('Canopy score: ', a)
print('Sklearn score: ', b)
print('Difference: ', a-b)
print()

print('Cluster centres')
cc = ctm.get_centers()

print('Test pickling: ')
#s = ctm.serialize()
f_string = pickle.dumps(ctm)
#print('string got', len(f_string), len(ctm.serialize()))
ctm = None
#ct_new = Clustering.from_string(s)
ct_new = pickle.loads(f_string)
#ct_new.display()
print("Tree reconstructed")
a_new = ct_new.evaluate(y)
if np.abs(a_new - a) < 1e-9*np.abs(a):
    print("Test for pickling passed")
else:
    print("Test for pickling failed")

print('======== Checks for LDA ==========')
# Load NIPS data
trngdata, vocab = read_corpus('data/nips.train')
testdata, vocab = read_corpus('data/nips.test', vocab)

# Init LDA model
canlda = LDA.init(10, 100, vocab)

# Train LDA model
canlda.fit(trngdata, testdata)

# Get topic matrix
tm = canlda.get_topic_matrix()

# Test on held out data using learned model
a = canlda.evaluate(testdata)

# Read word|topic distribution from gensim
with open('data/lda_gensim.pkl', 'rb') as f:
    m = pickle.load(f)
np.copyto(tm, m)

# Test on held out data using gensim model
b = canlda.evaluate(testdata)

print('Canopy score: ', a)
print('Gensim score: ', b)
print('Difference: ', a-b)

print('Test pickling: ')
#s = canlda.serialize()
f_string = pickle.dumps(canlda)
#print('string got', len(f_string), len(ctm.serialize()))
canlda = None
#ct_new = LDA.from_string(s)
ct_new = pickle.loads(f_string)
#ct_new.display()
print("Tree reconstructed")
b_new = ct_new.evaluate(testdata)
if np.abs(b_new - b) < 1e-2*np.abs(a):
    print("Test for pickling passed")
else:
    print("Test for pickling failed")
    

print('======== Checks for GLDA ==========')
# Load 20 News data
with open('data/20_news.pkl', 'rb') as f:
    d = pickle.load(f)
    # {'trngdata':trngdata, 'testdata':testdata, 'word_map':word_map, 'word_vec':word_vec}

# Init GLDA model
canglda = GLDA.init(10, 10, d['word_map'], d['word_vec'])

# Train GLDA model
canglda.fit(d['trngdata'], d['testdata'])

# Get topic matrix
tm = canglda.get_topic_matrix()

# Test on held out data using learned model
b = canglda.evaluate(d['testdata'])
print('Canopy score: ', b)

print('Test pickling: ')
#s = canlda.serialize()
f_string = pickle.dumps(canglda)
#print('string got', len(f_string), len(canglda.serialize()))
canlda = None
#ct_new = GLDA.from_string(s)
ct_new = pickle.loads(f_string)
#ct_new.display()
print("Tree reconstructed")
b_new = ct_new.evaluate(d['testdata'])
if np.abs(b_new - b) < 1e-2*np.abs(b):
    print("Test for pickling passed")
else:
    print("Test for pickling failed")