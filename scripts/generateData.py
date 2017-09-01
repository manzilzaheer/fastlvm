# Script to generate testing GMM data
import sys
import numpy as np
from sklearn import mixture

if len(sys.argv) != 5:
    print "Usage: python scripts/generateData.py <numDims> <numClusters> <numPoints> <numDataPacks>"
    sys.exit(-1)

version = 1		# version of the datapack

numDims = int(sys.argv[1])	# number of dimensions
numClusters = int(sys.argv[2])	# number of clusters
numPoints = int(sys.argv[3])	# per cluster points
numDataPacks = int(sys.argv[4])	# number of datapacks

# Generate the cluster proportion, means and variance first, try not to use Gaussians!!!
print "Generating cluster parameters"
pis = (1./numClusters) * np.ones(numClusters)  # currently fixed
means = 200 * np.random.rand(numClusters, numDims) - 100
# Variable stds
#stds =  2 * abs(np.random.rand(numClusters, numDims))
# Fixed stds
stds = 64 * np.ones((numClusters, numDims)) # abs(np.random.rand(numClusters, numDims))
variances = np.square(stds)

# writing down the ground truth
with open('data/synthetic_%d_%d_%d-gt.dat' % (numDims, numClusters, numPoints), 'w') as filePt:
    # Headers (version, number of clusters and number of dimensions)
    np.array(version, dtype='int32').tofile(filePt)
    np.array(numDims, dtype='int32').tofile(filePt)
    np.array(numClusters, dtype='int32').tofile(filePt)

    # now write the pis, means, vars
    pis.tofile(filePt)
    means.tofile(filePt)
    variances.tofile(filePt)


# Train data generation
for j in xrange(numDataPacks):
    print "Generating data pack %d" % j
    train = []
    for i in xrange(numClusters):
        train.append( means[i] + stds[i] * np.random.randn(numPoints, numDims) )

    train = np.vstack(train)

    # shuffle the points
    np.random.shuffle(train)
    
    # just a hack: first point from unique clusters
    # for i in xrange(numClusters):
    #     train[i] =  means[i] #+ 2 * np.random.randn(numDims)

    # Open a file and write the points
    with open('data/synthetic_%d_%d_%d-%d.dat' % (numDims, numClusters, numPoints, j), 'w') as filePt:
        # Headers (version, number of dimensions and number of points)
        np.array(version, dtype='int32').tofile(filePt)
        np.array(numDims, dtype='int32').tofile(filePt)
        np.array(numPoints * numClusters, dtype='int32').tofile(filePt)

        # now write the points
        train.tofile(filePt);

# Generate test data
print "Generating test data set"
numTest = int(numPoints)#Train-test split
test = []
for i in xrange(numClusters):
    test.append(  means[i] + stds[i] * np.random.randn(numTest, numDims) );
test = np.vstack(test)

# Open a file and write the points
with open('data/synthetic_%d_%d_%d-test.dat' % (numDims, numClusters, numPoints), 'w') as filePt:
    # Headers (version, number of dimensions and number of points)
    np.array(version, dtype='int32').tofile(filePt)
    np.array(numDims, dtype='int32').tofile(filePt)
    np.array(numTest * numClusters, dtype='int32').tofile(filePt)

    # now write the points
    test.tofile(filePt);


# Try scikit-learn GMM to have point of reference
g = mixture.GMM(n_components=numClusters, verbose=1)
#g.fit(train)
#skProb = np.sum(g.score(test))
#print 'Loglikelihood (sklearn) : %f' % skProb

g.weights_ = pis
g.means_ = means
g.covars_ = variances
gtProb = np.sum(g.score(test))

print 'Loglikelihood (true) : %f' % gtProb
