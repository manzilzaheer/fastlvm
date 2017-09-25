import kmeansc
import numpy as np
import pdb

from typing import NamedTuple
from typing import Union
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

Inputs = np.ndarray  # type: np.ndarray
Outputs = np.ndarray  # type: np.ndarray
Params = NamedTuple('Params', [
    ('cluster_centers', np.ndarray),  # Byte stream represening coordinates of cluster centers.
])


    
def init_covertree(k: int, points: Inputs) -> Outputs:
    import covertreec
    trunc = 3
    ptr = covertreec.new(points, trunc)
    #covertreec.display(ptr)
    seeds = covertreec.spreadout(ptr, k)
    covertreec.delete(ptr)
    return seeds
    
def init_kmeanspp(k: int, points: Inputs) -> Outputs:
    import utilsc
    seed_idx = utilsc.kmeanspp(k, points)
    seeds = points[seed_idx]
    return seeds
    

class KMeans(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params]):
    """
    This class provides functionality for unsupervised clustering, which according to Wikipedia is "the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar to each other than to those in other groups". It is a main task of exploratory data mining, and a common technique for statistical data analysis. The similarity measure can be, in general, any metric measure: standard Euclidean distance is the most common choice and the one currently implemented. In future, adding other metrics should not be too difficult. Standard packages, like those in scikit learn run on a single machine and often only on one thread. Whereas our underlying C++ implementation can be distributed to run on multiple machines. To enable the distribution through python interface is work in progress. In this class, we implement a K-Means clustering using Llyod's algorithm and speed-up using Cover Trees. The API is similar to sklearn.cluster.KMeans. The class is pickle-able.
    """

    __author__ = 'CMU'
    __metadata__ = {
        "common_name": "K-means Clustering",
        "algorithm_type": ["Clustering", "Instance Based"],
        "handles_classification": False,
        "handles_regression": False,
        "handles_multiclass": False,
        "handles_multilabel": False,
        "input_type": ["DENSE"],
        "output_type": ["PREDICTIONS"],
        "schema_version": 1.0,
        "compute_resources": {
            "sample_size": [400, 8],
            "sample_unit": ["MB", "GB"],
            "disk_per_node": [0, 0],
            "expected_running_time": [4, 36],
            "gpus_per_node": [0, 0],
            "cores_per_node": [32, 32],
            "mem_per_gpu": [0, 0],
            "mem_per_node": [1.1, 12],
            "num_nodes": [1, 1],
        },
    }


    def __init__(self, *, k: int = 10, iters: int = 100, initial_centres: Union[str, np.ndarray] = 'covertree', data: Union[None, np.ndarray] = None) -> None:
        super(KMeans, self).__init__()
        self.this = None
        
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
        
        self.this = kmeansc.new(k, iters, initial_centres)
        self.training_inputs = None  # type: Inputs
        self.validation_inputes = None # type: Inputs
        self.fitted = False

    def __del__(self) -> None:
        if self.this is not None:
            kmeansc.delete(self.this)

    def set_training_data(self, *, training_inputs: Inputs, validation_inputs: Inputs = None) -> None:
        """
        Sets training data for KMeans.

        Parameters
        ----------
        training_inputs : Inputs
            A NxD matrix of data points for training.
        validation_inputs : Inputs
            A N'xD matrix of data points for validaton.
        """

        self.training_inputs = training_inputs
        self.validation_inputes = validation_inputs
        self.fitted = False

    def fit(self) -> None:
        """
        Compute k-means clustering
        """
        if self.fitted:
            return

        if self.training_inputs is None:
            raise ValueError("Missing training data.")

        kmeansc.fit(self.this, self.training_inputs, self.validation_inputes)
        self.fitted = True

    def get_call_metadata(self) -> bool:
        """
        Returns metadata about the last ``fit`` call if it succeeded

        Returns
        -------
        Status : bool
            True/false status of fitting.

        """
        return self.fitted

    def produce(self, *, inputs: Inputs) -> Outputs:
        """
        Finds the closest cluster for the given set of test points using the learned model.

        Parameters
        ----------
        inputs : Inputs
            A NxD matrix of data points.

        Returns
        -------
        Outputs
            The index of the cluster each sample belongs to.

        """
        return kmeansc.predict(self.this, inputs)

    def evaluate(self, *, inputs: Inputs) -> float:
        """
        Finds the score of learned model on a set of test points
        
        Parameters
        ----------
        inputs : Inputs
            A NxD matrix of data points.

        Returns
        -------
        score : float
            The score (-ve of K-Means objective value) on the supplied points.
        """
        return kmeansc.evaluate(self.this, inputs)
        
    def get_centers(self) -> Outputs:
        """
        Get current cluster centers for this model.

        Returns
        ----------
        centers : numpy.ndarray
            A KxD matrix of cluster centres.
        """

        return kmeansc.centers(self.this)
    
    def get_params(self) -> Params:
        """
        Get parameters of KMeans.

        Parameters are basically the cluster centres in byte stream.

        Returns
        ----------
        params : Params
            A named tuple of parameters.
        """

        return Params(cluster_centers=kmeansc.serialize(self.this))

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters of KMeans.

        Parameters are basically the cluster centres in byte stream.

        Parameters
        ----------
        params : Params
            A named tuple of parameters.
        """
        self.this = kmeansc.deserialize(params.cluster_centers)

    def set_random_seed(self, *, seed: int) -> None:
        """
        NOT SUPPORTED YET
        Sets a random seed for all operations from now on inside the primitive.

        By default it sets numpy's and Python's random seed.

        Parameters
        ----------
        seed : int
            A random seed to use.
        """

        raise NotImplementedError("Not supported yet")

