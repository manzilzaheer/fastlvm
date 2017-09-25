import covertreec

import numpy as np
import pdb
from typing import NamedTuple
from typing import Union
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

Inputs = np.ndarray  # type: np.ndarray
Outputs = np.ndarray  # type: np.ndarray
Params = NamedTuple('Params', [
    ('tree', np.ndarray),  # Byte stream represening the tree.
])

class CoverTree(object):
    """
    This class provides functionality for unsupervised nearest neighbor search, which is the foundation of many other learning methods, notably manifold learning and spectral clustering. The goal is to find a number of points from the given database closest in distance to the query point. The distance can be, in general, any metric measure: standard Euclidean distance is the most common choice and the one currently implemented. In future, adding other metrics should not be too difficult. Standard packages, like those in scikit learn use KD-tree or Ball trees, which do not scale very well, especially with respect to dimension. For example, Ball Trees of scikit learn takes O(n2) construction time and a search query can be linear in worst case making it no better than brute force in some cases. In this class, we implement a modified version of the Cover Tree data structure that allow fast retrieval in logarithmic time. The key properties are: O(n log n) construction time, O(log n) retrieval, and polynomial dependence on the expansion constant of the underlying space. In addition, it allows insertion and removal of points in database. The class is pickle-able.
    """

    __author__ = 'CMU'
    __metadata__ = {
        "common_name": "Nearest Neighbor Search with Cover Trees",
        "algorithm_type": ["Instance Based"],
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
            "expected_running_time": [10, 1000],
            "gpus_per_node": [0, 0],
            "cores_per_node": [32, 32],
            "mem_per_gpu": [0, 0],
            "mem_per_node": [1, 12],
            "num_nodes": [1, 1],
        },
    }

    def __init__(self, *, trunc: int = -1) -> None:
        super(CoverTree, self).__init__()
        self.this = None
        self.trunc = trunc
        self.training_inputs = None  # type: Inputs
        self.fitted = False

    def __del__(self):
        if self.this is not None:
             covertreec.delete(self.this)
     
    def set_training_data(self, *, inputs: Inputs) -> None:
        """
        Sets training data for CoverTree.

        Parameters
        ----------
        inputs : Inputs
            A NxD matrix of data points for training.
        """

        self.training_inputs = inputs
        self.fitted = False
        
    def fit(self) -> None:
        """
        Construct the tree
        """
        if self.fitted:
            return

        if self.training_inputs is None:
            raise ValueError("Missing training data.")

        self.this = covertreec.new(self.training_inputs, self.trunc)
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

    def produce(self, *, inputs: Inputs, k: int = 3) -> Outputs:
        """
        Finds the closest points for the given set of test points using the tree constructed.

        Parameters
        ----------
        inputs : Inputs
            A NxD matrix of data points.

        Returns
        -------
        Outputs
            The k nearest neighbours of each point.

        """
        if self.this is None:
            raise ValueError('Fit model first')

        if k == 1:
            results = covertreec.NearestNeighbour(self.this, inputs)
        else:
            results = covertreec.kNearestNeighbours(self.this, inputs, k)
        
        return results

    def get_params(self) -> Params:
        """
        Get parameters of KMeans.

        Parameters are basically the cluster centres in byte stream.

        Returns
        ----------
        params : Params
            A named tuple of parameters.
        """

        return Params(tree=covertreec.serialize(self.this))

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters of cover tree.

        Parameters are basically the tree structure in byte stream.

        Parameters
        ----------
        params : Params
            A named tuple of parameters.
        """
        self.this = covertreec.deserialize(params.tree)

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
