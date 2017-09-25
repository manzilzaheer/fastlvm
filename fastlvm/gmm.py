import gmmc
import numpy as np
import pdb
    
from typing import NamedTuple
from typing import Union
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

Inputs = np.ndarray  # type: np.ndarray
Outputs = np.ndarray  # type: np.ndarray
Params = NamedTuple('Params', [
    ('mixture_parameters', np.ndarray),  # Byte stream represening coordinates of cluster centers.
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
    

class GMM(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params]):
    """
    This class provides functionality for unsupervised inference on Gaussian mixture model, which is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. It can be viewed as a generalization of the K-Means clustering to incorporate information about the covariance structure of the data. Standard packages, like those in scikit learn run on a single machine and often only on one thread. Whereas our underlying C++ implementation can be distributed to run on multiple machines. To enable the distribution through python interface is work in progress. In this class, we implement inference on (Bayesian) Gaussian mixture models using Canopy algorithm. The API is similar to sklearn.mixture.GaussianMixture. The class is pickle-able.
    """

    __author__ = 'CMU'
    __metadata__ = {
        "common_name": "Gaussian Mixture Models",
        "algorithm_type": ["Bayesian","Clustering","Probabilistic Graphical Models"],
        "handles_classification": False,
        "handles_regression": False,
        "handles_multiclass": False,
        "handles_multilabel": False,
        "input_type": ["DENSE"],
        "output_type": ["PREDICTIONS"],
        "schema_version": 1.0,
        "compute_resources": {
            "sample_size": [],
            "sample_unit": [],
            "disk_per_node": [],
            "expected_running_time": [],
            "gpus_per_node": [],
            "cores_per_node": [],
            "mem_per_gpu": [],
            "mem_per_node": [],
            "num_nodes": [],
        },
    }

    def __init__(self, *, k: int = 10, iters: int = 100, initial_centres: Union[str, np.ndarray] = 'covertree', data: Union[None, np.ndarray] = None, initial_vars: Union[None, np.ndarray] = None) -> None:
        super(GMM, self).__init__()
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

        self.this = gmmc.new(k, iters, initial_centres, initial_vars)
        self.training_inputs = None  # type: Inputs
        self.validation_inputes = None # type: Inputs
        self.fitted = False

    def __del__(self) -> None:
        if self.this is not None:
            gmmc.delete(self.this)

    def set_training_data(self, *, training_inputs: Inputs, validation_inputs: Inputs = None) -> None:
        """
        Sets training data for GMM.

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
        Inference on the Gaussian mixture model
        """
        if self.fitted:
            return

        if self.training_inputs is None:
            raise ValueError("Missing training data.")

        gmmc.fit(self.this, self.training_inputs, self.validation_inputes)
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
        return gmmc.predict(self.this, inputs)

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
            The log-likelihood on the supplied points.
        """
        return gmmc.evaluate(self.this, inputs)
 
    def get_centers(self) -> Outputs:
        """
        Get current cluster means and variances for this model.

        Returns
        ----------
        means : numpy.ndarray
            A KxD matrix of cluster means.
        vars : numpy.ndarray
            A KxD matrix of cluster variances.
        """

        return gmmc.centers(self.this)
    
    def get_params(self) -> Params:
        """
        Get parameters of GMM.

        Parameters are basically the mixture parameters (mean and variance) in byte stream.

        Returns
        ----------
        params : Params
            A named tuple of parameters.
        """

        return Params(mixture_parameters=gmmc.serialize(self.this))

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters of GMM.

        Parameters are basically the mixture parameters (mean and variance) in byte stream.

        Parameters
        ----------
        params : Params
            A named tuple of parameters.
        """
        self.this = gmmc.deserialize(params.mixture_parameters)

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

