import gmmc

import numpy as np
import pdb
import typing, os, sys

from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces import base
from d3m import container, utils
import d3m.metadata
from d3m.metadata import hyperparams, base as metadata_base
from d3m.metadata import params

Inputs = container.DataFrame  # type: DataFrame
Outputs = container.DataFrame  # type: DataFrame
OutputCenters = container.ndarray  # type: np.ndarray

class Params(params.Params):
    mixture_parameters: bytes  # Byte stream represening coordinates of cluster centers.

class HyperParams(hyperparams.Hyperparams):
    k = hyperparams.UniformInt(lower=1, upper=10000, default=10, semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description='The number of clusters to form as well as the number of centroids to generate.')
    iters = hyperparams.UniformInt(lower=1, upper=10000, default=100, semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description='The number of iterations of inference.')
    initialization = hyperparams.Enumeration[str](values=['random', 'firstk', 'kmeanspp', 'covertree'], default='covertree', semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description="'random': choose k observations (rows) at random from data for the initial centroids. 'kmeanspp' : selects initial cluster centers by finding well spread out points using cover trees to speed up convergence. 'covertree' : selects initial cluster centers by sampling to speed up convergence.")


def init_covertree(k: int, points):
    import covertreec
    trunc = 3
    ptr = covertreec.new(points, trunc)
    #covertreec.display(ptr)
    seeds = covertreec.spreadout(ptr, k)
    covertreec.delete(ptr)
    return seeds
    
def init_kmeanspp(k: int, points):
    import utilsc
    seed_idx = utilsc.kmeanspp(k, points)
    seeds = points[seed_idx]
    return seeds
    

class GMM(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, HyperParams]):

    metadata = metadata_base.PrimitiveMetadata({
        "id": "49af9397-d9a2-450f-93eb-c3b631ba6646",
        "version": "1.0",
        "name": "Gaussian Mixture Models",
        "description": "This class provides functionality for unsupervised inference on Gaussian mixture model, which is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. It can be viewed as a generalization of the K-Means clustering to incorporate information about the covariance structure of the data. Standard packages, like those in scikit learn run on a single machine and often only on one thread. Whereas our underlying C++ implementation can be distributed to run on multiple machines. To enable the distribution through python interface is work in progress. In this class, we implement inference on (Bayesian) Gaussian mixture models using Canopy algorithm. The API is similar to sklearn.mixture.GaussianMixture. The class is pickle-able.",
        "python_path": "d3m.primitives.cmu.fastlvm.GMM",
        "primitive_family": metadata_base.PrimitiveFamily.CLUSTERING,
        "algorithm_types": [ "K_MEANS_CLUSTERING" ],
        "keywords": ["large scale Gaussian Mixture Models", "clustering"],
        "source": {
            "name": "CMU",
            "uris": [ "https://github.com/manzilzaheer/fastlvm.git" ]
        },
        "installation": [
        {
            "type": "PIP",
            "package_uri": 'git+https://github.com/manzilzaheer/fastlvm.git@{git_commit}#egg=fastlvm'.format(
                                          git_commit=utils.current_git_commit(os.path.dirname(__file__)))
        }
        ]
    })


    def __init__(self, *, hyperparams: HyperParams) -> None:
        #super(GMM, self).__init__()
        super().__init__(hyperparams = hyperparams)
        self._this = None
        self._k = hyperparams['k']
        self._iters = hyperparams['iters']
        self._initialization = hyperparams['initialization']

        self._training_inputs = None  # type: Inputs
        self._validation_inputs = None # type: Inputs
        self._fitted = False
        
        self.hyperparams = hyperparams
                        
        
    def __del__(self) -> None:
        if self._this is not None:
            gmmc.delete(self._this)

    def set_training_data(self, *, inputs: Inputs) -> None:
        """
        Sets training data for GMM.

        Parameters
        ----------
        training_inputs : Inputs
            A NxD DataFrame of data points for training.
        """
        training_inputs = inputs.values
        self._training_inputs = training_inputs
        self._validation_inputs = training_inputs

        initial_centres = None
        if self._initialization == 'random':
            idx = np.random.choice(training_inputs.shape[0], self._k, replace=False)
            initial_centres = training_inputs[idx]
        elif self._initialization == 'firstk':
            initial_centres = training_inputs[:self._k]
        elif self._initialization == 'kmeanspp':
            initial_centres = init_kmeanspp(self._k, training_inputs)
        elif self._initialization == 'covertree':
            initial_centres = init_covertree(self._k, training_inputs)
        else:
            raise NotImplementedError('This type of initial means is not implemented')
        initial_vars = 0.5/np.var(training_inputs, axis=0)
        self._this = gmmc.new(self._k, self._iters, initial_centres, initial_vars)
        
        self._fitted = False

    
    def fit(self) -> None:
        """
        Inference on the Gaussian mixture model
        """
        if self._fitted:
            return

        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        gmmc.fit(self._this, self._training_inputs, self._validation_inputs)
        self._fitted = True

    def get_call_metadata(self) -> bool:
        """
        Returns metadata about the last ``fit`` call if it succeeded

        Returns
        -------
        Status : bool
            True/false status of fitting.

        """
        return self._fitted
        
    def produce(self, *, inputs: Inputs) -> base.CallResult[Outputs]:
        """
        Finds the closest cluster for the given set of test points using the learned model.

        Parameters
        ----------
        inputs : Inputs
            A NxD DataFrame of data points.

        Returns
        -------
        Outputs
            The index of the cluster each sample belongs to.

        """
        results = gmmc.predict(self._this, inputs.values)
        output = container.DataFrame(results, generate_metadata=False, source=self)
        # output.metadata = inputs.metadata.clear(source=self, for_value=output, generate_metadata=True)

        return base.CallResult(output)

    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, timeout: float = None,
                      iterations: int = None) -> base.MultiCallResult:
        output = self.produce(inputs=inputs)
        result = {}
        for method in produce_methods:
            result[method] = output.value
        return base.MultiCallResult(result)

    def evaluate(self, *, inputs: Inputs) -> float:
        """
        Finds the score of learned model on a set of test points
        
        Parameters
        ----------
        inputs : Inputs
            A NxD DataFrame of data points.

        Returns
        -------
        score : float
            The log-likelihood on the supplied points.
        """
        return gmmc.evaluate(self._this, inputs.values)
 
    def produce_centers(self) -> OutputCenters:
        """
        Get current cluster means and variances for this model.

        Returns
        ----------
        means : numpy.ndarray
            A KxD matrix of cluster means.
        vars : numpy.ndarray
            A KxD matrix of cluster variances.
        """

        return gmmc.centers(self._this)
    
    def get_params(self) -> Params:
        """
        Get parameters of GMM.

        Parameters are basically the mixture parameters (mean and variance) in byte stream.

        Returns
        ----------
        params : Params
            A named tuple of parameters.
        """

        return Params(mixture_parameters=gmmc.serialize(self._this))

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters of GMM.

        Parameters are basically the mixture parameters (mean and variance) in byte stream.

        Parameters
        ----------
        params : Params
            A named tuple of parameters.
        """
        self._this = gmmc.deserialize(params['mixture_parameters'])

