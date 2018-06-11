import hdpc

import numpy as np
import pdb
import typing, os, sys

from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces import base
from d3m import container, utils
import d3m.metadata
from d3m.metadata import hyperparams, base as metadata_base
from d3m.metadata import params

Inputs = container.List  # type: list of np.ndarray
Outputs = container.List  # type: list of np.ndarray
Predicts = container.ndarray  # type: np.ndarray

class Params(params.Params):
    topic_matrix: bytes  # Byte stream represening topics

class HyperParams(hyperparams.Hyperparams):
    k = hyperparams.UniformInt(lower=1, upper=10000, default=10, description='The number of clusters to form as well as the number of centroids to generate.')
    iters = hyperparams.UniformInt(lower=1, upper=10000, default=100, description='The number of iterations of inference.')
    vocab = hyperparams.UniformInt(lower=1, upper=1000000, default=1000, description='Vocab size.')
                        

class HDP(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, HyperParams]):
    """
    This class provides functionality for Hierarchical Dirichlet Process, which is a nonparametric Bayesian model for topic modelling on corpora of documents which seeks to represent the underlying thematic structure of the document collection. They have emerged as a powerful new technique of finding useful structure in an unstructured collection as it learns distributions over words. The high probability words in each distribution gives us a way of understanding the contents of the corpus at a very high level. In HDP, each document of the corpus is assumed to have a distribution over K topics, where the discrete topic distributions are drawn from a symmetric dirichlet distribution. As it is a nonparametric model, the number of topics K is inferred automatically. The API is similar to its parametric equivalent sklearn.decomposition.LatentDirichletAllocation. The class is pickle-able.
    """

    __author__ = 'CMU'
    __metadata__ = {
        "common_name": "Hierarchical Dirichlet Process Topic Modelling",
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

    metadata = metadata_base.PrimitiveMetadata({
        "id": "e582e738-2f7d-4b5d-964f-022d15f19018",
        "version": "1.0",
        "name": "Hierarchical Dirichlet Process Topic Modelling",
        "description": "This class provides functionality for Hierarchical Dirichlet Process, which is a nonparametric Bayesian model for topic modelling on corpora of documents which seeks to represent the underlying thematic structure of the document collection. They have emerged as a powerful new technique of finding useful structure in an unstructured collection as it learns distributions over words. The high probability words in each distribution gives us a way of understanding the contents of the corpus at a very high level. In HDP, each document of the corpus is assumed to have a distribution over K topics, where the discrete topic distributions are drawn from a symmetric dirichlet distribution. As it is a nonparametric model, the number of topics K is inferred automatically. The API is similar to its parametric equivalent sklearn.decomposition.LatentDirichletAllocation. The class is pickle-able.",
        "python_path": "d3m.primitives.cmu.fastlvm.HDP",
        "primitive_family": "CLUSTERING",
        "algorithm_types": [ "LATENT_DIRICHLET_ALLOCATION" ],
        "keywords": ["large scale HDP", "Bayesian Nonparametrics", "topic modeling", "clustering"],
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
        #super(HDP, self).__init__()
        super().__init__(hyperparams = hyperparams)
        self._this = None
        self._k = hyperparams['k']
        self._iters = hyperparams['iters']
        self._vocab = hyperparams['vocab']

        self._training_inputs = None  # type: Inputs
        self._validation_inputs = None # type: Inputs
        self._fitted = False
        self._ext = None

        self.hyperparams = hyperparams
        
        
    def __del__(self):
        if self._this is not None:
            hdpc.delete(self._this, self._ext)
        
    def set_training_data(self, *, training_inputs: Inputs, validation_inputs: Inputs) -> None:
        """
        Sets training data for HDP.

        Parameters
        ----------
        training_inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id.
        validation_inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id. This represents validation docs to validate the results learned after each iteration of canopy algorithm.
        """

        self._training_inputs = training_inputs
        self._validation_inputs = validation_inputs

        vocab = [''.join(['w',str(i)]) for i in range(self._vocab)]
        self._this = hdpc.new(self._k, self._iters, vocab)
        
        self._fitted = False

    
    def fit(self) -> None:
        """
        Inference on the hierarchical Dirichlet process model
        """
        if self._fitted:
            return

        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        hdpc.fit(self._this, self._training_inputs, self._validation_inputs)
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
        Finds the token topic assignment (and consequently topic-per-document distribution) for the given set of docs using the learned model.

        Parameters
        ----------
        inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id.

        Returns
        -------
        Outputs
            A list of 1d numpy array which represents index of the topic each token belongs to.

        """
        return base.CallResult(hdpc.predict(self._this, inputs))

    def evaluate(self, *, inputs: Inputs) -> float:
        """
        Finds the per-token log likelihood (-ve log perplexity) of learned model on a set of test docs.
        
        Parameters
        ----------
        inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id. This represents test docs to test the learned model.

        Returns
        -------
        score : float
            Final per-token log likelihood (-ve log perplexity).
        """
        return hdpc.evaluate(self._this, inputs)
 
    def produce_top_words(self, *, num_top: int) -> Outputs:
        """
        Get the top words of each topic for this model.

        Parameters
        ----------
        num_top : int
            The number of top words requested..

        Returns
        ----------
        topic_matrix : list
            A list of size k containing list of size num_top words.
        """

        return hdpc.top_words(self._this, num_top)

    def produce_topic_matrix(self) -> np.ndarray:
        """
        Get current word|topic distribution matrix for this model.

        Returns
        ----------
        topic_matrix : numpy.ndarray
            A numpy array of shape (vocab_size,k) with each column containing the word|topic distribution.
        """

        if self._ext is None:
            self._ext = hdpc.topic_matrix(self._this)
        return self._ext
   
    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, timeout: float = None, iterations: int = None, num_top: int) -> base.MultiCallResult:
	    pass 

    def get_params(self) -> Params:
        """
        Get parameters of HDP.

        Parameters are basically the topic matrix in byte stream.

        Returns
        ----------
        params : Params
            A named tuple of parameters.
        """

        return Params(topic_matrix=hdpc.serialize(self._this))

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters of HDP.

        Parameters are basically the topic matrix in byte stream.

        Parameters
        ----------
        params : Params
            A named tuple of parameters.
        """
        self._this = hdpc.deserialize(params['topic_matrix'])

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
