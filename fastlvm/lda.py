import ldac

import numpy as np
import pdb
import typing

from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
import d3m_metadata
from d3m_metadata.metadata import PrimitiveMetadata
from d3m_metadata import hyperparams
from d3m_metadata import params


Inputs = d3m_metadata.container.List[d3m_metadata.container.ndarray]  # type: list of np.ndarray
Outputs = d3m_metadata.container.List[d3m_metadata.container.ndarray]  # type: list of np.ndarray
Predicts = d3m_metadata.container.ndarray  # type: np.ndarray

class Params(params.Params):
    topic_matrix: bytes  # Byte stream represening topics

class HyperParams(hyperparams.Hyperparams):
    k = hyperparams.UniformInt(lower=1, upper=10000, default=10, description='The number of clusters to form as well as the number of centroids to generate.')
    iters = hyperparams.UniformInt(lower=1, upper=10000, default=100, description='The number of iterations of inference.')
    vocab = hyperparams.UniformInt(lower=1, upper=1000000, default=1000, description='Vocab size.')

                

class LDA(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, HyperParams]):
    
    metadata = PrimitiveMetadata({
        "id": "f410b951-1cb6-481c-8d95-2d97b31d411d",
        "version": "1.0",
        "name": "Latent Dirichlet Allocation Topic Modelling",
        "description": "This class provides functionality for unsupervised inference on latent Dirichlet allocation, which is a probabilistic topic model of corpora of documents which seeks to represent the underlying thematic structure of the document collection. They have emerged as a powerful new technique of finding useful structure in an unstructured collection as it learns distributions over words. The high probability words in each distribution gives us a way of understanding the contents of the corpus at a very high level. In LDA, each document of the corpus is assumed to have a distribution over K topics, where the discrete topic distributions are drawn from a symmetric dirichlet distribution. Standard packages, like those in scikit learn are inefficient in addition to being limited to a single machine. Whereas our underlying C++ implementation can be distributed to run on multiple machines. To enable the distribution through python interface is work in progress. The API is similar to sklearn.decomposition.LatentDirichletAllocation.",
        "python_path": "d3m.primitives.cmu.fastlvm.LDA",
        "primitive_family": "CLUSTERING",
        "algorithm_types": [ "K_MEANS_CLUSTERING" ],
        "keywords": ["large scale Gaussian Mixture Models", "clustering"],
        "source": {
            "name": "CMU",
            "uris": [ "https://github.com/manzilzaheer/fastlvm.git" ]
        },
        "installation": [
        {
            "type": "PIP",
            "package_uri": "git+https://github.com/manzilzaheer/fastlvm.git@d3m"
        }
        ]
    })
    

    def __init__(self, *, hyperparams: HyperParams, random_seed: int = 0, docker_containers: typing.Union[typing.Dict[str, str], None] = None) -> None:
        #super(LDA, self).__init__()
        self._this = None
        self._k = hyperparams['k']
        self._iters = hyperparams['iters']
        self._vocab = hyperparams['vocab']

        self._training_inputs = None  # type: Inputs
        self._validation_inputs = None # type: Inputs
        self._fitted = False
        self._ext = None

        self.hyperparams = hyperparams
        self.random_seed = random_seed
        self.docker_containers = docker_containers
        

    def __del__(self):
        if self._this is not None:
            ldac.delete(self._this, self._ext)
        
    def set_training_data(self, *, training_inputs: Inputs, validation_inputs: Inputs) -> None:
        """
        Sets training data for LDA.

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
        self._this = ldac.new(self._k, self._iters, vocab)
        
        self._fitted = False

    
    def fit(self) -> None:
        """
        Inference on the latent Dirichley allocation model
        """
        if self._fitted:
            return

        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        ldac.fit(self._this, self._training_inputs, self._validation_inputs)
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
        
    def produce(self, *, inputs: Inputs) -> Outputs:
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
        return ldac.predict(self._this, inputs)

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
        return ldac.evaluate(self._this, inputs)
 
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

        return ldac.top_words(self._this, num_top)

    def produce_topic_matrix(self) -> Predicts:
        """
        Get current word|topic distribution matrix for this model.

        Returns
        ----------
        topic_matrix : numpy.ndarray
            A numpy array of shape (vocab_size,k) with each column containing the word|topic distribution.
        """

        if self._ext is None:
            self._ext = ldac.topic_matrix(self._this)
        return self._ext
    
    def get_params(self) -> Params:
        """
        Get parameters of LDA.

        Parameters are basically the topic matrix in byte stream.

        Returns
        ----------
        params : Params
            A named tuple of parameters.
        """

        return Params(topic_matrix=ldac.serialize(self._this))

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters of LDA.

        Parameters are basically the topic matrix in byte stream.

        Parameters
        ----------
        params : Params
            A named tuple of parameters.
        """
        self.this = ldac.deserialize(params['topic_matrix'])

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

