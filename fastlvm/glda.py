import gldac

import numpy as np
import pdb
import typing, os, sys

from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
import d3m_metadata
from d3m_metadata.metadata import PrimitiveMetadata
from d3m_metadata import hyperparams, utils
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

    
class GLDA(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, HyperParams]):

    metadata = PrimitiveMetadata({
        "id": "a3d490a4-ef39-4de1-be02-4c43726b3b24",
        "version": "1.0",
        "name": "Gaussian Latent Dirichlet Allocation Topic Modelling",
        "description": "This class provides functionality for unsupervised inference on Gaussian latent Dirichlet allocation, which replace LDA's parameterization of 'topics' as categorical distributions over opaque word types with multivariate Gaussian distributions on the embedding space. This encourages the model to group words that are a priori known to be semantically related into topics, as continuous space word embeddings learned from large, unstructured corpora have been shown to be effective at capturing semantic regularities in language. Using vectors learned from a domain-general corpus (e.g. English Wikipedia), qualitatively, Gaussian LDA infers different (but still very sensible) topics relative to standard LDA. Quantitatively, the technique outperforms existing models at dealing with OOV words in held-out documents. No standard packages exists. Our underlying C++ implementation can be distributed to run on multiple machines. To enable the distribution through python interface is work in progress. In this class, we implement inference on Gaussian latent Dirichlet Allocation using Canopy algorithm. In case of full covariance matrices, it exploits the Cholesky decompositions of covariance matrices of the posterior predictive distributions and performs efficient rank-one updates. The API is similar to sklearn.decomposition.LatentDirichletAllocation.",
        "python_path": "d3m.primitives.cmu.fastlvm.GLDA",
        "primitive_family": "CLUSTERING",
        "algorithm_types": [ "LATENT_DIRICHLET_ALLOCATION" ],
        "keywords": ["large scale Gaussian LDA", "topic modeling", "clustering"],
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

    
    def __init__(self, *, hyperparams: HyperParams, random_seed: int = 0, docker_containers: typing.Union[typing.Dict[str, str], None] = None) -> None:
        #super(GLDA, self).__init__()
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
            gldac.delete(self._this, self._ext)
        
    def set_training_data(self, *, training_inputs: Inputs, validation_inputs: Inputs, vectors: Predicts) -> None:
        """
        Sets training data for GLDA.

        Parameters
        ----------
        training_inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id.
        validation_inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id. This represents validation docs to validate the results learned after each iteration of canopy algorithm.
        vectors : Predicts
           A contiguous numpy array of shape (vocab_size, dim) containing the continuous word embeddings. The order of the vectors should match the order of words in the vocab.
        """

        self._training_inputs = training_inputs
        self._validation_inputs = validation_inputs

        vocab = [''.join(['w',str(i)]) for i in range(self._vocab)]
        self._this = gldac.new(self._k, self._iters, vocab, vectors)
        
        self._fitted = False

    
    def fit(self) -> None:
        """
        Inference on the Gaussian latent Dirichley allocation model
        """
        if self._fitted:
            return

        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        gldac.fit(self._this, self._training_inputs, self._validation_inputs)
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
        return gldac.predict(self._this, inputs)

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
        return gldac.evaluate(self._this, inputs)
 
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

        return gldac.top_words(self._this, num_top)

    def produce_topic_matrix(self) -> np.ndarray:
        """
        Get current word|topic distribution matrix for this model.

        Returns
        ----------
        topic_matrix : numpy.ndarray
            A numpy array of shape (vocab_size,k) with each column containing the word|topic distribution.
        """

        if self._ext is None:
            self._ext = gldac.topic_matrix(self._this)
        return self._ext
    
    def get_params(self) -> Params:
        """
        Get parameters of GLDA.

        Parameters are basically the topic matrix in byte stream.

        Returns
        ----------
        params : Params
            A named tuple of parameters.
        """

        return Params(topic_matrix=gldac.serialize(self._this))

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters of LDA.

        Parameters are basically the topic matrix in byte stream.

        Parameters
        ----------
        params : Params
            A named tuple of parameters.
        """
        self._this = gldac.deserialize(params['topic_matrix'])

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
