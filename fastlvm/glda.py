import gldac
import numpy as np
import pdb
    
from typing import NamedTuple
from typing import Union
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

Inputs = list  # type: np.ndarray
Outputs = list  # type: np.ndarray
Params = NamedTuple('Params', [
    ('topic_matrix', np.ndarray),  # Byte stream represening topics
])

class GLDA(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params]):
    """
    This class provides functionality for unsupervised inference on Gaussian latent Dirichlet allocation, which replace LDA's parameterization of 'topics' as categorical distributions over opaque word types with multivariate Gaussian distributions on the embedding space. This encourages the model to group words that are a priori known to be semantically related into topics, as continuous space word embeddings learned from large, unstructured corpora have been shown to be effective at capturing semantic regularities in language. Using vectors learned from a domain-general corpus (e.g. English Wikipedia), qualitatively, Gaussian LDA infers different (but still very sensible) topics relative to standard LDA. Quantitatively, the technique outperforms existing models at dealing with OOV words in held-out documents. No standard packages exists. Our underlying C++ implementation can be distributed to run on multiple machines. To enable the distribution through python interface is work in progress. In this class, we implement inference on Gaussian latent Dirichlet Allocation using Canopy algorithm. In case of full covariance matrices, it exploits the Cholesky decompositions of covariance matrices of the posterior predictive distributions and performs efficient rank-one updates. The API is similar to sklearn.decomposition.LatentDirichletAllocation.
    """
    
    __author__ = 'CMU'
    __metadata__ = {
        "common_name": "Gaussian Latent Dirichlet Allocation Topic Modelling",
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

    def __init__(self, *, k: int = 10, iters: int = 100, vocab: Union[list, int], vectors: np.ndarray) -> None:
        super(GLDA, self).__init__()
        self.this = None

        if isinstance(vocab, int):
            if vocab < 0:
                raise ValueError('Vocab size must be non-negative!')
            vocab = [''.join(['w',i]) for i in range(vocab)]
        elif isinstance(vocab, list):
            if len(vocab) > 0:
                if not isinstance(vocab[0], str):
                    raise ValueError('Vocab must be list of stringss!')
        else:
            raise NotImplementedError('This type of vocab is not implemented')
        if isinstance(vectors, np.ndarray):
            pass
            #vectors = np.require(vectors, requirements=['A', 'C', 'O', 'W'])
            #print(id(vectors))
        elif isinstance(vectors, list):
            vectors = np.array(vectors)
            vectors = np.require(vectors, requirements=['A', 'C', 'O', 'W'])
        else:
            raise NotImplementedError('This type of vectors is not implemented')
        
        self.this = gldac.new(k, iters, vocab, vectors)
        self.word_vec = vectors
        self.ext = None
        self.word_vec = None

    def __del__(self):
        if self.this is not None:
            gldac.delete(self.this, self.ext)
        
    def set_training_data(self, *, training_inputs: Inputs, validation_inputs: Inputs = None) -> None:
        """
        Sets training data for GLDA.

        Parameters
        ----------
        training_inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id.
        validation_inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id. This represents validation docs to validate the results learned after each iteration of canopy algorithm.
        """

        self.training_inputs = training_inputs
        self.validation_inputs = validation_inputs
        self.fitted = False

    
    def fit(self) -> None:
        """
        Inference on the Gaussian latent Dirichley allocation model
        """
        if self.fitted:
            return

        if self.training_inputs is None:
            raise ValueError("Missing training data.")

        gldac.fit(self.this, self.training_inputs, self.validation_inputs)
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
        return gldac.predict(self.this, inputs)

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
        return gldac.evaluate(self.this, inputs)
 
    def get_top_words(self, *, num_top: int = 15) -> Outputs:
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

        return gldac.top_words(self.this, num_top)

    def get_topic_matrix(self) -> np.ndarray:
        """
        Get current word|topic distribution matrix for this model.

        Returns
        ----------
        topic_matrix : numpy.ndarray
            A numpy array of shape (vocab_size,k) with each column containing the word|topic distribution.
        """

        if self.ext is None:
            self.ext = gldac.topic_matrix(self.this)
        return self.ext
    
    def get_params(self) -> Params:
        """
        Get parameters of GLDA.

        Parameters are basically the topic matrix in byte stream.

        Returns
        ----------
        params : Params
            A named tuple of parameters.
        """

        return Params(topic_matrix=gldac.serialize(self.this))

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters of LDA.

        Parameters are basically the topic matrix in byte stream.

        Parameters
        ----------
        params : Params
            A named tuple of parameters.
        """
        self.this = gldac.deserialize(params.topic_matrix)

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
