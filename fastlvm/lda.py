import ldac
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

class LDA(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params]):
    """
    This class provides functionality for unsupervised inference on latent Dirichlet allocation, which is a probabilistic topic model of corpora of documents which seeks to represent the underlying thematic structure of the document collection. They have emerged as a powerful new technique of finding useful structure in an unstructured collection as it learns distributions over words. The high probability words in each distribution gives us a way of understanding the contents of the corpus at a very high level. In LDA, each document of the corpus is assumed to have a distribution over K topics, where the discrete topic distributions are drawn from a symmetric dirichlet distribution. Standard packages, like those in scikit learn are inefficient in addition to being limited to a single machine. Whereas our underlying C++ implementation can be distributed to run on multiple machines. To enable the distribution through python interface is work in progress. The API is similar to sklearn.decomposition.LatentDirichletAllocation.
    """

    __author__ = 'CMU'
    __metadata__ = {
        "common_name": "Latent Dirichlet Allocation Topic Modelling",
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

    def __init__(self, *, k: int = 10, iters: int = 100, vocab: Union[list, int]) -> None:
        super(LDA, self).__init__()
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
        
        self.this = ldac.new(k, iters, vocab)
        self.training_inputs = None  # type: Inputs
        self.validation_inputes = None # type: Inputs
        self.fitted = False
        self.ext = None

    def __del__(self):
        if self.this is not None:
            ldac.delete(self.this, self.ext)
        
    def set_training_data(self, *, training_inputs: Inputs, validation_inputs: Inputs = None) -> None:
        """
        Sets training data for LDA.

        Parameters
        ----------
        training_inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id.
        validation_inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id. This represents validation docs to validate the results learned after each iteration of canopy algorithm.
        """

        self.training_inputs = training_inputs
        self.validation_inputes = validation_inputs
        self.fitted = False

    
    def fit(self) -> None:
        """
        Inference on the latent Dirichley allocation model
        """
        if self.fitted:
            return

        if self.training_inputs is None:
            raise ValueError("Missing training data.")

        ldac.fit(self.this, self.training_inputs, self.validation_inputes)
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
        return ldac.predict(self.this, inputs)

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
        return ldac.evaluate(self.this, inputs)
 
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

        return ldac.top_words(self.this, num_top)

    def get_topic_matrix(self) -> np.ndarray:
        """
        Get current word|topic distribution matrix for this model.

        Returns
        ----------
        topic_matrix : numpy.ndarray
            A numpy array of shape (vocab_size,k) with each column containing the word|topic distribution.
        """

        if self.ext is None:
            self.ext = ldac.topic_matrix(self.this)
        return self.ext
    
    def get_params(self) -> Params:
        """
        Get parameters of LDA.

        Parameters are basically the topic matrix in byte stream.

        Returns
        ----------
        params : Params
            A named tuple of parameters.
        """

        return Params(topic_matrix=ldac.serialize(self.this))

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters of LDA.

        Parameters are basically the topic matrix in byte stream.

        Parameters
        ----------
        params : Params
            A named tuple of parameters.
        """
        self.this = ldac.deserialize(params.topic_matrix)

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

