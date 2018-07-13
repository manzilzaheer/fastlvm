import covertreec

import numpy as np
import pdb
import typing, sys, os

from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces import base
from d3m import container, utils
import d3m.metadata
from d3m.metadata import hyperparams, base as metadata_base
from d3m.metadata import params


Inputs = container.DataFrame  # type: DataFrame
Outputs = container.DataFrame  # type: DataFrame

class Params(params.Params):
    tree: bytes # Byte stream represening the tree.

class HyperParams(hyperparams.Hyperparams):
    trunc = hyperparams.UniformInt(lower=-1, upper=100,default=-1,semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],description='Level of truncation of the tree. -1 means no truncation.')
    k = hyperparams.UniformInt(lower=1, upper=10,default=3,semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],description='Number of neighbors.')

class CoverTree(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, HyperParams]):

    metadata = metadata_base.PrimitiveMetadata({
         "id": "a5f7beda-1144-4185-8cbe-f1de36cedf56",
         "version": "1.0",
         "name": "Nearest Neighbor Search with Cover Trees",
         "description": " This class provides functionality for unsupervised nearest neighbor search, which is the foundation of many other learning methods, notably manifold learning and spectral clustering. The goal is to find a number of points from the given database closest in distance to the query point. The distance can be, in general, any metric measure: standard Euclidean distance is the most common choice and the one currently implemented. In future, adding other metrics should not be too difficult. Standard packages, like those in scikit learn use KD-tree or Ball trees, which do not scale very well, especially with respect to dimension. For example, Ball Trees of scikit learn takes O(n2) construction time and a search query can be linear in worst case making it no better than brute force in some cases. In this class, we implement a modified version of the Cover Tree data structure that allow fast retrieval in logarithmic time. The key properties are: O(n log n) construction time, O(log n) retrieval, and polynomial dependence on the expansion constant of the underlying space. In addition, it allows insertion and removal of points in database. The class is pickle-able. This class provides functionality for unsupervised nearest neighbor search, which is the foundation of many other learning methods, notably manifold learning and spectral clustering. The goal is to find a number of points from the given database closest in distance to the query point. The distance can be, in general, any metric measure: standard Euclidean distance is the most common choice and the one currently implemented. In future, adding other metrics should not be too difficult. Standard packages, like those in scikit learn use KD-tree or Ball trees, which do not scale very well, especially with respect to dimension. For example, Ball Trees of scikit learn takes O(n2) construction time and a search query can be linear in worst case making it no better than brute force in some cases. In this class, we implement a modified version of the Cover Tree data structure that allow fast retrieval in logarithmic time. The key properties are: O(n log n) construction time, O(log n) retrieval, and polynomial dependence on the expansion constant of the underlying space. In addition, it allows insertion and removal of points in database. The class is pickle-able. This class provides functionality for unsupervised nearest neighbor search, which is the foundation of many other learning methods, notably manifold learning and spectral clustering. The goal is to find a number of points from the given database closest in distance to the query point. The distance can be, in general, any metric measure: standard Euclidean distance is the most common choice and the one currently implemented. In future, adding other metrics should not be too difficult. Standard packages, like those in scikit learn use KD-tree or Ball trees, which do not scale very well, especially with respect to dimension. For example, Ball Trees of scikit learn takes O(n2) construction time and a search query can be linear in worst case making it no better than brute force in some cases. In this class, we implement a modified version of the Cover Tree data structure that allow fast retrieval in logarithmic time. The key properties are: O(n log n) construction time, O(log n) retrieval, and polynomial dependence on the expansion constant of the underlying space. In addition, it allows insertion and removal of points in database. The class is pickle-able.",
         "python_path": "d3m.primitives.cmu.fastlvm.CoverTree",
         "primitive_family": "SIMILARITY_MODELING",
         "algorithm_types": [ "K_NEAREST_NEIGHBORS" ],
         "keywords": ["cover trees", "fast nearest neighbor search"],
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
        super().__init__(hyperparams = hyperparams)
        #super(CoverTree, self).__init__()
        self._this = None
        self._trunc = hyperparams['trunc']
        self._k = hyperparams['k']
        self._training_inputs = None  # type: Inputs
        self._fitted = False
        self.hyperparams = hyperparams

    def __del__(self):
        if self._this is not None:
             covertreec.delete(self._this)
     
    def set_training_data(self, *, inputs: Inputs) -> None:
        """
        Sets training data for CoverTree.

        Parameters
        ----------
        inputs : Inputs
            A NxD DataFrame of data points for training.
        """

        self._training_inputs = inputs.values
        self._fitted = False
        
    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        """
        Construct the tree
        """
        if self._fitted:
            return

        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        self._this = covertreec.new(self._training_inputs, self._trunc)
        self._fitted = True
 
    def get_call_metadata(self) -> bool:
        """
        Returns metadata about the last ``fit`` call if it succeeded

        Returns
        -------
        Status : bool
            True/false status of fitting.

        """
        return self.fitted

    def produce(self, *, inputs: Inputs) -> base.CallResult[Outputs]:
        """
        Finds the closest points for the given set of test points using the tree constructed.

        Parameters
        ----------
        inputs : Inputs
            A NxD DataFrame of data points.

        Returns
        -------
        Outputs
            The k nearest neighbours of each point.

        """
        if self._this is None:
            raise ValueError('Fit model first')

        k = self._k
        if k == 1:
            results, _ = covertreec.NearestNeighbour(self._this, inputs.values)
        else:
            results, _ = covertreec.kNearestNeighbours(self._this, inputs.values, k)

        output = container.DataFrame(results, generate_metadata=False, source=self)
        # output.metadata = inputs.metadata.clear(source=self, for_value=output, generate_metadata=True)

        return base.CallResult(output)

    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
	    pass

    def get_params(self) -> Params:
        """
        Get parameters of KMeans.
OB
        Parameters are basically the cluster centres in byte stream.

        Returns
        ----------
        params : Params
            A named tuple of parameters.
        """

        return Params(tree=covertreec.serialize(self._this))

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters of cover tree.

        Parameters are basically the tree structure in byte stream.

        Parameters
        ----------
        params : Params
            A named tuple of parameters.
        """
        self._this = covertreec.deserialize(params['tree'])

