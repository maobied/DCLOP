# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:38:58 2022
"""
    
import itertools
from math import erf, sqrt

import numpy as np
import sys
import warnings
#-----------------------------
from fastdist import fastdist 
#from scipy.spatial.distance import cdist, pdist
#from operator import itemgetter # for sorting zip
#import operator # for sorting zip
from sklearn.utils.validation import check_array
#from sklearn.metrics.pairwise import euclidean_distances
#from scipy.spatial import distance#
#from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
##-----------------------------------------------------------------------------

class LocalOutlierProbability(object):
    """
    :param data: a Pandas DataFrame or Numpy array of float data
    :param extent: an integer value [1, 2, 3] that controls the statistical 
    extent, e.g. lambda times the standard deviation from the mean (optional, 
    default 3)
    :param n_neighbors: the total number of neighbors to consider w.r.t. each 
    sample (optional, default 10)
    :param cluster_labels: a numpy array of cluster assignments w.r.t. each 
    sample (optional, default None)
    :return:
    """"""
    Based on the work of Kriegel, Kröger, Schubert, and Zimek (2009) in LoOP: 
    Local Outlier Probabilities.
    ----------
    References
    ----------
    .. [1] Breunig M., Kriegel H.-P., Ng R., Sander, J. LOF: Identifying 
           Density-based Local Outliers. ACM SIGMOD
           International Conference on Management of Data (2000).
    .. [2] Kriegel H.-P., Kröger P., Schubert E., Zimek A. LoOP: Local Outlier 
           Probabilities. 18th ACM conference on 
           Information and knowledge management, CIKM (2009).
    .. [3] Goldstein M., Uchida S. A Comparative Evaluation of Unsupervised 
           Anomaly Detection Algorithms for Multivariate Data. PLoS ONE 11(4):
           e0152173 (2016).
    .. [4] Hamlet C., Straub J., Russell M., Kerlin S. An incremental and 
           approximate local outlier probability algorithm for intrusion 
           detection and its evaluation. Journal of Cyber Security Technology 
           (2016). 
    """

    class Validate:

        """
        The Validate class aids in ensuring PyNomaly receives the right set
        of user inputs for proper execution of the Local Outlier Probability
        (LoOP) approach. Depending on the desired behavior, either an
        exception is raised to the user or PyNomaly continues executing
        albeit with some form of user warning.
        """

        """
        Private methods.
        """

        @staticmethod
        def _data(obj):
            """
            Validates the input data to ensure it is either a Pandas DataFrame
            or Numpy array.
            :param obj: user-provided input data.
            :return: a vector of values to be used in calculating the local
            outlier probability.
            """
            if obj.__class__.__name__ == 'DataFrame':
                points_vector = obj.values
                return points_vector
            elif obj.__class__.__name__ == 'ndarray':
                points_vector = obj
                return points_vector
            else:
                warnings.warn(
                    "Provided data or distance matrix must be in ndarray "
                    "or DataFrame.",
                    UserWarning)
                if isinstance(obj, list):
                    points_vector = np.array(obj)
                    return points_vector
                points_vector = np.array([obj])
                return points_vector

        def _inputs(self, obj):
            """
            Validates the inputs provided during initialization to ensure
            that the needed objects are provided.
            :param obj: a PyNomaly object.
            :return: a boolean indicating whether validation has failed or
            the data, distance matrix, and neighbor matrix.
            """
            if all(v is None for v in [obj.data, obj.distance_matrix]):
                warnings.warn(
                    "Data or a distance matrix must be provided.", UserWarning
                )
                return False
            elif all(v is not None for v in [obj.data, obj.distance_matrix]):
                warnings.warn(
                    "Only one of the following may be provided: data or a "
                    "distance matrix (not both).", UserWarning
                )
                return False
            if obj.data is not None: 
                points_vector = self._data(obj.data)
                return points_vector, obj.distance_matrix, obj.neighbor_matrix
            if all(matrix is not None for matrix in [obj.neighbor_matrix,
                                                     obj.distance_matrix]):
                dist_vector = self._data(obj.distance_matrix)
                neigh_vector = self._data(obj.neighbor_matrix)
            else:
                warnings.warn(
                    "A neighbor index matrix and distance matrix must both be "
                    "provided when not using raw input data.", UserWarning
                )
                return False
            if obj.distance_matrix.shape != obj.neighbor_matrix.shape:
                warnings.warn(
                    "The shape of the distance and neighbor "
                    "index matrices must match.", UserWarning
                ) 
                return False
            elif (obj.distance_matrix.shape[1] != obj.n_neighbors) \
                    or (obj.neighbor_matrix.shape[1] !=
                        obj.n_neighbors):
                warnings.warn("The shape of the distance or "
                              "neighbor index matrix does not "
                              "match the number of neighbors "
                              "specified.", UserWarning)
                return False
            return obj.data, dist_vector, neigh_vector

        @staticmethod
        def _cluster_size(obj):
            """
            Validates the cluster labels to ensure that the smallest cluster
            size (number of observations in the cluster) is larger than the
            specified number of neighbors.
            :param obj: a PyNomaly object.
            :return: a boolean indicating whether validation has passed.
            """ 
            c_labels = obj._cluster_labels()
            for cluster_id in set(c_labels):
                c_size = np.where(c_labels == cluster_id)[0].shape[0]
                if c_size > obj.n_neighbors:

                    return True

            warnings.warn(
                "Number of neighbors specified larger than smallest "
                "cluster. Specify a number of neighbors smaller than "
                "the smallest cluster size (observations in smallest "
                "cluster minus one).",
                UserWarning) #
            return False # True
        ##--------------------------------------------------------------------
            
        @staticmethod
        def _n_neighbors(obj):
            """
            Validates the specified number of neighbors to ensure that it is
            greater than 0 and that the specified value is less than the total
            number of observations.
            :param obj: a PyNomaly object.
            :return: a boolean indicating whether validation has passed.
            """
            if not obj.n_neighbors > 0:
                obj.n_neighbors = 10
                warnings.warn("n_neighbors must be greater than 0."
                              " Fit with " + str(obj.n_neighbors) +
                              " instead.",
                              UserWarning)
                return False
            elif obj.n_neighbors >= obj._n_observations():
                obj.n_neighbors = obj._n_observations() - 1
                warnings.warn(
                    "n_neighbors must be less than the number of observations."
                    " Fit with " + str(obj.n_neighbors) + " instead.",
                    UserWarning)
            return True

        @staticmethod
        def _extent(obj):
            """
            Validates the specified extent parameter to ensure it is either 1,
            2, or 3.
            :param obj: a PyNomaly object.
            :return: a boolean indicating whether validation has passed.
            
            #  ? = 1 ? ? ˜ 68%, ? = 2 ? ? ˜ 95%, and ? = 3 ? ? ˜ 99.7%
            """
            if obj.extent not in [1, 2, 3]:
                warnings.warn(
                    "extent parameter (lambda) must be 1, 2, or 3.",
                    UserWarning)
                return False
            return True

        @staticmethod
        def _missing_values(obj):
            """
            !!!                      nothing here                           !!!
            Validates the provided data to ensure that it contains no
            missing values.
            :param obj: a PyNomaly object.
            :return: a boolean indicating whether validation has passed.
            """
            if np.any(np.isnan(obj.data)):
                warnings.warn(
                    "Method does not support missing values in input data.",
                    UserWarning)
                return False
            return True

        @staticmethod
        def _fit(obj):
            """
            Validates that the model was fit prior to calling the stream()
            method.
            :param obj: a PyNomaly object.
            :return: a boolean indicating whether validation has passed.
            """
            if obj.is_fit is False: 
                warnings.warn(
                    "Must fit on historical data by calling fit() prior to "
                    "calling stream(x).",
                    UserWarning)
                return False
            return True

        @staticmethod
        def _no_cluster_labels(obj):
            """
            Checks to see if cluster labels are attempting to be used in
            stream() and, if so, calls fit() once again but without cluster
            labels. As PyNomaly does not accept clustering algorithms as input,
            the stream approach does not support clustering.
            :param obj: a PyNomaly object.
            :return: a boolean indicating whether validation has passed.
            """
            if len(set(obj._cluster_labels())) > 1:
                warnings.warn(
                    "Stream approach does not support clustered data. "
                    "Automatically refit using single cluster of points.",
                    UserWarning)
                return False
            return True

    """
    Decorators.
    """

    def accepts(*types):
        """
        A decorator that facilitates a form of type checking for the inputs
        which can be used in Python 3.4-3.7 in lieu of Python 3.5+'s type
        hints.
        :param types: the input types of the objects being passed as arguments
        in __init__.
        :return: a decorator.
        """

        def decorator(f):
            assert len(types) == f.__code__.co_argcount

            def new_f(*args, **kwds):
                for (a, t) in zip(args, types):
                    if type(a).__name__ == 'DataFrame':
                        a = np.array(a)
                    if isinstance(a, t) is False:
                        warnings.warn("Argument %r is not of type %s" % (a, t),
                                      UserWarning)
                opt_types = {
                    'distance_matrix': {
                        'type': types[2]
                    },
                    'neighbor_matrix': {
                        'type': types[3]
                    },
                    'extent': {
                        'type': types[4]
                    },
                    'n_neighbors': {
                        'type': types[5]
                    },
                    'cluster_labels': {
                        'type': types[6]
                    #---------------------------------------
                    },  #
                    'cluster_centers': { #
                        'type': types[7] #
                    },  #
                    'cluster_labels_temp0': { #
                        'type': types[8] #
                    },#
                    'distance_matrix_temp': {#
                        'type': types[9]#
                    },#
                    'neighbor_matrix_temp': {#
                        'type': types[10]#
                    },#  
                    'n_neighbors_temp0': {#
                        'type': types[11]#
                    },#  
                    'large_cluster_labels': {#
                        'type': types[12]#
                    }
                }
                for x in kwds:
                    opt_types[x]['value'] = kwds[x]
                for k in opt_types:
                    try:
                        if isinstance(opt_types[k]['value'],
                                      opt_types[k]['type']) is False:
                            warnings.warn("Argument %r is not of type %s." % (
                                k, opt_types[k]['type']), UserWarning)
                    except KeyError:
                        pass
                return f(*args, **kwds)

            new_f.__name__ = f.__name__
            return new_f

        return decorator

    @accepts(object, np.ndarray, np.ndarray, np.ndarray, (int, np.integer),
             (int, np.integer), list, np.ndarray,list, np.ndarray, np.ndarray, (int, np.integer), (float, np.float), (int, np.integer))
    def __init__(self, data=None, distance_matrix=None, neighbor_matrix=None, extent=3, n_neighbors=10, 
                 cluster_labels= None, cluster_centers = None, cluster_labels_temp0= None, distance_matrix_temp = None, neighbor_matrix_temp = None, n_neighbors_temp0 = 10, alpha=0.95, beta=5):
        
        self.data = data
        self.distance_matrix = distance_matrix
        self.neighbor_matrix = neighbor_matrix
        self.extent = extent
        self.n_neighbors = n_neighbors
        self.cluster_labels = cluster_labels
        self.points_vector = None
        self.points_vector_temp0 = None#
        self.prob_distances = None
        self.prob_distances_ev = None
        self.norm_prob_local_outlier_factor = None
        self.local_outlier_probabilities = None
        self._objects = {}
        self.is_fit = False
        #---------------------------------
        self.cluster_labels_temp0 = cluster_labels_temp0#
        self.cluster_centers = cluster_centers #
        self.distance_matrix_temp = distance_matrix_temp#
        self.neighbor_matrix_temp = neighbor_matrix_temp#
        self.n_neighbors_temp0 = n_neighbors_temp0#
        self.small_cluster_labels =None#
        self.large_cluster_labels =None#
        self.points_vector_temp_small = None#
        self.alpha = alpha #
        self.beta = beta # 
        #---------------------------------
        self.Validate()._inputs(self)
        self.Validate._extent(self)
       
    """
    Private methods.
    """

    @staticmethod
    def _standard_distance(cardinality: float, sum_squared_distance: float) \
            -> float:
        """ 
        Calculates the standard distance of an observation.
        :param cardinality: the cardinality of the input observation.
        :param sum_squared_distance: the sum squared distance between all
        neighbors of the input observation.
        :return: the standard distance.
        # """
        division_result = sum_squared_distance / cardinality
        st_dist = sqrt(division_result)
        return st_dist 

    @staticmethod 
    def _prob_distance(extent: int, standard_distance: float) -> float:
        """ 
        Calculates the probabilistic distance of an observation.
        :param extent: the extent value specified during initialization.
        :param standard_distance: the standard distance of the input
        observation.
        :return: the probabilistic distance.
        """
        return extent * standard_distance

    @staticmethod
    def _prob_outlier_factor(probabilistic_distance: np.ndarray, ev_prob_dist:
        np.ndarray) -> np.ndarray:
        """
        !! PLOF:= ratio of pdist(o) and the mean of pdist(o) ~ PLOF:= [pdist(o)/mean(pdist(o))] - 1 !!
        
        Calculates the probabilistic outlier factor of an observation. (i.e. one observation)
        :param probabilistic_distance: the probabilistic distance of the
        input observation.
        :param ev_prob_dist:
        :return: the probabilistic outlier factor.
        """
        if np.all(probabilistic_distance == ev_prob_dist):
            return np.zeros(probabilistic_distance.shape)
        else:
            ev_prob_dist[ev_prob_dist == 0.] = 1.e-8
            result = np.divide(probabilistic_distance, ev_prob_dist) - 1.
            return result

    @staticmethod
    def _norm_prob_outlier_factor(extent: float,
                ev_probabilistic_outlier_factor: list) -> list:
        """
        Calculates the normalized probabilistic outlier factor of an
        observation.
        :param extent: the extent value specified during initialization.
        :param ev_probabilistic_outlier_factor: the expected probabilistic
        outlier factor of the input observation.
        :return: the normalized probabilistic outlier factor.
        """
        npofs = []
        for i in ev_probabilistic_outlier_factor:
            npofs.append(extent * sqrt(i))
        return npofs

    @staticmethod
    def _local_outlier_probability(plof_val: np.ndarray, nplof_val: np.ndarray) \
            -> np.ndarray:
        """
        Calculates the local outlier probability of an observation.
        :param plof_val: the probabilistic outlier factor of the input
        observation.
        :param nplof_val: the normalized probabilistic outlier factor of the
        input observation.
        :return: the local outlier probability.
        """
        
        erf_vec = np.vectorize(erf)
        if np.all(plof_val == nplof_val):
            return np.zeros(plof_val.shape)
        else:
            return np.maximum(0, erf_vec(plof_val / (nplof_val * np.sqrt(2.))))
    #-------------------------------------------------------------------------    
    def _n_observations(self) -> int:
        """
        Calculates the number of observations in the data.
        :return: the number of observations in the input data.
        """
        if self.data is not None:
            return len(self.data)
        return len(self.distance_matrix)
               
    def _store(self) -> np.ndarray:
        """
        Initializes the storage matrix that includes the input value,
        cluster labels, local outlier probability, etc. for the input data.
        :return: an empty numpy array of shape [n_observations, 3].
        """ 
        return np.empty([self._n_observations(), 3], dtype=object)

    def _cluster_labels(self) -> np.ndarray:
        """
        Returns a numpy array of cluster labels that corresponds to the
        input labels or that is an array of all 0 values to indicate all
        points belong to the same cluster.
        :return: a numpy array of cluster labels.
        """
        if self.cluster_labels is None:
            if self.data is not None:
                return np.array([0] * len(self.data))
            return np.array([0] * len(self.distance_matrix))
        return np.array(self.cluster_labels)
    #------------------------------------------------------------------------------
    #####################  Take a temp copy of cluster labels ###############
    #------------------------------------------------------------------------------
    def _cluster_labels_temp0(self) -> np.ndarray:
        """
        Returns a numpy array of TEMP cluster labels that corresponds to the
        [input] EDITED labels AFTER ASSIGNING ENIDECES IN SMALL CLUSTERS TO THEIR 
        CLOSEST (NEAREST) LARGE CLUSTERS or that is an array of all 0 values to indicate all
        points belong to the same cluster.
        :return: a numpy array of cluster labels.
        """
        if self.cluster_labels_temp0 is None:
            if self.data is not None:
                return np.array([0] * len(self.data))
            return np.array([0] * len(self.distance_matrix))
        return np.array(self.cluster_labels_temp0)

#------------------------------------------------------------------------------

    @staticmethod
    def _euclidean(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        """
        Calculates the euclidean distance between two observations in the
        input data.
        :param vector1: a numpy array corresponding to observation 1.
        :param vector2: a numpy array corresponding to observation 2.
        :return: the euclidean distance between the two observations.
        """
        diff = vector1 - vector2
        return np.dot(diff, diff) ** 0.5 

    def _assign_distances(self, data_store: np.ndarray) -> np.ndarray:
        """ 
        Takes a distance matrix, produced by _distances or provided through
        user input, and assigns distances for each observation to the storage
        matrix, data_store.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        for vec, cluster_id in zip(range(self.distance_matrix_temp.shape[0]), 
                                   self._cluster_labels_temp0()):  
            data_store[vec][0] = cluster_id
            data_store[vec][1] = self.distance_matrix[vec]
            data_store[vec][2] = self.neighbor_matrix[vec]
        return data_store
     
    def _distances(self) -> None:  
        """
        Provides the distances between each observation and it's closest
        neighbors. When input data is provided, calculates the <<< euclidean >>>
        distance between every observation. Otherwise, the user-provided
        distance matrix is used.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        #------------------------------------------------------------------------------
        #------------------------------ EDITING ---------------------------------------
        #------------------------------------------------------------------------------
        distances_temp0 = np.full([self._n_observations(), self.n_neighbors_temp0], 9e10,
                            dtype=float)
        distances1_temp0 = distances_temp0
        indexes_temp0 = np.full([self._n_observations(), self.n_neighbors_temp0], 9e10,
                          dtype=float)

        self.points_vector_temp0 = self.Validate._data(self.data)
        ##------------------------------------------------------------------------------
        ## getting the closest large cluster for every data point in every small cluster
        ##------------------------------------------------------------------------------
        small_indices0_all = np.where(np.isin(self._cluster_labels(), self.small_cluster_labels))[0]
        small_indices0_all1 = set(small_indices0_all)
        #------------------------------------------------------------------------------
        centroid = np.asanyarray(self.cluster_centers) 
        centroid_shape = check_array(centroid)
        cent_samples, cent_features = centroid_shape.shape
        #------------------------------------------------------------------------------
        ## assigning each index in small clusters to its closest large cluster
        #------------------------------------------------------------------------------
        for j in small_indices0_all1:
            dist_to_closest_large_centroid0 = [9e15]
            index_of_closest_cluster0 = 9e15
            for k, cent_features in enumerate(centroid):
                if k in set(self.large_cluster_labels):
                    dist_to_closest_large_centroid_temp0 = [fastdist.euclidean(self.points_vector_temp0[j], centroid[k,:])]
                    if dist_to_closest_large_centroid_temp0 < dist_to_closest_large_centroid0:
                        dist_to_closest_large_centroid0 = dist_to_closest_large_centroid_temp0
                        index_of_closest_cluster0 = k
      
            self.cluster_labels_temp0[j] = index_of_closest_cluster0
            self._cluster_labels_temp0()[j] =  index_of_closest_cluster0

        ##print('assigned to closest large clusters has been finished')

        #------------------------------------------------------------------------------
                            ########## finding the k-nearest neighbors for every data point belongs to ###############
                            ########## small cluster from its assigned closest large cluster   ###############
        #------------------------------------------------------------------------------

        updated_cluster_labels_temp0 = set(np.unique(self._cluster_labels_temp0()))
        for cluster_id_temp0 in updated_cluster_labels_temp0:
            indices_temp0 = np.where(self._cluster_labels_temp0() == cluster_id_temp0) 
            clust_points_vector_temp0 = self.points_vector_temp0.take(indices_temp0, axis=0)[0]

            pairs_temp0 = itertools.combinations(
                np.ndindex(clust_points_vector_temp0.shape[0]), 2)

            pairs_temp0 = set(pairs_temp0)
            for p_temp0 in pairs_temp0: #calcualte the <<< euclidean >>> distances between every pair of points in each cluster!

                if indices_temp0[0][p_temp0[0]] in small_indices0_all1 and indices_temp0[0][p_temp0[1]] in small_indices0_all1:
                    continue
                d_temp0 = fastdist.euclidean(clust_points_vector_temp0[p_temp0[0]], clust_points_vector_temp0[p_temp0[1]])
                idx_temp0 = indices_temp0[0][p_temp0[0]]
                idx_max_temp0= distances_temp0[idx_temp0].argmax()

                if d_temp0 < distances_temp0[idx_temp0][idx_max_temp0]: 
                    if indices_temp0[0][p_temp0[1]] in small_indices0_all1:
                        pass 

                    if indices_temp0[0][p_temp0[1]] not in small_indices0_all1: 
    
                        distances_temp0[idx_temp0][idx_max_temp0] = d_temp0
                        indexes_temp0[idx_temp0][idx_max_temp0] = indices_temp0[0][p_temp0[1]] 
                
                idx_temp0 = indices_temp0[0][p_temp0[1]]
                idx_max_temp0 = distances_temp0[idx_temp0].argmax()   
                if d_temp0 < distances_temp0[idx_temp0][idx_max_temp0]: 
                    if indices_temp0[0][p_temp0[0]] in small_indices0_all1: 
                        continue

                    distances_temp0[idx_temp0][idx_max_temp0] = d_temp0
                    indexes_temp0[idx_temp0][idx_max_temp0] = indices_temp0[0][p_temp0[0]] 
        #------------------------------------------------------------------------------
        for j_temp0 in range(0, len(self.data)):
            
            zipped1_temp0 = zip(distances_temp0[j_temp0], indexes_temp0[j_temp0])
            zipped2_temp0 = zip(distances1_temp0[j_temp0], indexes_temp0[j_temp0])
            zipped3_temp0 = tuple(zipped1_temp0)
            zipped4_temp0 = tuple(sorted(zipped2_temp0)) 
            distances2_temp0, indexes2_temp0 = zip(*zipped4_temp0)
            distances1_temp0[j_temp0] = distances2_temp0

            indexes_temp0[j_temp0] = indexes2_temp0
        self.distance_matrix_temp = distances1_temp0 
        self.neighbor_matrix_temp = indexes_temp0 
        self.distance_matrix = distances1_temp0
        self.neighbor_matrix = indexes_temp0
    #------------------------------------------------------------------------------ 
    #------------------------------------------------------------------------------

    def _ssd(self, data_store: np.ndarray) -> np.ndarray:
        """
        Calculates the sum squared distance between neighbors for each
        observation in the input data.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        self.cluster_labels_u = np.unique(data_store[:, 0]) 
        ssd_array = np.empty([self._n_observations(), 1]) 
        for cluster_id in self.cluster_labels_u:
            indices = np.where(data_store[:, 0] == cluster_id)
            cluster_distances = np.take(data_store[:, 1], indices).tolist() 
            ssd = np.power(cluster_distances[0], 2).sum(axis=1) 
            for i, j in zip(indices[0], ssd):
                ssd_array[i] = j 

        data_store = np.hstack((data_store, ssd_array)) 

        return data_store
    #-----------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------

    def _standard_distances(self, data_store: np.ndarray) -> np.ndarray:
        """
        Calculated the standard distance for each observation in the input
        data. First calculates the cardinality and then calculates the standard
        distance with respect to each observation.
        :param data_store:
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        cardinality = [self.n_neighbors] * self._n_observations() 
        vals = data_store[:, 3].tolist() 
        std_distances = []
        for c, v in zip(cardinality, vals): 
            std_distances.append(self._standard_distance(c, v))

        return np.hstack((data_store, np.array([std_distances]).T))

    def _prob_distances(self, data_store: np.ndarray) -> np.ndarray:
        """
        Calculates the probabilistic distance for each observation in the
        input data.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        prob_distances = []
        for i in range(data_store[:, 4].shape[0]):
            prob_distances.append(
                self._prob_distance(self.extent, data_store[:, 4][i]))
        return np.hstack((data_store, np.array([prob_distances]).T)) 

    def _prob_distances_ev(self, data_store: np.ndarray) -> np.ndarray:
        """     
        Calculates the expected value of the probabilistic distance for
        each observation in the input data with respect to the cluster the
        observation belongs to.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        prob_set_distance_ev = np.empty([self._n_observations(), 1])

        for cluster_id in self.cluster_labels_u:
            indices = np.where(data_store[:, 0] == cluster_id)[0]

            for index in indices:
                nbrhood = data_store[index][2].astype(int)
                nbrhood_prob_distances = np.take(data_store[:, 5],
                                                 nbrhood).astype(float)   
                nbrhood_prob_distances_nonan = nbrhood_prob_distances[
                    np.logical_not(np.isnan(nbrhood_prob_distances))] 

                prob_set_distance_ev[index] = \
                    nbrhood_prob_distances_nonan.mean()
        self.prob_distances_ev = prob_set_distance_ev
        data_store = np.hstack((data_store, prob_set_distance_ev)) 
        return data_store

    def _prob_local_outlier_factors(self,
                                    data_store: np.ndarray) -> np.ndarray:
        """
        Calculates the probabilistic local outlier factor for each
        observation in the input data. (i.e. for all observations)
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        return np.hstack(
            (data_store,
             np.array([np.apply_along_axis(self._prob_outlier_factor, 0,
                                           data_store[:, 5],
                                           data_store[:, 6])]).T))
        
    def _prob_local_outlier_factors_ev(self,
                                       data_store: np.ndarray) -> np.ndarray:
        """
     
        Calculates the expected value (mean value) of the probabilistic local outlier factor
        for each observation in the input data with respect to the cluster the
        observation belongs to.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
  
        Important Editing !!!!!!!
        Calculates the expected value (mean value) of the probabilistic local outlier factor
        for each observation in the input data with respect to the Larg cluster (only) the
        observation belongs to, while ignoring for SC, and assign the expected value of PLOF of LC 
        for each observation in SC which assigned to this LC.
        """
        prob_local_outlier_factor_ev_dict = {}
        for cluster_id in self.cluster_labels_u: 

            indices = np.where(self.cluster_labels == cluster_id)
      
            prob_local_outlier_factors = np.take(data_store[:, 7],
                                                  indices).astype(float)

            prob_local_outlier_factors_nonan = prob_local_outlier_factors[
                np.logical_not(np.isnan(prob_local_outlier_factors))]
            prob_local_outlier_factor_ev_dict[cluster_id] = (
                    np.power(prob_local_outlier_factors_nonan, 2).sum() /
                    float(prob_local_outlier_factors_nonan.size)
            )
 
        data_store = np.hstack(
            (data_store, np.array([[prob_local_outlier_factor_ev_dict[x] for x
                                    in data_store[:, 0].tolist()]]).T))
             
        return data_store

    def _norm_prob_local_outlier_factors(self, data_store: np.ndarray) \
            -> np.ndarray:
        """
        Calculates the normalized probabilistic local outlier factor for each
        observation in the input data.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        print('data_store[:, 8]', data_store[:, 8])
        return np.hstack((data_store, np.array([self._norm_prob_outlier_factor(
            self.extent, data_store[:, 8].tolist())]).T))

    def _local_outlier_probabilities(self,
                                     data_store: np.ndarray) -> np.ndarray:
        """
        Calculates the local outlier probability for each observation in the
        input data.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        return np.hstack(
            (data_store,
             np.array([np.apply_along_axis(self._local_outlier_probability, 0,
                                           data_store[:, 7],
                                           data_store[:, 9])]).T))

    """
    Public methods
    """

    def fit(self) -> 'LocalOutlierProbability':

        """
        Calculates the local outlier probability for each observation in the
        input data according to the input parameters extent, n_neighbors, and
        cluster_labels.
        :return: self, which contains the local outlier probabilities as
        self.local_outlier_probabilities.
        """
        
        self.Validate._n_neighbors(self)
        
        if self.Validate._cluster_size(self) is False:
            sys.exit()
        if self.data is not None and self.Validate._missing_values(
                self) is False:
            sys.exit()

#-------------------------------------------------------------------------------------------------
#--------------- getting and ordering clusters' size in descending order! "LC & SC" --------------
#-------------------------------------------------------------------------------------------------
        
        self.cluster_sizes = np.bincount(self.cluster_labels)
        print('self.cluster_sizes', self.cluster_sizes)
        cluster_sizes = np.bincount(self.cluster_labels)
        print('cluster_sizes', cluster_sizes)
        # get the actual number of clusers
        self.n_clusters = cluster_sizes.shape[0] 
        print('self.n_clusters', self.n_clusters)
        ##----------------------------------------------------------------------
        X = check_array(self.data) 
        n_samples, n_features = X.shape

        sorted_cluster_indices0 = np.argsort(self.cluster_sizes * -1) 
        print ("cluster indices in descending order according to sizes ", type(sorted_cluster_indices0), sorted_cluster_indices0)
        sorted_cluster_indices = np.delete(sorted_cluster_indices0, np.where(self.cluster_sizes[sorted_cluster_indices0] == 0))

        alpha_list = []
        beta_list = []
        for i in range(1, self.n_clusters):
            temp_sum = np.sum(self.cluster_sizes[sorted_cluster_indices[:i]]) #!#
            print('self.cluster_sizes[sorted_cluster_indices[:i]', self.cluster_sizes[sorted_cluster_indices[:i]])
            print ("temp_sum", temp_sum)
            if temp_sum >= n_samples * self.alpha:
                print ("n_samples * self.alpha", n_samples * self.alpha)
                alpha_list.append(i)
                print ("alpha_list", alpha_list)
                break ##
            
            if self.cluster_sizes[sorted_cluster_indices[i - 1]] / self.cluster_sizes[
                    sorted_cluster_indices[i]] >= self.beta:
                print('i = ', i)
                print('self.cluster_sizes[sorted_cluster_indices[i - 1]]', self.cluster_sizes[sorted_cluster_indices[i - 1]])
                print('self.cluster_sizes[sorted_cluster_indices[i]', self.cluster_sizes[sorted_cluster_indices[i]])
                beta_list.append(i)
                print ("beta_list", beta_list)
                
        ###find the separation index fulfills both alpha and beta
        intersection = np.intersect1d(alpha_list, beta_list)
        print ("intersection", intersection)
        if len(intersection) > 0:
            self.clustering_threshold = intersection[0]
        elif len(alpha_list) > 0:
            print ("hi from alpha_list")
            self.clustering_threshold = alpha_list[0]
        elif len(beta_list) > 0:
            print ("hi from beta_list")
            self.clustering_threshold = beta_list[0]

        else: 
 
            raise ValueError("Could not form valid cluster separation. Please "
             "change n_cluster or change clustering method")
        # #------------------------------------------------------------------
        
        self.small_cluster_labels = sorted_cluster_indices[self.clustering_threshold:]
        self.large_cluster_labels = sorted_cluster_indices[0:self.clustering_threshold]

        self.large_cluster_centers = self.cluster_centers[self.large_cluster_labels]
        self.large_n_clusters = np.asarray(len(self.large_cluster_labels))
        self.small_n_clusters = np.asarray(len(self.small_cluster_labels))###
        #print('large n_clusters', self.large_n_clusters)
        #print('small n_clusters', self.small_n_clusters)
#---------------------------------------------------------------------------

        store = self._store()
        
        if self.data is not None:
            self._distances()
        store = self._assign_distances(store)
        store = self._ssd(store)
        store = self._standard_distances(store)
        store = self._prob_distances(store)
        self.prob_distances = store[:, 5]
        store = self._prob_distances_ev(store)
        store = self._prob_local_outlier_factors(store)
        store = self._prob_local_outlier_factors_ev(store)
        store = self._norm_prob_local_outlier_factors(store)
        ## compute 'nCPLOF' norm probability local outlier factor for large clustersonly 
        self.norm_prob_local_outlier_factor = store[:, 9]
        store = self._local_outlier_probabilities(store)
        self.local_outlier_probabilities = store[:, 10]
        self.prob_distances 
        
        self.is_fit = True
        
        return self

