# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:39:28 2022

@author: Muhamed
"""
"""
DCLOP implementation: main
"""
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from keras.utils import plot_model
import seaborn as sns
from sklearn.model_selection import train_test_split

from keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from keras import regularizers, layers
from keras.layers import Dense, Input, Lambda, BatchNormalization, Activation
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
#-------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.activations import *
from tensorflow.keras import initializers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU, Lambda
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
#-------------------
from sklearn.cluster import KMeans
from sklearn import metrics
#import metrics
from sklearn.preprocessing import StandardScaler
#==============================================================================
mydateparse = lambda mydates: pd.datetime.strptime(mydates, '%Y-%m-%d %H:%M:%S')
ADCSdata=pd.read_csv("TM_data.csv")

ADCSdata1 = ADCSdata.copy()  

## delete DN column from the dataset
del ADCSdata1['DN']

## set Datetime column to be the index
ADCSdata1.set_index('DateTime', inplace=True)

#---------------------------------------------------------------------
ADCSdata_std = ADCSdata1.copy()
ADCSdata_rbst = ADCSdata1.copy()
## scaling the features
stdsc = StandardScaler(copy=True)
featuretostdscaler = ADCSdata_std.columns
ADCSdata_std.loc[:, featuretostdscaler] = stdsc.fit_transform(ADCSdata_std[featuretostdscaler])
X = np.array(ADCSdata_std)
X.shape
##------------------------------------------------------------------------------ 
## Deep Clustering
## Define deep clustering layer
class ClusteringLayer(Layer):
    
    """
    Clustering layer converts input sample (feature) to soft label.

    """
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs): #alpha=1
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='he_uniform', name='clusters') #edit #shape=(self.n_clusters, input_dim)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
  
    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.        
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        print('inputs', inputs)
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) 
        print('q', q)
        return q
      
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
## ----------------------------------------------------------------------------
save_dir='results/dec' 
## ============================================================================
n_clusters = 7
## Hyperparameters
batch_size = 10
pretrain_epochs = 300
encoded_dimensions = 7 
pretrain_optimizer = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0, decay=0.0)
dims = [X.shape[-1], 360, 60, 360, encoded_dimensions]

save_dir = './results'
## -------------------------------------------------------
#### adapted dynamically weighted loss function
c = 0.1
def smoothL1(y_true, y_pred): 
    hub = tf.keras.losses.Huber(c) 
    xl = hub(y_true, y_pred)
    xw = K.abs(y_true - y_pred)
    xw = K.mean(K.switch(xw <= c, xw/2, xw), axis=-1)
    x  = (xw*0.1) * xl
    return K.sum(x, axis=-1)
### ------------------------------------------------------
class ParametricRelu(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        
        super(ParametricRelu, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.alpha = self.add_weight(
            name = 'alpha', shape=(input_shape[0]),
            initializer = 'zeros',
            trainable = True 
        )
        super(ParametricRelu, self).build(input_shape)
    
    def call(self, x):

        pos = K.relu(x)
        neg = self.alpha * K.relu(-x)
        return pos + neg
## ---------------------------------------------------------------------------
n_stacks = len(dims) - 1
act = ParametricRelu()
input_data = Input(shape=(dims[0],), name='input')
x = input_data
## internal layers in encoder
for i in range(n_stacks-1):
    x = Dense(dims[i + 1], activation=act, name='encoder1_%d' % i)(x) 
## hidden layer 
encoded = Dense(dims[-1], activation=None, name='encoder1_%d' % (n_stacks - 1))(x) 
x = encoded

## internal layers in decoder
for i in range(n_stacks-1, 0, -1):
    x = Dense(dims[i],  activation=act, name='decoder1_%d' % i)(x) 
## output
decoded = Dense(dims[0], name='decoder1_0')(x)
x = decoded
## ---------------------------------------------------------------------------
autoencoder1 = Model(inputs=input_data, outputs=decoded)
autoencoder1.summary()
encoder1 = Model(inputs=input_data, outputs=encoded)
encoder1.summary()
#-------------------------
##checking:: retrieve the last layer of the autoencoder model
encoded_input = Input(shape=(encoded_dimensions,))
decoder1_layer1 = autoencoder1.layers[-4]
decoder1_layer2 = autoencoder1.layers[-3]
decoder1_layer3 = autoencoder1.layers[-2]
decoder1_layer4 = autoencoder1.layers[-1] 

decoder1 = Model(inputs=encoded_input, outputs= decoder1_layer4(decoder1_layer3(decoder1_layer2(decoder1_layer1(encoded_input)))))
decoder1.summary()

ae1_weights = autoencoder1.get_weights()
autoencoder1.compile(optimizer=pretrain_optimizer, loss=smoothL1, loss_weights=1.0, metrics=['accuracy'])#, run_eagerly=True)
enc0 = encoder1.predict(X)
print('Pretraining Starting autoencoder1.....')
AE1 = autoencoder1.fit(X, X, batch_size=batch_size, epochs=pretrain_epochs)
print('Pretraining Finish autoencoder1.....')
enc1 = encoder1.predict(X)
ae1_weights1 = autoencoder1.get_weights()

dec1 = decoder1.predict(enc1)
ae1 = autoencoder1.predict(X)
print('autoencoder1.losses', autoencoder1.losses)
## ---------------------------------------------------------------------------
## plotting dynamicall weighted loss 
plt.plot(AE1.history["loss"], label="Dynamically weighted Loss")
plt.legend()
plt.show()
## ---------------------------------------------------------------------------
## Save the pre-trained autoencoder weights
autoencoder1.save_weights(save_dir + '/ae1_weights.h5')
## Load the pre-trained auto encoder weights
autoencoder1.load_weights(save_dir + '/ae1_weights.h5')
##----------------------------------------------------------------------------
## for checking
enc2 = encoder1.predict(X)
ae1_weights2 = autoencoder1.get_weights()

dec1 = decoder1.predict(enc1)
ae2 = autoencoder1.predict(X)

X_encoded = encoder1.predict(X)
X_decoded = decoder1.predict(enc1)
X_decoded1 = autoencoder1.predict(X)
ae_error1 = np.mean(np.power(X - X_decoded, 2), axis=1)
ae_error2 = K.mean(K.square(X - X_decoded), axis=1)
ADCSdata['ae_error1'] = ae_error1
ADCSdata['ae_error2'] = ae_error2
## ---------------------------------------------------------------------------
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder1.output)
model = Model(inputs=encoder1.input, outputs=[clustering_layer, autoencoder1.output]) 
model.compile(loss=['kld', smoothL1], loss_weights=[0.1, 1.0], optimizer=pretrain_optimizer, metrics=['accuracy'])

plot_model(model, to_file='model.png', show_shapes=True)
from IPython.display import Image
Image(filename='model.png')
model.summary()

model.save_weights(save_dir + '/joint_model_weights.h5')
model.load_weights(save_dir + '/joint_model_weights.h5')
## ----------------------------------------------------------------------------
## ----------------------------------------------------------------------------
X_decoding_befkmeans = autoencoder1.predict(X)
X_embedding_kmeans = encoder1.predict(X)
kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(encoder1.predict(X))
kmeans_cluster_centers = kmeans.cluster_centers_
y_pred_kmeans = kmeans.predict(encoder1.predict(X))

model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
csize_kmeans = np.bincount(y_pred_kmeans)
y_pred_last = np.copy(y_pred_kmeans)
y_pred_kmeans = np.copy(y_pred_kmeans)

## ----------------------------------------------------------------------------
## computing an auxiliary target distribution
def target_distribution(q):
     weight = q ** 2 / q.sum(0)
     return (weight.T / weight.sum(1)).T
### ---------------------------------------------------------------------------
## Deep clustering
## Hyperparameters
loss = 0
index = 0
maxiter = 1200
update_interval = 20
index_array = np.arange(X.shape[0])
print('index_array', index_array)
tol = 0.0001
## ----------------------------------------------------------------------------
model.save_weights(save_dir + '/joint_model_weights.h5')
model.load_weights(save_dir + '/joint_model_weights.h5')
### --------------------------------------------------------------------------
loss = [0, 0, 0]
index = 0

for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q, _  = model.predict(X, verbose=0)
        p = target_distribution(q)  
        y_pred = q.argmax(1)
    idx = index_array[index * batch_size: min((index+1) * batch_size, X.shape[0])]
    loss = model.train_on_batch(x=X[idx], y=[p[idx], X[idx]])
    index = index + 1 if (index + 1) * batch_size <= X.shape[0] else 0

model.save_weights(save_dir + '/joint_model_weights.h5')
model.load_weights(save_dir + '/joint_model_weights.h5')
####--------------------------------------------------------------------------
ae8 = autoencoder1.predict(X)
dec8 = decoder1.predict(enc1)
enc8 = encoder1.predict(X)
ae1_weights8 = autoencoder1.get_weights()
###---------------------------------------------------------------------------
q, _ = model.predict(X, verbose=0)
p = target_distribution(q)  
y_pred = q.argmax(1)
y_predp = p.argmax(1)

model.save_weights(save_dir + '/joint_model_weights.h5')
model.load_weights(save_dir + '/joint_model_weights.h5')
###---------------------------------------------------------------------------
dec_labels = y_pred
clusters_dec_label = np.unique(dec_labels)
n_clusters_dec = len(clusters_dec_label)
clusters_dec_label_list = clusters_dec_label.tolist()
clusters_dec_label_set = set(np.unique(dec_labels))

ADCSdata['dec_cluster_labels'] = dec_labels
ADCSdata['dec_cluster_labels_temp0'] = dec_labels
dec_cluster_labels = dec_labels
dec_cluster_labels_temp0 = dec_labels

X_embedding = encoder1.predict(X) 
## ---------------------------------------------------------------------------
model_weights_clust = model.get_layer(name='clustering').get_weights()

dec_cluster_centers_from_model = model.get_layer(name='clustering').get_weights()[0]
dec_cluster_centers = np.full([n_clusters, X_embedding.shape[1]], 9e10,
                            dtype=float)

for i in clusters_dec_label_list:
    dec_cluster_centers[i, :] = np.mean(
            X_embedding[np.where(dec_labels == i)], axis=0)
#-----------------------------------------------------------------------------
csize_dec = ADCSdata.groupby('dec_cluster_labels')['dec_cluster_labels'].size()
###==============================================================================
#########               Clustering validation
###==============================================================================
###---------------------------------------------------------------------------
###      Silhouette coefficient
###---------------------------------------------------------------------------
size_clusters = np.bincount(dec_labels)

# Sort the order from the largest to the smallest
sorted_cluster_indices = np.argsort(size_clusters * -1)
silhouette_dc = metrics.silhouette_score(X_embedding, dec_labels)
###---------------------------------------------------------------------------
###      Calinski-Harabasz Index 
###---------------------------------------------------------------------------
I_CH_dec = metrics.calinski_harabasz_score(X_embedding, dec_labels)
###---------------------------------------------------------------------------
###      Davies-Bouldin Index
###---------------------------------------------------------------------------
I_DB_dec = metrics.davies_bouldin_score(X_embedding, dec_labels)
## ---------------------------------------------------------------------------
##============================================================================
###### Plot the deep clusters using all possible latent rep. for comparison
##============================================================================
plt.xlabel('Latent representation space 1', labelpad=4, fontsize=11)
plt.ylabel('Latent representation space 2', labelpad=4, fontsize=11)
plt.scatter(X_embedding[:, 0], X_embedding[:, 1], c = dec_labels)
plt.show()
plt.close()

plt.xlabel('Latent representation space 1', labelpad=4, fontsize=11)
plt.ylabel('Latent representation space 2', labelpad=4, fontsize=11)
plt.scatter(X_embedding[:, 0], X_embedding[:, 2], c = dec_labels)
plt.show()
plt.close()

plt.xlabel('Latent representation space 1', labelpad=4, fontsize=11)
plt.ylabel('Latent representation space 2', labelpad=4, fontsize=11)
plt.scatter(X_embedding[:, 0], X_embedding[:, 3], c = dec_labels)
plt.show()
plt.close()

plt.xlabel('Latent representation space 1', labelpad=4, fontsize=11)
plt.ylabel('Latent representation space 2', labelpad=4, fontsize=11)
plt.scatter(X_embedding[:, 0], X_embedding[:, 4], c = dec_labels)
plt.show()
plt.close()

plt.xlabel('Latent representation space 1', labelpad=4, fontsize=11)
plt.ylabel('Latent representation space 2', labelpad=4, fontsize=11)
plt.scatter(X_embedding[:, 1], X_embedding[:, 2], c = dec_labels)
plt.show()
plt.close()

plt.xlabel('Latent representation space 1', labelpad=4, fontsize=11)
plt.ylabel('Latent representation space 2', labelpad=4, fontsize=11)
plt.scatter(X_embedding[:, 1], X_embedding[:, 3], c = dec_labels)
plt.show()
plt.close()

plt.xlabel('Latent representation space 1', labelpad=4, fontsize=11)
plt.ylabel('Latent representation space 2', labelpad=4, fontsize=11)
plt.scatter(X_embedding[:, 1], X_embedding[:, 4], c = dec_labels)
plt.show()
plt.close()

plt.xlabel('Latent representation space 1', labelpad=4, fontsize=11)
plt.ylabel('Latent representation space 2', labelpad=4, fontsize=11)
plt.scatter(X_embedding[:, 1], X_embedding[:, 5], c = dec_labels)
plt.show()
plt.close()

plt.xlabel('Latent representation space 1', labelpad=4, fontsize=11)
plt.ylabel('Latent representation space 2', labelpad=4, fontsize=11)
plt.scatter(X_embedding[:, 2], X_embedding[:, 3], c = dec_labels)
plt.show()
plt.close()

plt.xlabel('Latent representation space 1', labelpad=4, fontsize=11)
plt.ylabel('Latent representation space 2', labelpad=4, fontsize=11)
plt.scatter(X_embedding[:, 2], X_embedding[:, 4], c = dec_labels)
plt.show()
plt.close()

plt.xlabel('Latent representation space 1', labelpad=4, fontsize=11)
plt.ylabel('Latent representation space 2', labelpad=4, fontsize=11)
plt.scatter(X_embedding[:, 2], X_embedding[:, 5], c = dec_labels)
plt.show()
plt.close()

plt.xlabel('Latent representation space 1', labelpad=4, fontsize=11)
plt.ylabel('Latent representation space 2', labelpad=4, fontsize=11)
plt.scatter(X_embedding[:, 3], X_embedding[:, 4], c = dec_labels)
plt.show()
plt.close()

plt.xlabel('Latent representation space 1', labelpad=4, fontsize=11)
plt.ylabel('Latent representation space 2', labelpad=4, fontsize=11)
plt.scatter(X_embedding[:, 3], X_embedding[:, 5], c = dec_labels)
plt.show()
plt.close()

plt.xlabel('Latent representation space 1', labelpad=4, fontsize=11)
plt.ylabel('Latent representation space 2', labelpad=4, fontsize=11)
plt.scatter(X_embedding[:, 4], X_embedding[:, 5], c = dec_labels)
plt.show()
plt.close()

###===========================================================================
##### CONFUSION MATRIX FOR CLUSTERING DAY AND NIGHT 
###===========================================================================
### Note that, for the evaluation purpose and because of the deadline, 
### I already added a column 'DN' to lable day and night observations
### I will work on the code to make the model recognize it from the time.
###===========================================================================
DN = ADCSdata['DN']
### Compute confusion matrix for all clusters
def confusion_matrix(act_labels, pred_labels):
    uniqueLabels = list(set(act_labels))
    clusters = list(set(pred_labels))
    cm = [[0 for i in range(len(clusters))] for i in range(len(uniqueLabels))]
    for i, act_label in enumerate(uniqueLabels):
        for j, pred_label in enumerate(pred_labels):
            if act_labels[j] == act_label:
                cm[i][pred_label] = cm[i][pred_label] + 1
    return cm

cnf_matrix_dc = confusion_matrix(DN, dec_cluster_labels)

print('\n'.join([''.join(['{:4}'.format(item) for item in row])
      for row in cnf_matrix_dc]))

###===========================================================================
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:38:58 2021

@author: Muhammad
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
    
__author__ = 'Valentino Constantinou'
__version__ = '0.3.0'
__license__ = 'Apache License, Version 2.0'

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


###########               Implementing CLOP                      #############
###===========================================================================
dec_clust = LocalOutlierProbability(X_embedding, extent=3, n_neighbors=10, n_neighbors_temp0=10, cluster_labels=list(dec_labels), cluster_labels_temp0=list(dec_labels), cluster_centers = dec_cluster_centers_from_model).fit() 
dec_scores_loop = dec_clust.local_outlier_probabilities
## for checking
ADCSdata['dec_cluster_labels_temp0'] = dec_clust.cluster_labels_temp0 
ADCSdata['LoOP_scores_dec'] = dec_scores_loop

#### ==========================================================================
DN = ADCSdata['DN']
# Compute confusion matrix
def confusion_matrix(act_labels, pred_labels):
    uniqueLabels = list(set(act_labels))
    clusters = list(set(pred_labels))
    cm = [[0 for i in range(len(clusters))] for i in range(len(uniqueLabels))]
    for i, act_label in enumerate(uniqueLabels):
        for j, pred_label in enumerate(pred_labels):
            if act_labels[j] == act_label:
                cm[i][pred_label] = cm[i][pred_label] + 1
    return cm

#### -------------------------------------------------------------------------
## for deep clustering
cnf_matrix_dc = confusion_matrix(DN, dec_cluster_labels)
###=============================================================================

