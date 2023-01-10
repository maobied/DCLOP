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
ADCSdata=pd.read_csv("skcubeADCS20170623to20180304_D-N_labels.csv")

ADCSdata1 = ADCSdata.copy()  

## delete DN column from the dataset
del ADCSdata1['DN']

## set Datetime column to be the index
ADCSdata1.set_index('DateTime', inplace=True)
###   for ADCS only
del ADCSdata1['Subsystem']
del ADCSdata1['B-dot gain']
del ADCSdata1['SUN Z- (x)']
del ADCSdata1['SUN Z- (y)']
del ADCSdata1['SUN Z- IRRAD']
del ADCSdata1['Gyro TEMP']
del ADCSdata1['ADCS Mode']

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

