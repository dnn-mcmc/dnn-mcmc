#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from stochastic_mlp import StochasticMLP
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

ndata = 2000
numPerRank = int(np.floor((ndata * 0.8) / size))

x_train = None
y_train = None
x_test = None
y_test = None

# Make data
if rank == 0:
    np.random.seed(1234)
    X, Y = make_moons(ndata, noise = 0.1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=73)
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

x_trn = np.empty((numPerRank, 2), dtype = "float64")
y_trn = np.empty((numPerRank, 1), dtype = "int")
comm.Scatter(x_train, x_trn, root = 0)
comm.Scatter(y_train, y_trn, root = 0)

train_ds = tf.data.Dataset.from_tensor_slices((x_trn, y_trn)).batch(32)

# initialize model and network
tf.random.set_seed(1234)
model = StochasticMLP(hidden_layer_sizes = [50], n_outputs=1)
network = [model.call(x) for x, y in train_ds]    
kernels = [model.generate_hmc_kernel(x, y) for x, y in train_ds]

# Burn-in
burnin = 500

for i in range(burnin):
    
    if rank == 0 and i % 100 == 0: print("Step %d" % i)
        
    network_new = []
    kernels_new = []
    
    res = [model.propose_new_state_hamiltonian(x, net, y, ker) \
               for (x, y), net, ker in zip(train_ds, network, kernels)]
    
    network_new, kernels_new = zip(*res)
         
    network = network_new
    kernels = kernels_new
    
# Training
epochs = 500
start_time = time.time()

for epoch in range(epochs):
    
    loss = 0.0
    acc = 0.0
    for bs, (x, y) in enumerate(train_ds):
        
        # weight updating
        # main processor receives new models from subprocessors and calculate the average
        
        weights = model.get_weights()
        
        weights_new = []
        for i in range(len(weights)):
            weight = np.array(weights[i], dtype = 'double')
            weight_sum = np.zeros(weight.shape, dtype = 'double')
            comm.Reduce(weight, weight_sum, op = MPI.SUM, root = 0)
            weights_new.append(weight_sum / size)
            
        # broadcast weights to subprocessors and update their models
        weights = comm.bcast(weights_new, root = 0)
        model.set_weights(weights)
        
        # update the model one step 
        model.update_weights(x, network[bs], y, 0.1)           
        
        # sampling one step
        res = [model.propose_new_state_hamiltonian(x, net, y, ker, is_update_kernel = False) \
                   for (x, y), net, ker in zip(train_ds, network, kernels)]
        network = res
        loss += -1 * tf.reduce_mean(model.target_log_prob(x, network[bs], y))
    
    # calculate total loss
    loss = np.array(loss, dtype = 'double')
    loss_sum = np.array(0.0, dtype = 'double')
    comm.Reduce(loss, loss_sum, op = MPI.SUM, root = 0)

    # calculate total training accuracy
    preds = [model.get_predictions(images) for images, labels in train_ds]
    acc = accuracy_score(np.concatenate(preds), y_trn)
    acc = np.array(acc, dtype = 'double')
    train_acc = np.array(0.0, dtype = 'double')
    comm.Reduce(acc, train_acc, op = MPI.SUM, root = 0)
    train_acc /= size

    # output training process
    if rank == 0:
        print("Epoch %d/%d: - %.4fs/step - loss: %.4f - accuracy: %.4f" 
            % (epoch + 1, epochs, (time.time() - start_time) / (epoch + 1), loss_sum, train_acc))
