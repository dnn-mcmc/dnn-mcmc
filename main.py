import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras import initializers
from sklearn.metrics import accuracy_score
import time

from MLP_model import StochasticMLP

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Select binary data
label_sub = [0,1]
x_train_sub = [x for x, y in zip(x_train, y_train) if y in label_sub]
y_train_sub = [y for y in y_train if y in label_sub]
x_test_sub = [x for x, y in zip(x_test, y_test) if y in label_sub]
y_test_sub = [y for y in y_test if y in label_sub]


print('There are', len(x_train_sub), 'training images.')
print('There are', len(x_test_sub), 'test images.')

train_ds = tf.data.Dataset.from_tensor_slices((x_train_sub, y_train_sub)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test_sub, y_test_sub)).batch(32)

# generate chains 
model = StochasticMLP(hidden_layer_sizes = [100, 50], n_outputs = 2)
network = [model.call(images) for images, labels in train_ds]

# test HMC
hmc = [model.run_chain(images, net, labels) for (images, labels), net in zip(train_ds, network)]
print(hmc)

'''
# chains initialization
sampling = 1
for i in range(sampling):
    
    print("In %d sampling step" % i)

    #start_time = time.time()
    h_proposed = [model.propose_new_state(images, net, labels) for (images, labels), net in zip(train_ds, network)]
    #print("--- propose new state %s seconds ---" % (time.time() - start_time))
    
    #start_time = time.time()
    network_new = [model.accept_reject(images, net, net_proposed, labels) 
                        for (images, labels), net, net_proposed in zip(train_ds, network, h_proposed)]
    network = network_new
    #print("--- step into new state %s seconds ---" % (time.time() - start_time))

# model training
epochs = 10
for epoch in range(epochs):
    
    print("Epoch %d" % epoch)
    for bs, (images, labels) in enumerate(train_ds):
    
        # weight-updating step
        print("Updating with %d mini-batch" % bs)
        #start_time = time.time()
        model.update_weights(images, network[bs], labels, 0.001)
        #print("--- update weights %s seconds ---" % (time.time() - start_time))

        h_proposed = [model.propose_new_state(images, net, labels) for (images, labels), net in zip(train_ds, network)]
        network_new = [model.accept_reject(images, net, net_proposed, labels) 
                            for (images, labels), net, net_proposed in zip(train_ds, network, h_proposed)]
        network = network_new
        #print("--- one batch %s seconds ---" % (time.time() - start_time))

# predict
print("begin prediction")
#start_time = time.time()
prediction_labels = [model.get_predictions(images) for images, labels in test_ds]
#print("--- get predictions %s seconds ---" % (time.time() - start_time))
#print(prediction_labels)

y_pred = sum([list(label) for label in prediction_labels], [])
acc = accuracy_score(y_test_sub, y_pred)
print("accuracy = ", acc)
'''