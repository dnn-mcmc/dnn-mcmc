import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

def standard_backprop(size, dat, epochs):
    '''
    Standard Backpropogation training
    '''
    
    batch_size = 4
    
    print("Start Standard Backprop")
    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(2,)),
            layers.Dense(size, activation = "sigmoid"),
            layers.Dense(1, activation = "sigmoid")
        ]
    )   
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    st = time.time()
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    history = model.fit(dat, batch_size=batch_size, epochs=epochs)
    train_time = time.time() - st
    
    return train_time, history

# XOR data
x_train = np.array([[0, 0],
           [0, 1],
           [1, 0],
           [1, 1]])
y_train = np.array([[0],
           [1],
           [1],
           [0]])
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(4)

# training
N = 10
epochs = 10000
size = 32
res_bp = []

for i in range(N):
    
    print("---------------------------------------")
    print(f"Running {i}")
    
    time_bp, history_bp = standard_backprop(size, train_ds, epochs)
    hist_bp = {"acc": history_bp.history['accuracy'], "loss": history_bp.history['loss']}
    rbp = {'time': time_bp, 'history': hist_bp}
    
    res_bp.append(rbp)
    
def cal_avg(res):
    
    metrics = ['acc', 'loss']
    avg = {}
    for metric in metrics:
        arr_metric = np.zeros((N, epochs))
        for i in range(N):
            arr_metric[i] = np.array(res[i]['history'][metric])
        avg_metric = np.mean(arr_metric, axis = 0)
        avg[metric] = avg_metric
        
    return avg

avg_bp = cal_avg(res_bp)

def plot_all(res, method, metric):
    
    plt.style.use('seaborn')
    nrow = 4
    ncol = 3
    
    fig, ax = plt.subplots(nrow, ncol, sharex = True)
    fig.suptitle(method + "_" + metric)
    for i in range(nrow):
        for j in range(ncol):
            if i * ncol + j < N:
                ax[i, j].plot(res[i * ncol + j]['history'][metric])
                ax[i, j].set_title(f"Run {i * ncol + j}")
    plt.savefig(method + "_" + metric + '.pdf')
    plt.close()
    
metrics = ['acc', 'loss']

for metric in metrics:
    plot_all(res_bp, 'bp', metric)

plt.style.use('seaborn')
for metric in metrics:
    plt.plot(avg_bp[metric], label = 'bp')
    plt.title(metric)
    plt.xlabel("epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig("average_" + metric + '.pdf')
    plt.close()