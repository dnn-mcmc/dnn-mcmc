import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.math as tm
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

def convert2_zero_one(x):
    
    t = [tf.math.sigmoid(i) for i in x]    
    return t

def cont_bern_log_norm(lam, l_lim=0.49, u_lim=0.51):
    '''
    computes the log normalizing constant of a continuous Bernoulli distribution in a numerically stable way.
    returns the log normalizing constant for lam in (0, l_lim) U (u_lim, 1) and a Taylor approximation in
    [l_lim, u_lim].
    cut_y below might appear useless, but it is important to not evaluate log_norm near 0.5 as tf.where evaluates
    both options, regardless of the value of the condition.
    '''
    
    cut_lam = tf.where(tm.logical_or(tm.less(lam, l_lim), tm.greater(lam, u_lim)), lam, l_lim * tf.ones_like(lam))
    log_norm = tm.log(tm.abs(2.0 * tm.atanh(1 - 2.0 * cut_lam))) - tm.log(tm.abs(1 - 2.0 * cut_lam))
    taylor = tm.log(2.0) + 4.0 / 3.0 * tm.pow(lam - 0.5, 2) + 104.0 / 45.0 * tm.pow(lam - 0.5, 4)
    return tf.where(tm.logical_or(tm.less(lam, l_lim), tm.greater(lam, u_lim)), log_norm, taylor)

class StochasticMLP(Model):
    
    def __init__(self, hidden_layer_sizes=[100], n_outputs=10, lr=1e-3):
        super(StochasticMLP, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.fc_layers = [Dense(layer_size) for layer_size in hidden_layer_sizes]
        self.output_layer = Dense(n_outputs)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate = lr)
        
    def call(self, x):
        
        network = []
        
        for i, layer in enumerate(self.fc_layers):
            
            logits = layer(x)
            x = tfp.distributions.Bernoulli(logits=logits).sample()
            network.append(x)

        final_logits = self.output_layer(x) # initial the weight of output layer
            
        return network

    def get_weight(self):
        
        weights = []
        for layer in self.fc_layers:
            weights.append(layer.get_weights())
        
        weights.append(self.output_layer.get_weights())
        
        return weights
    
    def target_log_prob(self, x, h, y, is_gibbs = False, is_hmc = False, is_loss = False):
        
        # get current state
        if is_hmc:
            h_current = tf.split(h, self.hidden_layer_sizes, axis = 1)
        else:    
            h_current = [tf.cast(h_i, dtype=tf.float32) for h_i in h]
        h_current = convert2_zero_one(h_current)
        h_previous = [x] + h_current[:-1]
    
        nlog_prob = 0. # negative log probability
        
        if not is_loss:
            for i, (cv, pv, layer) in enumerate(zip(h_current, h_previous, self.fc_layers)):
            
                logits = layer(pv)
                ce = tf.nn.sigmoid_cross_entropy_with_logits(labels = cv, logits = logits)
                if not is_gibbs:
                    ce += cont_bern_log_norm(tf.nn.sigmoid(logits))
            
                nlog_prob += tf.reduce_sum(ce, axis = -1)
        
        fce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(y, tf.float32), logits=self.output_layer(h_current[-1]))
        nlog_prob += tf.reduce_sum(fce, axis = -1)
            
        return -1 * nlog_prob
    
    def gibbs_new_state(self, x, h, y):
        
        '''
            generate a new state for the network node by node in Gibbs setting.
        '''
        
        h_current = h
        h_current = [tf.cast(h_i, dtype=tf.float32) for h_i in h_current]
        
        in_layers = self.fc_layers
        out_layers = self.fc_layers[1:] + [self.output_layer]
        
        prev_vals = [x] + h_current[:-1]
        curr_vals = h_current
        next_vals = h_current[1:] + [y]
        
        for i, (in_layer, out_layer, pv, cv, nv) in enumerate(zip(in_layers, out_layers, prev_vals, curr_vals, next_vals)):

            # node by node
            
            nodes = tf.transpose(cv)
            prob_parents = tm.sigmoid(in_layer(pv))
            
            out_layer_weights = out_layer.get_weights()[0]
            
            next_logits = out_layer(cv)
            
            new_layer = []
            
            for j, node in enumerate(nodes):
                
                # get info for current node (i, j)
                
                prob_parents_j = prob_parents[:, j]
                out_layer_weights_j = out_layer_weights[j]
                
                # calculate logits and logprob for node is 0 or 1
                next_logits_if_node_0 = next_logits[:, :] - node[:, None] * out_layer_weights_j[None, :]
                next_logits_if_node_1 = next_logits[:, :] + (1 - node[:, None]) * out_layer_weights_j[None, :]
                
                #print(next_logits_if_node_0, next_logits_if_node_1)
                
                logprob_children_if_node_0 = -1 * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(nv, dtype = tf.float32), logits=next_logits_if_node_0), axis = -1)
                
                logprob_children_if_node_1 = -1 * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(nv, dtype = tf.float32), logits=next_logits_if_node_1), axis = -1)
                
                # calculate prob for node (i, j)
                prob_0 = (1 - prob_parents_j) * tm.exp(logprob_children_if_node_0)
                prob_1 = prob_parents_j * tm.exp(logprob_children_if_node_1)
                prob_j = prob_1 / (prob_1 + prob_0)
            
                # sample new state with prob_j for node (i, j)
                new_node = tfp.distributions.Bernoulli(probs = prob_j).sample() # MAY BE SLOW
                
                # update nodes and logits for following calculation
                new_node_casted = tf.cast(new_node, dtype = "float32")
                next_logits = next_logits_if_node_0 * (1 - new_node_casted)[:, None] \
                            + next_logits_if_node_1 * new_node_casted[:, None] 
                
                # keep track of new node values (in prev/curr/next_vals and h_new)
                new_layer.append(new_node)
           
            new_layer = tf.transpose(new_layer)
            h_current[i] = new_layer
            prev_vals = [x] + h_current[:-1]
            curr_vals = h_current
            next_vals = h_current[1:] + [y]
        
        return h_current
    
    def generate_hmc_kernel(self, x, y, step_size = pow(1000, -1/4)):
        
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn = lambda v: self.target_log_prob(x, v, y, is_hmc = True),
            num_leapfrog_steps = 2,
            step_size = step_size),
            num_adaptation_steps = int(1000 * 0.8))
        
        return adaptive_hmc
    
    # new proposing-state method with HamiltonianMonteCarlo
    def propose_new_state_hamiltonian(self, x, h, y, hmc_kernel, is_update_kernel = True):
    
        h_current = h
        h_current = [tf.cast(h_i, dtype=tf.float32) for h_i in h_current]
        h_current = tf.concat(h_current, axis = 1)

        # run the chain (with burn-in)
        num_burnin_steps = 0
        num_results = 1

        samples = tfp.mcmc.sample_chain(
            num_results = num_results,
            num_burnin_steps = num_burnin_steps,
            current_state = h_current, # may need to be reshaped
            kernel = hmc_kernel,
            trace_fn = None,
            return_final_kernel_results = True)
    
        # Generate new states of chains
        #h_state = rerange(samples[0][0])
        h_state = samples[0][0]
        h_new = tf.split(h_state, self.hidden_layer_sizes, axis = 1) 
        
        # Update the kernel if necesssary
        if is_update_kernel:
            new_step_size = samples[2].new_step_size.numpy()
            ker_new = self.generate_hmc_kernel(x, y, new_step_size)
            return(h_new, ker_new)
        else:
            return h_new
    
    def update_weights(self, x, h, y, is_gibbs = False):
        
        with tf.GradientTape() as tape:
            loss = -1 * tf.reduce_mean(self.target_log_prob(x, h, y, is_gibbs = is_gibbs))
        
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    
    def get_predictions(self, x):

        logits = 0.0
        for layer in self.fc_layers:
            logits = layer(x)
            x = tm.sigmoid(logits)
        
        logits = self.output_layer(x)
        probs = tm.sigmoid(logits)
        labels = tf.cast(tm.greater(probs, 0.5), tf.int32)

        return labels
    
    def get_loss(self, x, y):
        
        logits = 0.0
        for layer in self.fc_layers:
            logits = layer(x)
            x = tm.sigmoid(logits)
            
        logits = self.output_layer(x)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.cast(y, tf.float32), logits = logits)
        
        return tf.reduce_sum(loss, axis = -1)
    
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
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    st = time.time()
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    history = model.fit(dat, batch_size=batch_size, epochs=epochs)
    train_time = time.time() - st
    
    return train_time, history

def hmc(size, dat, epochs, burnin = 500, nrec = 100):
    '''
    HMC training
    '''
    
    targets = np.concatenate([target for data, target in dat.as_numpy_iterator()])
    
    print("Start HMC")
    model = StochasticMLP(hidden_layer_sizes = [size], n_outputs=1, lr = 0.01)
    network = [model.call(data) for data, target in dat]
    kernels = [model.generate_hmc_kernel(data, target) for data, target in dat]
    
    print("Start HMC Burning")
    burnin_losses = []
    for i in range(burnin):
        
        if(i % 100 == 0): print("Step %d" % i)

        res = []
        burnin_loss = 0.0
        for bs, (data, target) in enumerate(dat):
            res.append(model.propose_new_state_hamiltonian(data, network[bs], target, kernels[bs]))
            burnin_loss += -1 * tf.reduce_sum(model.target_log_prob(data, network[bs], target))
    
        network, kernels = zip(*res)
        burnin_losses.append(burnin_loss / (bs + 1))
        
    
    print("Start HMC Training")
    
    losses = []
    accs = []
    start_time = time.time()
    samples = []
    
    for epoch in range(epochs):
        
        for bs, (data, target) in enumerate(dat):
        
            model.update_weights(data, network[bs], target)
            network = [model.propose_new_state_hamiltonian(x, net, y, ker, is_update_kernel = False) \
                       for (x, y), net, ker in zip(dat, network, kernels)]
        
        if epoch >= epochs - nrec:
            sample = {"net": network, "weight": model.get_weight()}
            samples.append(sample)  
        
        loss = 0.0
        for data, target in dat:
            loss += tf.reduce_mean(model.get_loss(data, target))
        loss /= (bs + 1)
        losses.append(loss)       
        
        preds = [model.get_predictions(data) for data, target in dat]
        acc = accuracy_score(np.concatenate(preds), targets)
        accs.append(acc)
    
        print("Epoch %d/%d: - %.4fs/step - loss: %.4f - accuracy: %.4f" 
            % (epoch + 1, epochs, (time.time() - start_time) / (epoch + 1), loss, acc))

    train_time = time.time() - start_time
    return samples, burnin_losses, train_time, {"acc": accs, "loss": losses}

def gibbs(size, dat, epochs, burnin = 500, nrec = 100):
    '''
    Gibbs Training
    '''
    
    targets = np.concatenate([target for data, target in dat.as_numpy_iterator()])
    
    print("Start Gibbs")
    model = StochasticMLP(hidden_layer_sizes = [size], n_outputs=1, lr = 0.01)
    network = [model.call(data) for data, target in dat]

    print("Start Gibbs Burning")    
    burnin_losses = []
    for i in range(burnin):
    
        if(i % 100 == 0): print("Step %d" % i)

        res = []
        burnin_loss = 0.0
        for bs, (data, target) in enumerate(dat):
            res.append(model.gibbs_new_state(data, network[bs], target))
            burnin_loss += -1 * tf.reduce_sum(model.target_log_prob(data, network[bs], target, is_gibbs = True))
            
        network = res
        burnin_losses.append(burnin_loss / (bs + 1))
    
    # Training
    losses = []
    accs = []
    start_time = time.time()
    samples = []
    
    for epoch in range(epochs):
        
        # train
        for bs, (data, target) in enumerate(dat):
        
            model.update_weights(data, network[bs], target, is_gibbs = True)
            network = [model.gibbs_new_state(x, net, y) for (x, y), net in zip(dat, network)]
            
        if epoch >= epochs - nrec:
            sample = {"net": network, "weight": model.get_weight()}
            samples.append(sample)
            
        loss = 0.0
        for data, target in dat:
            loss += tf.reduce_mean(model.get_loss(data, target))
        loss /= (bs + 1)
        losses.append(loss)       
        
        preds = [model.get_predictions(data) for data, target in dat]
        acc = accuracy_score(np.concatenate(preds), targets)
        accs.append(acc)
    
        print("Epoch %d/%d: - %.4fs/step - loss: %.4f - accuracy: %.4f" 
              % (epoch + 1, epochs, (time.time() - start_time) / (epoch + 1), loss, acc))

    train_time = time.time() - start_time
    return samples, burnin_losses, train_time, {"acc": accs, "loss": losses}

x_train = np.array([[0, 0],
           [0, 1],
           [1, 0],
           [1, 1]])
y_train = np.array([[0],
           [1],
           [1],
           [0]])
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(4)

N = 1
epochs = 10
burnin = 10
nrec = 5
size = 32
res_bp, res_hmc, res_gibbs = [], [], []
samples_hmc, samples_gibbs = [], []

for i in range(N):
    
    print("---------------------------------------")
    print(f"Running {i}")
    
    
    time_bp, history_bp = standard_backprop(size, train_ds, epochs)
    s_hmc, burnin_loss_hmc, time_hmc, history_hmc = hmc(size, train_ds, epochs, burnin, nrec)
    s_gibbs, burnin_loss_gibbs, time_gibbs, history_gibbs = gibbs(size, train_ds, epochs, burnin, nrec)
    
    hist_bp = {"acc": history_bp.history['accuracy'], "loss": history_bp.history['loss']}
    rbp = {'time': time_bp, 'history': hist_bp}
    rhmc = {'time': time_hmc, 'burnin': burnin_loss_hmc, 'history': history_hmc}
    rgibbs = {'time': time_gibbs, 'burnin': burnin_loss_gibbs, 'history': history_gibbs}
    
    res_bp.append(rbp)
    res_hmc.append(rhmc)
    res_gibbs.append(rgibbs)
    
    samples_hmc.append(s_hmc)
    samples_gibbs.append(s_gibbs)

res_all = [res_bp, res_hmc, res_gibbs]
samples_all = [samples_hmc, samples_gibbs]

with open('hist_xor.npy', 'wb') as f:
    np.save(f, np.array(res_all))
    
with open('sample_xor.npy', 'wb') as f:
    np.save(f, np.array(samples_all))

# calculate average curve for each method
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
avg_hmc = cal_avg(res_hmc)
avg_gibbs = cal_avg(res_gibbs)
avg_all = [avg_bp, avg_hmc, avg_gibbs]
    
# plot all the running times for each method
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
    
methods = ['bp', 'hmc', 'gibbs']
metrics = ['acc', 'loss']
for i, method in enumerate(methods):
    for metric in metrics:
        plot_all(res_all[i], method, metric)

plt.style.use('seaborn')
for metric in metrics:
    for i, method in enumerate(methods):
        plt.plot(avg_all[i][metric], label = method)
    plt.title(metric)
    plt.xlabel("epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig("average_" + metric + '.pdf')
    plt.close()
    