import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras import initializers


class StochasticMLP(Model):
    
    def __init__(self, hidden_layer_sizes=[100], n_outputs=10):
        super(StochasticMLP, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.fc_layers = [Dense(layer_size) for layer_size in hidden_layer_sizes]
        self.output_layer = Dense(n_outputs)
    
    def call(self, x):
        
        x = Flatten()(x)
        
        network = []
        
        for i, layer in enumerate(self.fc_layers):
            
            logits = layer(x)
            x = tfp.distributions.Bernoulli(logits=logits).sample()
            network.append(x)

            #network['h%i_logits' % i] = logits
            #network['h%i_values' % i] = x
        
        final_logits = self.output_layer(x) # initial the weight of output layer
            
        return network
    
    
    def propose_new_state(self, x, h, y):
        '''returns new proposed h values
        x: inputs
        h: list of layer values
        y: labels
        returns h_proposed'''
        
        x = Flatten()(x)

        #h_current = [h['h%i_values' % i] for i in range(len(self.fc_layers))]
        h_current = h
        h_current = [tf.cast(h_i, dtype=tf.float32) for h_i in h_current]
        
        in_layers = self.fc_layers
        out_layers = self.fc_layers[1:] + [self.output_layer]
        
        prev_vals = [x] + h_current[:-1]
        curr_vals = h_current
        next_vals = h_current[1:] + [y]
        
        h_new = []
        
        for i, (in_layer, out_layer, pv, cv, nv) in enumerate(
            zip(in_layers, out_layers, prev_vals, curr_vals, next_vals)):
            
            prob_parents = tf.math.sigmoid(in_layer(pv))
            
            out_layer_weights = out_layer.get_weights()[0]
            
            next_logits = out_layer(cv)
            
            # if h1 node is a 1, subtract its weight
            next_logits_if_node_is_0 = next_logits[:, tf.newaxis, :] - cv[:, :, np.newaxis] * out_layer_weights[tf.newaxis, :, :]
        
            # if h1 node is a 0, add its weight
            next_logits_if_node_is_1 = next_logits[:, tf.newaxis, :] + (1 - cv[:, :, np.newaxis]) * out_layer_weights[tf.newaxis, :, :]
            
            if i < (len(curr_vals) - 1):
                
                nv_tiled = tf.cast(np.tile(nv[:, np.newaxis, :], (1, cv.shape[-1], 1)), dtype=tf.float32)
                
                logprob_children_if_node_is_0 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=nv_tiled, logits=next_logits_if_node_is_0), axis=-1)

                logprob_children_if_node_is_1  = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=nv_tiled, logits=next_logits_if_node_is_1), axis=-1)
                
            else:
                
                nv_tiled = tf.cast(np.tile(nv[:, np.newaxis], (1, cv.shape[-1])), dtype=tf.int32)
                
                logprob_children_if_node_is_0 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=nv_tiled, logits=next_logits_if_node_is_0)

                logprob_children_if_node_is_1  = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=nv_tiled, logits=next_logits_if_node_is_1)
            
            prob_0 = (1 - prob_parents) * tf.math.exp(logprob_children_if_node_is_0)
            prob_1 = prob_parents * tf.math.exp(logprob_children_if_node_is_1)
        
            prob = prob_1 / (prob_1 + prob_0)
            
            new_layer_state = tfp.distributions.Bernoulli(probs=prob).sample()
            h_new.append(new_layer_state)
       
        # not sample output labels
        # h_new['labels'] = tf.argmax(
        #     tfp.distributions.Multinomial(10, logits=self.output_layer(h_current[-1])).sample(),
        #     axis=1)
            
        return h_new

    
    def target_log_prob(self, x, h, y):
        
        x = Flatten()(x)
        
        h_current = h
        h_current = [tf.cast(h_i, dtype=tf.float32) for h_i in h_current]
        
        h_previous = [x] + h_current[:-1]
        
        nlog_prob = 0. # negative log probability
        
        for i, (cv, pv, layer) in enumerate(
            zip(h_current, h_previous, self.fc_layers)):
            
            ce = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=cv, logits=layer(pv))
            
            nlog_prob += tf.reduce_sum(ce, axis = -1)
            
        nlog_prob += tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(y, tf.int32), logits=self.output_layer(h_current[-1]))
            
        return -1 * nlog_prob
    
    def accept_reject(self, x, h, h_p, y):

        log_prob_curr = self.target_log_prob(x, h, y)
        log_prob_prop = self.target_log_prob(x, h_p, y)
        
        #ratio = [np.exp(-np.maximum(0, prop - curr)) for curr, prop in zip(log_prob_curr, log_prob_prop)]
        ratio = np.exp(-np.maximum(0, log_prob_prop - log_prob_curr))
        acceptance = tfp.distributions.Bernoulli(probs = ratio).sample()
        
        h_new = []
        
        for i in range(len(self.fc_layers)):
            #h_new['h%i_values' % i] = h_p['h%i_values' % i] * acceptance[:, np.newaxis] \
            #    + h['h%i_values' % i] * (1 - acceptance)[:, np.newaxis]
            acc_layer_state = h_p[i] * acceptance[:, np.newaxis] + h[i] * (1 - acceptance)[:, np.newaxis]
            h_new.append(acc_layer_state)
        
        return h_new
    
    # update weights using tensorflow functions
    def update_weights(self, x, h, y, lr = 0.001):
        
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        with tf.GradientTape() as tape:
            loss = -1 * tf.reduce_mean(self.target_log_prob(x, h, y))
        
        grads = tape.gradient(loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

    def target_log_prob2(self, x, h, y):

        x = Flatten()(x)
        h_current = tf.split(h, self.hidden_layer_sizes, axis = 1)
        h_previous = [x] + h_current[:-1]
        
        nlog_prob = 0.
        
        for i, (cv, pv, layer) in enumerate(
            zip(h_current, h_previous, self.fc_layers)):
            
            ce = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=cv, logits=layer(pv))
            
            nlog_prob += tf.reduce_sum(ce, axis = -1)
            
        nlog_prob += tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(y, tf.int32), logits=self.output_layer(h_current[-1]))
            
        return -1 * nlog_prob

    # new proposing-state method with HamiltonianMonteCarlo
    def propose_new_state_hamiltonian(self, x, h, y):
    
        h_current = h
        h_current = [tf.cast(h_i, dtype=tf.float32) for h_i in h_current]
        h_current = tf.concat([h_current[0], h_current[1]], axis=1)

        # initialize the HMC transition kernel
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn = lambda v: self.target_log_prob2(x, v, y),
            num_leapfrog_steps = 2,
            step_size = pow(1000, -1/4)),
            num_adaptation_steps=int(10 * 0.8))

        # run the chain (with burn-in)
        num_results = 1
        num_burnin_steps = 2

        samples = tfp.mcmc.sample_chain(
            num_results = num_results,
            num_burnin_steps = num_burnin_steps,
            current_state = h_current, # may need to be reshaped
            kernel = adaptive_hmc,
            trace_fn = None)

        new_state = tf.math.sign(tf.math.sign(samples[0]) - 1) + 1
        h_new = tf.split(new_state, self.hidden_layer_sizes, axis = 1)

        return(h_new)

    def get_predictions(self, x):
        x = Flatten()(x)

        for layer in self.fc_layers:
            logits = layer(x)
            x = tfp.distributions.Bernoulli(logits = logits).sample()
        
        final_logits = self.output_layer(x)
        #final_predictions = tf.nn.softmax(final_logits)
        final_labels = tf.argmax(tfp.distributions.Bernoulli(logits = final_logits).sample(), axis = 1)

        return final_labels

    def save_model(self, file):
        with open(file, 'wb') as f:
            for layer in self.fc_layers:
                np.save(f, np.array(layer.get_weights()))
            np.save(f, self.output_layer.get_weights())
    
    def load_model(self, file):
        with open(file, 'rb', file) as f:
            for layer in self.fc_layers:
                layer.set_weights(np.load(f, allow_pickle = True))
            self.output_layer.set_weights(np.load(f, allow_pickle = True))
