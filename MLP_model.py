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
        
        network = {}
        
        for i, layer in enumerate(self.fc_layers):
            
            logits = layer(x)
            x = tfp.distributions.Bernoulli(logits=logits).sample()
            
            #network['h%i_logits' % i] = logits
            network['h%i_values' % i] = x
            
        final_logits = self.output_layer(x)
        #final_predictions = tf.nn.softmax(final_logits)
        
        #network['final_logits'] = final_logits
        #network['final_predictions'] = final_predictions

        return network
    
    
    def propose_new_state(self, x, h, y):
        '''returns new proposed h values
        x: inputs
        h: dictionary of layer values
        y: labels
        
        returns h_proposed'''
        
        x = Flatten()(x)
        
        h_current = [h['h%i_values' % i] for i in range(len(self.fc_layers))]
        h_current = [tf.cast(h_i, dtype=tf.float32) for h_i in h_current]
        
        in_layers = self.fc_layers
        out_layers = self.fc_layers[1:] + [self.output_layer]
        
        prev_vals = [x] + h_current[:-1]
        curr_vals = h_current
        next_vals = h_current[1:] + [y]
        
        h_new = {}
        
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
            
            h_new['h%i_values' % i] = tfp.distributions.Bernoulli(probs=prob).sample()
       
        # not sample output labels
        # h_new['labels'] = tf.argmax(
        #     tfp.distributions.Multinomial(10, logits=self.output_layer(h_current[-1])).sample(),
        #     axis=1)
            
        return h_new

    
    def target_log_prob(self, x, h, y):
        
        x = Flatten()(x)
        
        h_current = [h['h%i_values' % i] for i in range(len(self.fc_layers))]
        h_current = [tf.cast(h_i, dtype=tf.float32) for h_i in h_current]
        
        h_previous = [x] + h_current[:-1]
        
        log_prob = 0.
        
        for i, (cv, pv, layer) in enumerate(
            zip(h_current, h_previous, self.fc_layers)):
            
            ce = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=cv, logits=layer(pv))
            
            log_prob += tf.reduce_sum(ce, axis = -1)
            
        log_prob += tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(y, tf.int32), logits=self.output_layer(h_current[-1]))
            
        return log_prob
    
    ### log prob should have shape (batch_size, )
    
    ### use this ^^ to determine whether to accept or reject samples
    ### this is done for each element of the batch, independently
        
    ### but then, to update weights, we want tf.reduce_mean(self.target_log_prob(...), axis=0)
    ### (should be only one axis at this point)
        
    ### can use e.g. loss = -1 * tf.reduce_mean(self.target_log_prob(...))
    ### tf.keras.optimizers.Adam().minimize(lambda: loss, var_list_fn=lambda: model.trainable_weights)
    
    
    def accept_reject(self, x, h, h_p, y):

        log_prob_curr = self.target_log_prob(x, h, y)
        log_prob_prop = self.target_log_prob(x, h_p, y)
        
        #ratio = [np.exp(-np.maximum(0, prop - curr)) for curr, prop in zip(log_prob_curr, log_prob_prop)]
        ratio = np.exp(-np.maximum(0, log_prob_prop - log_prob_curr))
        acceptance = tfp.distributions.Bernoulli(probs = ratio).sample()
        
        h_new = {}
        
        for i in range(len(self.fc_layers)):
            h_new['h%i_values' % i] = h_p['h%i_values' % i] * acceptance[:, np.newaxis] \
                + h['h%i_values' % i] * (1 - acceptance)[:, np.newaxis]
        
        # not necessary to sample output labels
        #h_new['labels'] = h_p['labels'] * acceptance[:, np.newaxis] + h['labels'] * (1 - acceptance)[:, np.newaxis]
        
        return h_new
    
    # manually update weights (not necessary)
    '''
    def weight_update(self, x, h, y, lr):
        
        x = Flatten()(x)
        
        h_current = [h['h%i_values' % i] for i in range(len(self.fc_layers))]
        h_current = [tf.cast(h_i, dtype=tf.float32) for h_i in h_current]
        
        layers = self.fc_layers + [self.output_layer]
        
        prev_vals = [x] + h_current
        next_vals = h_current + [y]
        
        learning_rate = lr
        for i, (layer, pv, nv) in enumerate(zip(layers, prev_vals, next_vals)):
            
            grad = pv.T @ (layer(pv) - nv)
            batch_grad = np.sum(grad, axis = 0)
            
            layer_weights = layer.get_weights()[0]
            layer_weights -= learning_rate * batch_grad
            layer.set_weights(layer_weights)
    '''
    
    # update weights using tensorflow functions
    def update_weights(self, x, h, y, lr = 0.001):
        
        #loss = -1 * tf.reduce_mean(self.target_log_prob(x, h, y))
        #tf.keras.optimizers.Adam().minimize(lambda: loss, lambda: self.trainable_weights)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        with tf.GradientTape() as tape:
            loss = -1 * tf.reduce_mean(self.target_log_prob(x, h, y))
        
        grads = tape.gradient(loss, self.trainable_weights)
        #print(grads)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))
    
    def get_predictions(self, x):
        x = Flatten()(x)

        for layer in self.fc_layers:
            logits = layer(x)
            x = tfp.distributions.Bernoulli(logits = logits).sample()
        
        final_logits = self.output_layer(x)
        #final_predictions = tf.nn.softmax(final_logits)
        final_labels = tf.argmax(tfp.distributions.Bernoulli(logits = final_logits).sample(), axis = 1)

        return final_labels

    #def target_log_prob_wrapped(self, current_state):


    # new proposing-state method with HamiltonianMonteCarlo
    def propose_new_state_hamiltoniam(self, x, h, y):
        
        # Initialize the HMC transition kernel.
        num_results = 1 # we only take one step
        num_burnin_steps = int(1e3)
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn = lambda h: self.target_log_prob(x, h, y),
                num_leapfrog_steps = 3,
                step_size = 1.),
            num_adaptation_steps=int(num_burnin_steps * 0.8))

        # build current state
        h_current = [h['h%i_values' % i] for i in range(len(self.fc_layers))]
        h_current = [tf.cast(h_i, dtype=tf.float32) for h_i in h_current] # may need to flatten

        # run the chain
        samples, is_accepted = tfp.mcmc.sample_chain(
            num_results = num_results,
            num_burnin_steps = num_burnin_steps,
            current_state = h_current, # may need to be reshaped
            kernel = adaptive_hmc,
            trace_fn = lambda _, pkr: pkr.inner_results.is_accepted)

        samples_mean = tf.reduce_mean(samples)

        # main time costing step: propose_new_state & accept/reject
        
        # Questions: 
        # 1. num_results? 
        # 2. target_log_prob_wrapped? (receives current_state = [x, h, y], but only samples h)
        # 2.5 h or h_current? (whether need to be flattened into lists according to the type of current_state)
        # 3. new state? (sample * is_accepted, then reduce_mean?)
        # 4.* mechanism of HMC (num_leapfrog_steps, step_size)

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
