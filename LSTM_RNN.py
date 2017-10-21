import utils
import numpy as np

class LSTM_RNN:
    def __init__(self, lr, in_dim, h_him, out_dim):

        # Hyperparamaters
        self.lr = lr  # Learning rate
        self.in_dim = in_dim  # Dimension of input layer
        self.h_dim = h_him  # Dimension of hidden layer
        self.out_dim = out_dim  # Dimension of output layer

        # "Variables", i.e. the parameters to learn (inspired by Tensorflow)
        self.vars = {
            # Weights between input and hidden state
            'in_h': init_lstm_weights(h_him, in_dim, False),
            # Weights between hidden state at consecutive timesteps
            'h_h': init_lstm_weights(h_him, h_him, False),
            # Weights between hidden and output state
            'h_out': np.random.random((out_dim, h_him)) * 0.01,
            # Output layer bias
            'bias': np.zeros(out_dim)
        }

        # Holds the state of the neural network at each point in time
        self.state = {
            'in': {},  # Input layer

            # Internal state of the LSTM cell
            'c': {},  # Value of the cell
            'i': {},  # Input gate
            'f': {},  # Forget gate
            'o': {},  # Output gate
            'g': {},  # Transformation gate (before cell values are updated)

            'h': {},  # Hidden layer
            'out': {}  # Output layer
        }

    # Forward Propagation
    def forward_pass(self, inputs, targets, first_prev_h, first_prev_c):
        
        loss = 0

        # For each input in the set
        for t in xrange(len(inputs)):

            # Encode input layer
            self.state['in'][t] = one_hot_vec(self.in_dim, inputs[t])

            # Get previous states if available
            prev_h = self.state['h'][t - 1] if t > 0 else first_prev_h
            prev_c = self.state['h'][t - 1] if t > 0 else first_prev_c

            # Update LSTM cell gates and value
            self.state['i'][t] = sigmoid(
                np.dot(self.vars['in_h']['i'], self.state['in'][t]) +
                np.dot(self.vars['h_h']['i'], prev_h),
                False)
            self.state['f'][t] = sigmoid(
                np.dot(self.vars['in_h']['f'], self.state['in'][t]) +
                np.dot(self.vars['h_h']['f'], prev_h),
                False)
            self.state['o'][t] = sigmoid(
                np.dot(self.vars['in_h']['o'], self.state['in'][t]) +
                np.dot(self.vars['h_h']['o'], prev_h),
                False)
            self.state['g'][t] = tanh(
                np.dot(self.vars['in_h']['g'], self.state['in'][t]) +
                np.dot(self.vars['h_h']['g'], prev_h),
                False)
            self.state['c'][t] = np.multiply(prev_c, self.state['f'][t].T) +\
                np.multiply(self.state['g'][t], state['i'][t].T)
            
            # Propagating from the cell to the hidden layer
            self.state['h'][t] = np.multiply(
                tanh(self.state['c'][t], False),
                self.state['o'][t])

            # Softmax loss
            loss += -np.log(state['out'][t][targets[t], 0])
        
        # Return the loss
        return loss

    # Backward Bropagation
    def backward_pass(self, inputs, targets, first_prev_h, first_prev_c):

        # Initialising gradients
        D_vars = {
            'in_h': init_dict_like(self.vars['in_h']),
            'h_h': init_dict_like(self.vars['h_h']),
            'h_out': np.zeros_like(self.vars['h_out']),
            'bias': np.zeros(OUT_DIM)
        }

        # Initialising Adagrad memory variables
        M_vars = {
            'in_h': init_dict_like(self.vars['in_h']),
            'h_h': init_dict_like(self.vars['h_h']),
            'h_out': np.zeros_like(self.vars['h_out']),
            'bias': np.zeros(OUT_DIM)
        }

        # Backward propagation
        for t in reversed(xrange(len(inputs))):

            # Softmax loss
            D_out = np.copy(self.state['out'][t])
            D_out[targets[t]] -= 1

            # Hidden to output
            D_vars['h_out'] += np.dot(D_out, self.state['h'][t].T)
            D_vars['bias'] += D_out

            # Backpropagate through the LSTM cell
            D_o = np.multiply(
                self.state['h'][t],
                tanh(self.state['c'][t], False))
            D_c[t] += np.multiply(
                self.state['h'][t],
                np.multiply(
                    self.state['o'][t],
                    tanh(self.state['c'][t], True)))
            D_i = np.multiply(D_c[t], self.state['g'][t])
            D_g = np.multiply(D_c[t], self.state['i'][t])
            prev_c = self.state['c'][t - 1] if t > 0 else first_prev_c
            D_f = np.multiply(D_c[t], prev_c)
            D_c[t - 1] = np.multiply(D_c[t], self.state['f'][t])
            prev_h = self.state['h'][t - 1] if t > 0 else first_prev_h
            D_prime = {}
            D_prime['g'] = np.multiply(
                D_g,
                (np.dot(self.vars['in_h']['g'], self.state['in'][t]) +
                np.dot(self.vars['h_h']['g'], prev_h)))
            D_prime['i'] = np.multiply(
                D_i,
                np.multiply(self.state['i'][t], 1 - self.state['i'][t]))
            D_prime['f'] = np.multiply(
                D_f,
                np.multiply(self.state['f'][t], 1 - self.state['f'][t]))
            D_prime['o'] = np.multiply(
                D_o, np.multiply(self.state['o'][t], 1 - self.state['o'][t]))

            for k, d in D_prime.iteritems():
                # Multiply the first set of gradients by the input
                D_vars['in_h'][k] = np.dot(d, self.state['in'][t].T)
                # Multiply the second set by the previous hidden state
                D_vars['h_h'][k] = np.dot(d, prev_h.T)

        # Adagrad parameter update
        for p, D_p, M_p in zip(
                [self.vars['in_h']['i'], self.vars['in_h']['f'],
                 self.vars['in_h']['o'], self.vars['in_h']['g'],
                 self.vars['h_h']['i'], self.vars['h_h']['f'],
                 self.vars['h_h']['o'], self.vars['h_h']['g'],
                 self.vars['h_out'], self.vars['bias']],
                [D_vars['in_h']['i'], D_vars['in_h']['f'], D_vars['in_h']['o'],
                 D_vars['in_h']['g'],
                 D_vars['h_h']['i'], D_vars['h_h']['f'], D_vars['h_h']['o'],
                 D_vars['h_h']['g'], D_vars['h_out'],
                 D_vars['bias']],
                [M_vars['in_h']['i'], M_vars['in_h']['f'], M_vars['in_h']['o'],
                 M_vars['in_h']['g'],
                 M_vars['h_h']['i'], M_vars['h_h']['f'], M_vars['h_h']['o'],
                 M_vars['h_h']['g'], M_vars['h_out'],
                 M_vars['bias']]):
            M_p += D_p * D_p
            p += -(self.lr) * D_p / np.sqrt(M_p + 1e-8)  # Avoiding x / 0
            

        # Return latest hidden state
        return self.state['h'][len(inputs) - 1]

    # The training function
    def train(self, iterations, inputs, targets, slice_len):

        n = 0  # Iteration count
        p = 0  # Pointer to the position in the data
        while n <= iterations:

            # Resetting pointers and states if we reach the end of our data
            if (p + slice_len + 1 >= len(inputs)) or n == 0:
                p = 0
                h = np.zeros((self.h_dim, 1))
                c = np.zeros((self.h_dim, 1))

            # Slicing the inputs we want for this pass
            inputs_slice = inputs[p:p + slice_len]
            targets_slice = targets[p:p + slice_len]

            # Forward Propogation
            loss = self.forward_pass(inputs_slice, targets_slice, h, c)
            # Backwards Propogation
            h = self.backward_pass(inputs_slice, targets_slice, h, c)
            
            p += slice_len
            n += 1
