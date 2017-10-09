import copy, numpy as np # Python math libraries
import utilities # My utility functions, in a separate file

class LSTM_RNN:
    
    def __init__(self, LEARNING_RATE, IN_DIM, H_DIM, OUT_DIM):

        self.LEARNING_RATE = LEARNING_RATE
        self.IN_DIM        = IN_DIM
        self.H_DIM         = H_DIM
        self.OUT_DIM       = OUT_DIM

        # "Synapses", i.e. weights between layers
        self.syn = {
            'in_h': gen_lstm_syn(H_DIM, IN_DIM, False),
            'h_h': gen_lstm_syn(H_DIM, H_DIM, False),
            'h_out': np.random.random((OUT_DIM, H_DIM)) * 0.01
        }

        # Biases
        self.bias = {
            'out': np.zeros((OUT_DIM, 1))
            # TODO: add biases to LSTM cells
        }

    # Forward and backward propagation
    def propagation(self, inputs, targets, prev_h_state_init, prev_c_state_init):

        # "Layers" hold the nodes of the neural network, and the states of the LSTM cell, at each point in time
        layers = {
            'in': {}, # Input layer
                      # Internal state of the LSTM cell
            'c': {},  # Value of the cell
            'i': {},  # Input gate
            'f': {},  # Forget gate
            'o': {},  # Output gate
            'g': {},  # Transformation gate (before cell values are updated)

            'h': {},  # Hidden layer
            'out': {} # Output layer
        }

        # Keep track of the total loss
        loss = 0

        # Forward propagration
        for t in xrange(len(inputs)):

            # Encoding input layer
            layers['in'][t]  = one_hot_vec(self.IN_DIM, inputs[t])

            # Edge case for initial hidden and cell states
            prev_h           = layers['h'][t - 1] if t > 0 else prev_h_state_init
            prev_c           = layers['c'][t - 1] if t > 0 else prev_c_state_init

            # Calculating LSTM cell gate values
            layers['i'][t] = sigmoid(np.dot(self.syn['in_h']['i'], layers['in'][t]) + np.dot(self.syn['h_h']['i'], prev_h), False)
            layers['f'][t] = sigmoid(np.dot(self.syn['in_h']['f'], layers['in'][t]) + np.dot(self.syn['h_h']['f'], prev_h), False)
            layers['o'][t] = sigmoid(np.dot(self.syn['in_h']['o'], layers['in'][t]) + np.dot(self.syn['h_h']['o'], prev_h), False)
            layers['g'][t] = tanh(np.dot(self.syn['in_h']['g'], layers['in'][t]) + np.dot(self.syn['h_h']['g'], prev_h), False)
            
            # Updating the cell value
            layers['c'][t] = np.multiply(prev_c, layers['f'][t].T) + np.multiply(layers['g'][t], layers['i'][t].T)

            # Output of the LSTM cell
            layers['h'][t] = np.multiply(tanh(layers['c'][t], False), layers['o'][t])
                        
            # Propagating to the output layer and normalising
            layers['out'][t] = normalise(np.exp(np.dot(self.syn['h_out'], layers['h'][t]) + self.bias['out']))
            
            # Softmax loss
            loss             += -np.log(layers['out'][t][targets[t], 0])

        # Initialising the derivatives 
        D_syn = {
            'in_h': init_dict(self.syn['in_h']),
            'h_h': init_dict(self.syn['h_h']),
            'h_out': np.zeros_like(self.syn['h_out'])
        }
        D_bias       = init_dict(self.bias)
        next_h_deriv = np.zeros_like(layers['h'][0])    

        # Initialising adagrad memory variables
        M_syn = {
            'in_h': init_dict(self.syn['in_h']),
            'h_h': init_dict(self.syn['h_h']),
            'h_out': np.zeros_like(self.syn['h_out'])
        }
        M_bias  = init_dict(self.bias)    

        # Backward propagation
        for t in reversed(xrange(len(inputs))):

            # Softmax loss
            D_out             = np.copy(layers['out'][t])
            D_out[targets[t]] -= 1
            
            # Hidden to output
            D_syn['h_out'] += np.dot(D_out, layers['h'][t].T)
            D_bias['out']  += D_out
            
            # Backpropagrate through the LSTM cell
            D_o        = np.multiply(layers['h'][t], tanh(layers['c'][t], False))
            D_c[t]     += np.multiply(layers['h'][t], np.multiply(layers['o'][t], tanh(layers['c'][t], True)))
            D_i        = np.multiply(D_c[t], layers['g'][t])
            D_g        = np.multiply(D_c[t], layers['i'][t])
            prev_c     = layers['c'][t - 1] if t > 0 else prev_c_state_init
            D_f        = np.multiply(D_c[t], prev_c)
            D_c[t - 1] = np.multiply(D_c[t], layers['f'][t])
            prev_h     = layers['h'][t - 1] if t > 0 else prev_h_state_init
            D_g_prime  = np.multiply(D_g, (np.dot(self.syn['in_h']['g'], layers['in'][t]) + np.dot(self.syn['h_h']['g'], prev_h) + self.bias['h']['g']))
            D_i_prime  = np.multiply(D_i, np.multiply (layers['i'][t], 1 - layers['i'][t]))
            D_f_prime  = np.multiply(D_f, np.multiply (layers['f'][t], 1 - layers['f'][t]))
            D_o_prime  = np.multiply(D_o, np.multiply (layers['o'][t], 1 - layers['o'][t]))

            # Multiply the first set of gradients by the input
            D_syn['in_h']['i'] = np.dot(D_i_prime, layers['in'][t].T)
            D_syn['in_h']['f'] = np.dot(D_f_prime, layers['in'][t].T)
            D_syn['in_h']['o'] = np.dot(D_o_prime, layers['in'][t].T)
            D_syn['in_h']['g'] = np.dot(D_g_prime, layers['in'][t].T)

            # Multiply the second set by the hidden state
            D_syn['h_h']['i'] = np.dot(D_i_prime, prev_h.T)
            D_syn['h_h']['f'] = np.dot(D_f_prime, prev_h.T)
            D_syn['h_h']['o'] = np.dot(D_o_prime, prev_h.T)
            D_syn['h_h']['g'] = np.dot(D_g_prime, prev_h.T)

        # Adagrad paramater update
        for p, D_p, M_p in zip([self.syn['in_h']['i'], self.syn['in_h']['f'], self.syn['in_h']['o'], self.syn['in_h']['g'], self.syn['h_h']['i'], self.syn['h_h']['f'], self.syn['h_h']['o'], self.syn['h_h']['g'], self.syn['h_out'], self.bias['out']],
                                [D_syn['in_h']['i'], D_syn['in_h']['f'], D_syn['in_h']['o'], D_syn['in_h']['g'], D_syn['h_h']['i'], D_syn['h_h']['f'], D_syn['h_h']['o'], D_syn['h_h']['g'], D_syn['h_out'], D_bias['out']],
                                [M_syn['in_h']['i'], M_syn['in_h']['f'], M_syn['in_h']['o'], M_syn['in_h']['g'], M_syn['h_h']['i'], M_syn['h_h']['f'], M_syn['h_h']['o'], M_syn['h_h']['g'], M_syn['h_out'], M_bias['out']]):
            M_p += D_p * D_p
            p += -(self.LEARNING_RATE) * D_p / np.sqrt(M_p + 1e-8) # In case of a division by zero error

        # Pass out the latest hidden state
        return layers['h'][len(inputs)-1]

    # The training function
    def train(self, iterations, inputs, targets):

        # (Arbitary) training input sequence length
        SEQ_LEN = 200

        n = 0 # Iteration count
        p = 0 # Pointer to the position in the data
        while n <= iterations:

            # Cycling over the data 
            if (p + SEQ_LEN + 1 >= len(inputs)) or n == 0:
                p = 0
                # Resetting the hidden state
                h = np.zeros((self.H_DIM, 1))
                c = np.zeros((self.H_DIM, 1))

            # Slicing the inputs we want in this sequence
            seq_inputs  = inputs[p:p+SEQ_LEN]
            seq_targets = targets[p:p+SEQ_LEN]

            # Propagrate forwards and backwards through the network, returning a new hidden state
            h = self.propagation(inputs, targets, h, c)  

            p += SEQ_LEN                                                                                                                                                      
            n += 1
        