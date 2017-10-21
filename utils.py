''' ---------------------------------
Utility Functions

Simple utility functions, placed in here to make the main code easier to read
--------------------------------- '''


# Initialises a dictionary with the same keys as d but with zero-vector values
def init_dict_like(d):
    return {k: np.zeros_like(v) for k, v in d.iteritems()}


# Normalises a vector
def normalise(vec):
    return vec / np.sum(vec)

# Generates a one-hot-vector of length len, with the ith element 1
def one_hot_vec(len, i):
    vec = np.zeros((len, 1))
    vec[i] = 1

    return vec


# TODO: possibly replace with hard sigmoid (faster)
def sigmoid(x, D):
    if not D:
        return 1 / (1 + np.exp(- x))
    else:
        s = sigmoid(x, False)
        return s - (s ** 2)


def tanh(x, D):
    if not D:
        return np.tanh(x)
    else:
        return 1.0 - (np.tanh(x) ** 2)


# Initialises the LSTM weight matrices
def init_lstm_weights(X_DIM, Y_DIM, zeroed):
    def layer():
        if zeroed:
            return np.zeros((X_DIM, Y_DIM))
        else:
            return np.random.random((X_DIM, Y_DIM)) * 0.01

    return {
        'i': layer(),
        'f': layer(),
        'o': layer(),
        'g': layer()
    }
