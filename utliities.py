def init_dict(d):
    return {k: np.zeros_like(v) for k, v in d.iteritems()} 

def normalise(vec):
    return vec / np.sum(vec)

def one_hot_vec(len, i):
    vec    = np.zeros((len, 1))
    vec[i] = 1

    return vec

def sigmoid(x, D):
# TODO: maybe replace with hard sigmoid, for efficiency
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

# Need 4 of each synapse, for the gates between the layers
def gen_lstm_syn(X_DIM, Y_DIM, zeroed):

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