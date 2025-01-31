import numpy as np
import matplotlib.pyplot as plt
import h5py


with h5py.File('data3.h5', 'r') as file:
    trX = np.array(file['trX'])
    trY = np.array(file['trY'])
    tstX = np.array(file['tstX'])
    tstY = np.array(file['tstY'])


# data preprecessing to normalize
data_mean = trX.mean()
data_std = trX.std()

trX = (trX - data_mean) / data_std
tstX = (tstX - data_mean) / data_std



def softmax(X, axis=1):
    max_vals = np.max(X, axis=axis, keepdims=True)
    e_X = np.exp(X - max_vals)
    Y = e_X / np.sum(e_X, axis=axis, keepdims=True)
    return Y

def tanh_backward(X):
    return 1 - np.tanh(X)**2


def plot_confusion_matrix(predictions, true_labels, class_names=None):
    """
    Plots a confusion matrix for the given predictions and true labels without sklearn.

    Parameters:
    predictions (numpy.ndarray): Model predictions of shape (N, num_classes) (softmax outputs or similar).
    true_labels (numpy.ndarray): True labels of shape (N, num_classes) (one-hot encoded).
    class_names (array, optional): The names of the classes. If not provided, integers will be assigned.

    Returns:
    (matplotlib.figure.Figure): The confusion matrix plot.
    """
    # Convert one-hot encoded labels to integer labels
    num_classes = true_labels.shape[1]
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Populate confusion matrix
    for t, p in zip(true_classes, pred_classes):
        confusion_matrix[t, p] += 1
        
    # Plot confusion matrix
    plot = plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, cmap='viridis')
    plt.colorbar()
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    
    if class_names == None:
        plt.xticks(ticks=np.arange(num_classes), labels=[f"Class {i+1}" for i in range(num_classes)])
        plt.yticks(ticks=np.arange(num_classes), labels=[f"Class {i+1}" for i in range(num_classes)])
    else:
        plt.xticks(ticks=np.arange(num_classes), labels=class_names)
        plt.yticks(ticks=np.arange(num_classes), labels=class_names)
        

    # Add numbers to each cell
    for i in range(num_classes):
        for j in range(num_classes):
            value = confusion_matrix[i, j]
            max_value = np.max(confusion_matrix) / 2

            if value > max_value:
                text_color = 'white'
            else:
                text_color = 'black'

            plt.text(j, i, value, ha='center', va='center', color=text_color)

    return plot




def initRecWb(input_size, hidden_size):
    w0_ih = np.sqrt(6 / (input_size + hidden_size))
    W_ih = np.random.uniform(-w0_ih, w0_ih, (input_size, hidden_size))
    
    w0_hh = np.sqrt(6 / (2 * hidden_size))
    W_hh = np.random.uniform(-w0_hh, w0_hh, (hidden_size, hidden_size))
    
    b_ih = np.zeros((1, hidden_size))
    
    WbRec = {'W_hh': W_hh, 'W_ih': W_ih, 'b_ih': b_ih}
    return WbRec

def initSeqWb(sizes):
    Wb = {}
    
    for i in range(len(sizes) - 1):
            w_0 = np.sqrt(6 / (sizes[i] + sizes[i+1]))
            Wb['W' + str(i)] = np.random.uniform(-w_0, w_0, (sizes[i], sizes[i+1]))
            Wb['b' + str(i)] = np.zeros((1, sizes[i + 1]))
    
    return Wb      

def reccurentForward(WbRec, data):
    batch_size, time_steps, features = data.shape
    
    h_t = np.zeros((batch_size, WbRec['W_hh'].shape[0]))
    
    recCache = {}
    
    for t in range(time_steps):
        x_t = data[:, t, :]
                
        Z = np.matmul(x_t, WbRec['W_ih']) + np.matmul(h_t, WbRec['W_hh']) + WbRec['b_ih']
        h_t = np.tanh(Z)
        
        recCache['Zrec_' + str(t)] = Z
        recCache['H_' + str(t)] = h_t
        
    return [h_t, recCache]


def sequantialForward(Wb, data):    
    num_layers = len(Wb) // 2
    
    cache = {}
    
    A = data
    
    cache['A-1'] = data
    cache['Z-1'] = np.zeros_like(data)
    
    for layer in range(num_layers - 1):
        Z = np.matmul(A, Wb['W' + str(layer)]) + Wb['b' + str(layer)]
        A = np.tanh(Z)
        
        cache['Z' + str(layer)] = Z
        cache['A' + str(layer)] = A
        
    Z = np.matmul(A, Wb['W' + str(num_layers - 1)]) + Wb['b' + str(num_layers - 1)]
    A = softmax(Z, axis=1)
    
    cache['Z' + str(num_layers - 1)] = Z
    cache['A' + str(num_layers - 1)] = A
    
    return [A, cache]
    

def sequentialBackward(Wb, labels, cache):
    
    batch_size = labels.shape[0]
    
    num_layers = len(Wb) // 2
    # _, cache = sequantialForward(Wb, recLastState)
    
    lastActivation = cache['A' + str(num_layers-1)]
    
    J = -np.sum(labels * np.log(lastActivation)) / batch_size
    grads = {}
    
    A_prev = cache['A' + str(num_layers-1)]
    dZ = lastActivation - labels
    dA_prev = 0
    
    for layer in reversed(range(num_layers)):
        A_prev = cache['A' + str(layer-1)]
        grads['db' + str(layer)] = np.sum(dZ, axis=0, keepdims=True) / batch_size
        grads['dW' + str(layer)] = np.matmul(A_prev.T, dZ) / batch_size
        dA_prev = np.matmul(dZ, Wb['W' + str(layer)].T)
        dZ = dA_prev * tanh_backward(cache['Z' + str(layer-1)])
            
    return [J, grads, dA_prev]
            
def recBackward(WbRec, input, last_dA, recCache, max_lookback_distance=None):
    """
    Backpropagation through time for a single-layer RNN.

    Parameters
    ----------
    WbRec : dict
        Dictionary containing the recurrent layer parameters:
          - 'W_ih': Input-to-hidden weights, shape (D_in, D_h)
          - 'W_hh': Hidden-to-hidden weights, shape (D_h, D_h)
          - 'b_ih': Bias for hidden state, shape (D_h,)
    last_dA : np.ndarray
        The gradient of the loss w.r.t. the last hidden state, shape (N, D_h).
    recCache : dict
        Cache from the forward pass containing:
          - 'X_t', 'Zrec_t', 'H_t' for t=1,...,T
        Must contain all time steps that were used in forward propagation.
    max_lookback_distance : int or None
        The number of steps to backprop through time.
        If None or greater than the total number of steps, all steps are used.

    Returns
    -------
    grads : dict
        Dictionary containing:
          - 'dW_ih'
          - 'dW_hh'
          - 'db_ih'
    """
    
    batch_size = input.shape[0]

    W_ih = WbRec['W_ih']  # (D_in, D_h)
    W_hh = WbRec['W_hh']  # (D_h, D_h)
    b_ih = WbRec['b_ih']  # (D_h,)

    # Determine the number of timesteps T from the recCache keys
    # We assume keys are like 'H_1', 'H_2', ... 'H_T'
    # Extract the max t by checking keys:
    timesteps = [int(k.split('_')[1]) for k in recCache.keys() if k.startswith('H_')]
    T = max(timesteps) if len(timesteps) > 0 else 0

    # If max_lookback_distance not provided or too large, use the full length
    if max_lookback_distance is None or max_lookback_distance > T:
        max_lookback_distance = T

    # Initialize gradients
    dW_ih = np.zeros_like(W_ih)
    dW_hh = np.zeros_like(W_hh)
    db_ih = np.zeros_like(b_ih)

    # The hidden state gradient at the final step
    dH_next = last_dA  # shape (N, D_h)

    # Backprop through time
    for t in range(T, T - max_lookback_distance, -1):
        # Load cached values
        H_t = recCache[f'H_{t}']  # shape (N, D_h)
        Z_t = recCache[f'Zrec_{t}']  # shape (N, D_h)
        # X_t = recCache[f'X_{t}']  # shape (N, D_in)
        X_t = input[:, t, :]

        # For H_{t-1}, if t=1, we might have H_0 in cache or use zeros
        if t == 1:
            H_prev = recCache.get('H_0', np.zeros((X_t.shape[0], H_t.shape[1])))
        else:
            H_prev = recCache[f'H_{t-1}']

        # Compute dZ_t = dH_t * (1 - H_t^2) because H_t = tanh(Z_t)
        # dH_t is currently dH_next
        # dZ_t = dH_next * (1 - H_t**2)  # shape (N, D_h)
        dZ_t = dH_next * tanh_backward(Z_t)

        # Compute gradients for parameters
        # dW_ih: sum over batch of X_t^T dZ_t
        dW_ih += np.matmul(X_t.T, dZ_t) / batch_size # (D_in, N) @ (N, D_h) = (D_in, D_h)
        # dW_hh: sum over batch of H_{t-1}^T dZ_t
        dW_hh += np.matmul(H_prev.T, dZ_t) / batch_size  # (D_h, N) @ (N, D_h) = (D_h, D_h)
        # db_ih: sum over batch dimension
        db_ih += dZ_t.sum(axis=0) / batch_size  # (D_h,)

        # Backprop into H_{t-1}
        dH_prev = np.matmul(dZ_t, W_hh.T)  # (N, D_h) @ (D_h, D_h) = (N, D_h)

        # Update dH_next for the next iteration
        dH_next = dH_prev

    recGrads = {
        'dW_ih': dW_ih,
        'dW_hh': dW_hh,
        'db_ih': db_ih
    }

    return recGrads

###########################
#### LSTM CODE ##########
#####################

def initLSTMWb(input_size, hidden_size):
    WbLSTM = {}
    w0_ih = np.sqrt(6 / (input_size + hidden_size))
    w0_hh = np.sqrt(6 / (2 * hidden_size))
    
    ih_names = ['W_f', 'W_i', 'W_o', 'W_c']
    hh_names = ['U_f', 'U_i', 'U_o', 'U_c']
    b_names = ['b_f', 'b_i', 'b_o', 'b_c']
    
    WbLSTM['W_f'] = np.random.uniform(-w0_ih, w0_ih, (input_size, hidden_size)),
    
    for name in ih_names:
        WbLSTM[name] = np.random.uniform(-w0_ih, w0_ih, (input_size, hidden_size))
    for name in hh_names:
        WbLSTM[name] = np.random.uniform(-w0_hh, w0_hh, (hidden_size, hidden_size))
    for name in b_names:
        WbLSTM[name] = np.zeros((1, hidden_size))
        
    return WbLSTM


def lstmForward(WbLSTM, data):
    batch_size, time_steps, features = data.shape

    W_f = WbLSTM['W_f']
    W_i = WbLSTM['W_i']
    W_o = WbLSTM['W_o']
    W_c = WbLSTM['W_c']
  
    U_f = WbLSTM['U_f']
    U_i = WbLSTM['U_i']
    U_o = WbLSTM['U_o']
    U_c = WbLSTM['U_c']
  
    b_f = WbLSTM['b_f']
    b_i = WbLSTM['b_i']
    b_o = WbLSTM['b_o']
    b_c = WbLSTM['b_c']
    

    # Initialize hidden state (h_t) and cell state (c_t) to zeros
    hidden_size = W_f.shape[1]
    h_t = np.zeros((batch_size, hidden_size))
    c_t = np.zeros((batch_size, hidden_size))

    lstmCache = {}

    for t in range(time_steps):
        x_t = data[:, t, :]

        # Compute gates
        f_t = 1 / (1 + np.exp(-(np.matmul(x_t, W_f) + np.matmul(h_t, U_f) + b_f)))  # Forget gate
        i_t = 1 / (1 + np.exp(-(np.matmul(x_t, W_i) + np.matmul(h_t, U_i) + b_i)))  # Input gate
        o_t = 1 / (1 + np.exp(-(np.matmul(x_t, W_o) + np.matmul(h_t, U_o) + b_o)))  # Output gate
        g_t = np.tanh(np.matmul(x_t, W_c) + np.matmul(h_t, U_c) + b_c)              # Candidate cell state

        # Update cell state and hidden state
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * np.tanh(c_t)

        # Save intermediate values for backpropagation or debugging
        lstmCache['f_t_' + str(t)] = f_t
        lstmCache['i_t_' + str(t)] = i_t
        lstmCache['o_t_' + str(t)] = o_t
        lstmCache['g_t_' + str(t)] = g_t
        lstmCache['c_t_' + str(t)] = c_t
        lstmCache['h_t_' + str(t)] = h_t

    return [h_t, lstmCache]

def lstmBackward(WbLSTM, data, last_dA, lstmCache, max_lookback_distance=None):
    """
    Backpropagation through time for an LSTM layer.

    Parameters
    ----------
    WbLSTM : dict
        Dictionary containing the LSTM layer parameters:
          - 'W_f', 'W_i', 'W_o', 'W_c': Input-to-hidden weights, each shape (D_in, D_h)
          - 'U_f', 'U_i', 'U_o', 'U_c': Hidden-to-hidden weights, each shape (D_h, D_h)
          - 'b_f', 'b_i', 'b_o', 'b_c': Biases for gates, each shape (D_h,)
    data : np.ndarray
        Input data to the LSTM, shape (N, T, D_in)
    last_dA : np.ndarray
        Gradient of the loss w.r.t. the last hidden state, shape (N, D_h)
    lstmCache : dict
        Cache from the forward pass containing:
          - 'f_t_t', 'i_t_t', 'o_t_t', 'g_t_t', 'c_t_t', 'h_t_t' for t=1,...,T
        Must contain all time steps that were used in forward propagation.
    max_lookback_distance : int or None
        The number of steps to backprop through time.
        If None or greater than the total number of steps, all steps are used.

    Returns
    -------
    grads : dict
        Dictionary containing gradients for:
          - 'dW_f', 'dW_i', 'dW_o', 'dW_c'
          - 'dU_f', 'dU_i', 'dU_o', 'dU_c'
          - 'db_f', 'db_i', 'db_o', 'db_c'
    """
    N, T, D_in = data.shape
    hidden_size = WbLSTM['W_f'].shape[1]

    # Initialize gradients for weights, biases, and recurrent connections
    dW_f = np.zeros_like(WbLSTM['W_f'])
    dW_i = np.zeros_like(WbLSTM['W_i'])
    dW_o = np.zeros_like(WbLSTM['W_o'])
    dW_c = np.zeros_like(WbLSTM['W_c'])

    dU_f = np.zeros_like(WbLSTM['U_f'])
    dU_i = np.zeros_like(WbLSTM['U_i'])
    dU_o = np.zeros_like(WbLSTM['U_o'])
    dU_c = np.zeros_like(WbLSTM['U_c'])

    db_f = np.zeros_like(WbLSTM['b_f'])
    db_i = np.zeros_like(WbLSTM['b_i'])
    db_o = np.zeros_like(WbLSTM['b_o'])
    db_c = np.zeros_like(WbLSTM['b_c'])

    # Initialize gradients w.r.t. hidden and cell states
    dH_next = last_dA
    dC_next = np.zeros((N, hidden_size))

    # If max_lookback_distance is not provided, use the full sequence length
    if max_lookback_distance is None or max_lookback_distance > T:
        max_lookback_distance = T

    # Backprop through time
    for t in range(T - 1, T - max_lookback_distance - 1, -1):
        # Load cached values for time step t
        f_t = lstmCache[f'f_t_{t}']
        i_t = lstmCache[f'i_t_{t}']
        o_t = lstmCache[f'o_t_{t}']
        g_t = lstmCache[f'g_t_{t}']
        c_t = lstmCache[f'c_t_{t}']
        h_t = lstmCache[f'h_t_{t}']
        x_t = data[:, t, :]
        h_prev = lstmCache[f'h_t_{t-1}'] if t > 0 else np.zeros_like(h_t)
        c_prev = lstmCache[f'c_t_{t-1}'] if t > 0 else np.zeros_like(c_t)

        # Gradients w.r.t. cell state and output gate
        dO_t = dH_next * np.tanh(c_t)  # Gradient of output gate
        dC_t = dH_next * o_t * (1 - np.tanh(c_t)**2) + dC_next  # Gradient of cell state

        # Gradients w.r.t. gates
        dF_t = dC_t * c_prev
        dI_t = dC_t * g_t
        dG_t = dC_t * i_t

        # Apply activation function derivatives
        dF_t *= f_t * (1 - f_t)  # Sigmoid derivative
        dI_t *= i_t * (1 - i_t)  # Sigmoid derivative
        dO_t *= o_t * (1 - o_t)  # Sigmoid derivative
        dG_t *= 1 - g_t**2       # Tanh derivative

        # Accumulate parameter gradients
        dW_f += np.matmul(x_t.T, dF_t) / N
        dW_i += np.matmul(x_t.T, dI_t) / N
        dW_o += np.matmul(x_t.T, dO_t) / N
        dW_c += np.matmul(x_t.T, dG_t) / N

        dU_f += np.matmul(h_prev.T, dF_t) / N
        dU_i += np.matmul(h_prev.T, dI_t) / N
        dU_o += np.matmul(h_prev.T, dO_t) / N
        dU_c += np.matmul(h_prev.T, dG_t) / N

        db_f += dF_t.sum(axis=0) / N
        db_i += dI_t.sum(axis=0) / N
        db_o += dO_t.sum(axis=0) / N
        db_c += dG_t.sum(axis=0) / N

        # Backpropagate into previous hidden and cell states
        dH_next = np.matmul(dF_t, WbLSTM['U_f'].T) + \
                  np.matmul(dI_t, WbLSTM['U_i'].T) + \
                  np.matmul(dO_t, WbLSTM['U_o'].T) + \
                  np.matmul(dG_t, WbLSTM['U_c'].T)

        dC_next = dC_t * f_t

    # Package gradients into a dictionary
    lstmGrads = {
        'dW_f': dW_f, 'dW_i': dW_i, 'dW_o': dW_o, 'dW_c': dW_c,
        'dU_f': dU_f, 'dU_i': dU_i, 'dU_o': dU_o, 'dU_c': dU_c,
        'db_f': db_f, 'db_i': db_i, 'db_o': db_o, 'db_c': db_c
    }

    return lstmGrads


########
# GRU CODE
###################

def initGRUWb(input_size, hidden_size):
    WbGRU = {}
    
    w0_hh = np.sqrt(6 / (2 * hidden_size))
    w0_ih = np.sqrt(6 / (input_size + hidden_size))
    
    WbGRU['W_z'] = np.random.uniform(-w0_ih, w0_ih, (input_size, hidden_size))
    WbGRU['W_r'] = np.random.uniform(-w0_ih, w0_ih, (input_size, hidden_size))
    WbGRU['W_h'] = np.random.uniform(-w0_ih, w0_ih, (input_size, hidden_size))
    
    WbGRU['U_z'] = np.random.uniform(-w0_hh, w0_hh, (hidden_size, hidden_size))
    WbGRU['U_r'] = np.random.uniform(-w0_hh, w0_hh, (hidden_size, hidden_size))
    WbGRU['U_h'] = np.random.uniform(-w0_hh, w0_hh, (hidden_size, hidden_size))
    
    WbGRU['b_z'] = np.zeros((1, hidden_size))
    WbGRU['b_r'] = np.zeros((1, hidden_size))
    WbGRU['b_h'] = np.zeros((1, hidden_size))
    
    return WbGRU
    
def gruForward(WbGRU, data):
    batch_size, time_steps, features = data.shape

    W_z = WbGRU['W_z']
    W_r = WbGRU['W_r']
    W_h = WbGRU['W_h']
  
    U_z = WbGRU['U_z']
    U_r = WbGRU['U_r']
    U_h = WbGRU['U_h']
  
    b_z = WbGRU['b_z']
    b_r = WbGRU['b_r']
    b_h = WbGRU['b_h']
    
    # Initialize hidden state (h_t) to zeros
    hidden_size = W_z.shape[1]
    h_t = np.zeros((batch_size, hidden_size))

    gruCache = {}

    for t in range(time_steps):
        x_t = data[:, t, :]

        # Compute gates
        z_t = 1 / (1 + np.exp(-(np.matmul(x_t, W_z) + np.matmul(h_t, U_z) + b_z)))  # Update gate
        r_t = 1 / (1 + np.exp(-(np.matmul(x_t, W_r) + np.matmul(h_t, U_r) + b_r)))  # Reset gate
        h_hat_t = np.tanh(np.matmul(x_t, W_h) + np.matmul(r_t * h_t, U_h) + b_h)    # Candidate hidden state
        
        # Update hidden state
        h_t = z_t * h_t + (1 - z_t) * h_hat_t

        # Save intermediate values for backpropagation or debugging
        gruCache['z_t_' + str(t)] = z_t
        gruCache['r_t_' + str(t)] = r_t
        gruCache['h_hat_t_' + str(t)] = h_hat_t
        gruCache['h_t_' + str(t)] = h_t

    return [h_t, gruCache]


def gruBackward(WbGRU, data, last_dA, gruCache, max_lookback_distance=None):
    """
    Backpropagation through time for a GRU layer.

    Parameters
    ----------
    WbGRU : dict
        Dictionary containing the GRU layer parameters:
          - 'W_z', 'W_r', 'W_h': Input-to-hidden weights, each shape (D_in, D_h)
          - 'U_z', 'U_r', 'U_h': Hidden-to-hidden weights, each shape (D_h, D_h)
          - 'b_z', 'b_r', 'b_h': Biases for gates, each shape (D_h,)
    data : np.ndarray
        Input data to the GRU, shape (N, T, D_in)
    last_dA : np.ndarray
        Gradient of the loss w.r.t. the last hidden state, shape (N, D_h)
    gruCache : dict
        Cache from the forward pass containing:
          - 'z_t_t', 'r_t_t', 'h_hat_t_t', 'h_t_t' for t=1,...,T
        Must contain all time steps that were used in forward propagation.
    max_lookback_distance : int or None
        The number of steps to backprop through time.
        If None or greater than the total number of steps, all steps are used.

    Returns
    -------
    grads : dict
        Dictionary containing gradients for:
          - 'dW_z', 'dW_r', 'dW_h'
          - 'dU_z', 'dU_r', 'dU_h'
          - 'db_z', 'db_r', 'db_h'
    """
    N, T, D_in = data.shape
    hidden_size = WbGRU['W_z'].shape[1]

    # Initialize gradients for weights, biases, and recurrent connections
    dW_z = np.zeros_like(WbGRU['W_z'])
    dW_r = np.zeros_like(WbGRU['W_r'])
    dW_h = np.zeros_like(WbGRU['W_h'])

    dU_z = np.zeros_like(WbGRU['U_z'])
    dU_r = np.zeros_like(WbGRU['U_r'])
    dU_h = np.zeros_like(WbGRU['U_h'])

    db_z = np.zeros_like(WbGRU['b_z'])
    db_r = np.zeros_like(WbGRU['b_r'])
    db_h = np.zeros_like(WbGRU['b_h'])

    # Initialize gradients w.r.t. hidden state
    dH_next = last_dA

    # If max_lookback_distance is not provided, use the full sequence length
    if max_lookback_distance is None or max_lookback_distance > T:
        max_lookback_distance = T

    # Backprop through time
    for t in range(T - 1, T - max_lookback_distance - 1, -1):
        # Load cached values for time step t
        z_t = gruCache[f'z_t_{t}']
        r_t = gruCache[f'r_t_{t}']
        h_hat_t = gruCache[f'h_hat_t_{t}']
        h_t = gruCache[f'h_t_{t}']
        x_t = data[:, t, :]
        h_prev = gruCache[f'h_t_{t-1}'] if t > 0 else np.zeros_like(h_t)

        # Gradients w.r.t. hidden state
        dZ_t = dH_next * (h_prev - h_hat_t)
        dH_hat_t = dH_next * (1 - z_t)
        dH_prev = dH_next * z_t

        # Apply activation function derivatives
        dZ_t *= z_t * (1 - z_t)  # Sigmoid derivative
        dR_t = np.matmul(dH_hat_t, WbGRU['U_h'].T) * h_prev
        dR_t *= r_t * (1 - r_t)  # Sigmoid derivative
        dH_hat_t *= 1 - h_hat_t**2  # Tanh derivative

        # Accumulate parameter gradients
        dW_z += np.matmul(x_t.T, dZ_t) / N
        dW_r += np.matmul(x_t.T, dR_t) / N
        dW_h += np.matmul(x_t.T, dH_hat_t) / N

        dU_z += np.matmul(h_prev.T, dZ_t) / N
        dU_r += np.matmul(h_prev.T, dR_t) / N
        dU_h += np.matmul((r_t * h_prev).T, dH_hat_t) / N

        db_z += dZ_t.sum(axis=0) / N
        db_r += dR_t.sum(axis=0) / N
        db_h += dH_hat_t.sum(axis=0) / N

        # Backpropagate into previous hidden state
        dH_next = dH_prev + np.matmul(dZ_t, WbGRU['U_z'].T) + np.matmul(dR_t, WbGRU['U_r'].T)

    # Package gradients into a dictionary
    gruGrads = {
        'dW_z': dW_z, 'dW_r': dW_r, 'dW_h': dW_h,
        'dU_z': dU_z, 'dU_r': dU_r, 'dU_h': dU_h,
        'db_z': db_z, 'db_r': db_r, 'db_h': db_h
    }

    return gruGrads


###################
### SGD CODE AND UTILS
##############


def SGD_Q3(WbRec, Wb, data, labels, val_data, val_labels, lr, batch_size,
           stop_loss=0.1, max_epochs=50, alpha=0, verbose=True, recurrent_type='Simple', max_lookback_distance=None):
    
    sample_number = data.shape[0]
    
    recFuncForward = reccurentForward
    recFuncBackward = recBackward
    
    if recurrent_type == 'LSTM' or recurrent_type == 'lstm':
        recFuncForward = lstmForward
        recFuncBackward = lstmBackward
    elif recurrent_type == 'GRU' or recurrent_type == 'gru':
        recFuncForward = gruForward
        recFuncBackward = gruBackward

    
    dParams = {}
    for param in Wb:
            dParams['d' + param] = np.zeros_like(Wb[param])
    for recParam in WbRec:
        dParams['d' + recParam] = np.zeros_like(WbRec[recParam])
        
        
    passes_per_epoch = sample_number // batch_size
    
    train_loss_history = np.zeros((max_epochs, 1))
    val_loss_history = np.zeros((max_epochs, 1))
    train_accuracy_history = np.zeros((max_epochs, 1))
    val_accuracy_history = np.zeros((max_epochs, 1))
    
    J_train = 0
    J_val = 0
    val_acc_percent = 0
    train_acc_percent = 0
    
    print('Starting training...')
    print('Recurrency type:', recurrent_type)
    print(f'Target Validation Loss: {stop_loss} | Maximum Number of Epochs: {max_epochs}.\n')
    
    epoch = 0
    J_val = stop_loss + 1
    
    while J_val > stop_loss and epoch != max_epochs:
        
        perm = np.random.permutation(sample_number)
        data = data[perm, :]
        labels = labels[perm, :]
        
        J_train = 0
        J_val = 0
        
        correct_labels = 0
        total_labels = sample_number
        epoch += 1
        
        if verbose:
            print(f'Epoch: [{epoch}/{max_epochs}] -> ', end='')
        
        for batch in range(passes_per_epoch):
            
            batch_data = batch*batch_size
            
            h_t, recCache = recFuncForward(WbRec, data[batch_data:batch_data + batch_size, :])
            A, cache = sequantialForward(Wb, h_t)
            
            J, grads, last_dA = sequentialBackward(Wb, labels[batch_data:batch_data + batch_size, :], cache)
            recGrads = recFuncBackward(WbRec, data[batch_data:batch_data + batch_size, :], last_dA, recCache, max_lookback_distance)
            
            
            J_train += J
            pred_labels = np.argmax(A, axis=1)
            true_labels = np.argmax(labels[batch_data:batch_data + batch_size, :], axis=1)
            correct_labels += np.where(pred_labels == true_labels, 1, 0).sum()
        
                        
            for d_param in grads:
                dParams[d_param] = alpha * dParams[d_param] - lr * grads[d_param]
            for d_param in recGrads:
                dParams[d_param] = alpha * dParams[d_param] - lr * recGrads[d_param]
                
            
            # gradient clipping to prevent exploding gradients
            # total_norm = np.sqrt(sum(np.sum(dParam**2) for dParam in dParams.values()))
            # clip_coef = grad_clip / (total_norm + 1e-6)
    
            # if total_norm > grad_clip:
            #     for key in recGrads:
            #     # for key in dParams:
            #         dParams[key] *= clip_coef
                    
            
            for param in Wb:
                Wb[param] += dParams['d' + param]
            for param in WbRec:
                WbRec[param] += dParams['d' + param]
                


        if sample_number % batch_size != 0:
            
            batch_data = passes_per_epoch*batch_size

            h_t, recCache = recFuncForward(WbRec, data[batch_data:, :])
            A, cache = sequantialForward(Wb, h_t)
            
            J, grads, last_dA = sequentialBackward(Wb, labels[batch_data:, :], cache)
            recGrads = recFuncBackward(WbRec, data[batch_data:, :], last_dA, recCache, max_lookback_distance)
            
            
            J_train += J
            pred_labels = np.argmax(A, axis=1)
            true_labels = np.argmax(labels[batch_data:batch_data + batch_size, :], axis=1)
            correct_labels += np.where(pred_labels == true_labels, 1, 0).sum()
            
                        
            for d_param in grads:
                dParams[d_param] = alpha * dParams[d_param] - lr * grads[d_param]
            for d_param in recGrads:
                dParams[d_param] = alpha * dParams[d_param] - lr * recGrads[d_param]
                
            
            # # gradient clipping to prevent exploding gradients
            # total_norm = np.sqrt(sum(np.sum(dParam**2) for dParam in dParams.values()))
            # clip_coef = grad_clip / (total_norm + 1e-6)
    
            # if total_norm > grad_clip:
            #     for key in recGrads:
            #     # for key in dParams:
            #         dParams[key] *= clip_coef
            
            
            for param in Wb:
                Wb[param] += dParams['d' + param]
            for param in WbRec:
                WbRec[param] += dParams['d' + param]
                
        
        h_t, recCache = recFuncForward(WbRec, val_data)
        A, cache = sequantialForward(Wb, h_t)
        
        J_val, _, _ = sequentialBackward(Wb, val_labels, cache)
        val_pred_labels = np.argmax(A, axis=1)
        val_true_labels = np.argmax(val_labels, axis=1)
        val_correct_labels = np.where(val_pred_labels == val_true_labels, 1, 0).sum()
        val_total_labels = val_labels.shape[0]
        
        train_acc_percent = (correct_labels / total_labels) * 100
        val_acc_percent = (val_correct_labels / val_total_labels) * 100
        J_train = J_train * batch_size / sample_number
        
        train_loss_history[epoch-1] = J_train
        val_loss_history[epoch-1] = J_val
        train_accuracy_history[epoch-1] = train_acc_percent
        val_accuracy_history[epoch-1] = val_acc_percent
                    
        if verbose:
            print(f'Training Cost: {J_train:.4f} | Training Accuracy: {train_acc_percent:.2f}% | Validation Cost: {J_val:.4f} | Validation Accuracy: {val_acc_percent:.2f}%')
        
    train_loss_history[epoch:] = J_train
    val_loss_history[epoch:] = J_val
    train_accuracy_history[epoch:] = train_acc_percent
    val_accuracy_history[epoch:] = val_acc_percent

    metrics = {'train_loss':train_loss_history, 'val_loss':val_loss_history,
               'train_accuracy':train_accuracy_history, 'val_accuracy':val_accuracy_history}
     
    print('\nTraining completed.')
    
    return [WbRec, Wb, metrics]


def get_accuracy(WbRec, Wb, testing_data, testing_labels, recurrency_type='Simple'):
    
    Forward = reccurentForward
    
    if recurrency_type == 'Simple' or recurrency_type == 'simple':
        Forward = reccurentForward
    elif recurrency_type == 'LSTM' or recurrency_type == 'lstm':
        Forward = lstmForward
    elif recurrency_type == 'GRU' or recurrency_type == 'gru':
        Forward = gruForward
 
    h_t, recCache = Forward(WbRec, testing_data)
    A, cache = sequantialForward(Wb, h_t)

    pred_labels = np.argmax(A, axis=1)
    true_labels = np.argmax(testing_labels, axis=1)

    correct = np.sum(np.where(true_labels == pred_labels, 1, 0))
    total = testing_data.shape[0]
    
    return [correct / total, A]


def initWbQ3(layer_sizes, recurrency_type):
    if recurrency_type == 'Simple' or recurrency_type == 'simple':
        initFuncWb = initRecWb
    elif recurrency_type == 'LSTM' or recurrency_type == 'lstm':
        initFuncWb = initLSTMWb
    elif recurrency_type == 'GRU' or recurrency_type == 'gru':
        initFuncWb = initGRUWb

    WbRec = initFuncWb(layer_sizes[0], layer_sizes[1])
    WbSeq = initSeqWb(layer_sizes[1:])
    
    return [WbRec, WbSeq]



data_count = trX.shape[0]

val_train_split = 0.1
train_start_index = int(data_count * val_train_split)

rand_perm = np.random.permutation(data_count)
trX = trX[rand_perm]
trY = trY[rand_perm]

data_train = trX[train_start_index:]
data_val = trX[:train_start_index]

labels_train = trY[train_start_index:]
labels_val = trY[:train_start_index]

print('Training data count:', data_train.shape[0])
print('Validation data count:', data_val.shape[0], '\n')

recurrent_types = ['Simple', 'LSTM', 'GRU']
max_lookback_list = {'Simple':None, 'LSTM':None, 'GRU':None}
layer_sizes = [3, 128, 64, 64, 6]
class_names = ['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking']


learning_rate = 0.01
batch_size = 32
max_epochs = 50
stop_loss = 0.5
alpha = 0.85
epochs = np.arange(1, max_epochs+1, 1)


for rec_type in recurrent_types:
    WbRec, Wb = initWbQ3(layer_sizes, rec_type)
    
    trainedWbRec, trainedWb, loss_hist = SGD_Q3(WbRec, Wb,
                                         data_train, labels_train,
                                         data_val, labels_val, 
                                         lr=learning_rate, batch_size=batch_size, stop_loss=stop_loss,
                                         max_epochs=max_epochs, alpha=alpha,
                                         max_lookback_distance=max_lookback_list[rec_type],
                                         recurrent_type=rec_type)
    
    test_accuracy, test_predictions = get_accuracy(WbRec, Wb, tstX, tstY, rec_type)
    conf_mat = plot_confusion_matrix(test_predictions, tstY, class_names)
    plt.title(F'Confusion Matrix Over Test Set for {rec_type} Recurrent Cell')
    plt.savefig(f'conf_mat_{rec_type}_test.png')
    
    train_accuracy, train_predictions = get_accuracy(WbRec, Wb, data_train, labels_train, rec_type)
    conf_mat = plot_confusion_matrix(train_predictions, labels_train, class_names)
    plt.title(F'Confusion Matrix Over Training Set for {rec_type} Recurrent Cell')
    plt.savefig(f'conf_mat_{rec_type}_train.png')
    
        
    metrics_figure = plt.figure(figsize=(15,10))
    
    metrics_figure.add_subplot(2,2,1)
    plt.plot(epochs, loss_hist['train_loss'], 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Training Cost')
    
    metrics_figure.add_subplot(2,2,2)
    plt.plot(epochs, loss_hist['val_loss'], 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Validadtion Cost')
    
    metrics_figure.add_subplot(2,2,3)
    plt.plot(epochs, loss_hist['train_accuracy'], 'b')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy (%)')
    
    metrics_figure.add_subplot(2,2,4)
    plt.plot(epochs, loss_hist['val_accuracy'], 'b')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy (%)')
    
    plt.suptitle(f"Training Metrics for {rec_type} Recurrent Cell\nTest Accuracy: {100 * test_accuracy:.2f}%")
    plt.savefig(f'loss_and_acc_{rec_type}')
    
    print(f'Test accuracy for {rec_type} recurrent cell: {100 * test_accuracy:.2f}%\n')
    
    
plt.show()

