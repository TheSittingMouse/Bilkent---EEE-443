
# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
import h5py



#############################################
############ QUESTION - 1 ###################
#############################################

## UTILITY FUNCTIONS THAT ARE USED IN THIS QUESTION             

def sigmoid(X):
    """Returns the sigmoid function of the array X element-wise.
    
                sigmoid(X) = 1 / (1 + exp(-X))

    Args:
        X (np.array.ndarray): Input array.

    Returns:
        float: Sigmoid of X
    """
    return 1 / (1 + np.exp(-X))

def sigmoid_backward(X):
    """The derivative of the sigmoid function for back-propagation.
    
            sigmoid_backward(X) = exp(-X) / ((1 + exp(-X))^2)

    Args:
        X (np.arrray.ndarray): Input array

    Returns:
        _type_: The derivative of sigmoid at X
    """
    return np.exp(-X) / ((1 + np.exp(-X))**2)

def KL_div_ber(P, Q):
    """Returns the KL divergence between Bernoulli random variables with means P and Q. 
    
                KL-div = (P * log(P / Q)) + ((1-P) * log((1-P) / (1-Q)))

    Args:
        P (float): Must be between 0 and 1.
        Q (float): Must be between 0 and 1.

    Returns:
        float: The KL divergence value.
    """
    return (P * np.log(P / Q)) + ((1-P) * np.log((1-P) / (1-Q)))
    
def grad_KL_div_ber(P, Q):
    """The derivative of the KL divergence of Bernoulli random variables with means P and Q, with respect to Q.

                        grad_KL_div_ber = (-P / Q) + ((1-P) / (1-Q))

    Args:
        P (float): Must be between 0 and 1.
        Q (float): Must be between 0 and 1.

    Returns:
        float: The derivative of KL divergence with respect to Q.
    """
    return (-P / Q) + ((1-P) / (1-Q))

def img_to_flat(data):
    """Takes an set of images, with sample dimention at last dimention, and flattens the images.

    Args:
        data (np.array.ndarray): Images with shape: (width, height, samples)

    Returns:
        np.array.ndarray: Flattened images with shape: (width * height, samples)
    """
    sample_size = data.shape[2]
    flat_size = data.shape[0] * data.shape[1]
    return np.reshape(data, (flat_size, sample_size))

def flat_to_img(data, img_shape):
    """Takes a set of data samples and reshapes the first dimention to the desired image shape

    Args:
        data (np.array.ndarray): Data with shape: (data, samples)
        img_shape (array): Desired image shape: (width, height)

    Raises:
        ValueError: It must hold that -> heights * width = data

    Returns:
        np.array.ndarray: Data reshaped to images of shape: (width, height, samples)
    """
    if data.shape[0] != img_shape[0] * img_shape[1]:
        raise ValueError('Need the shapes to match.')
    
    sample_size = data.shape[1]
    return np.reshape(data, (img_shape[0], img_shape[1], sample_size))

# Utility functions for creating the autoencoder network.
def init_AE_Wb(input_size, hidden_size):
    """Initializes the weights and biases for a single hidden layer autoencoder network. Uses Xavier initialization technique.

    Args:
        input_size (int): The input size, same as the output size
        hidden_size (int): The hidden layer size

    Returns:
        array: array that contains the weights and biasses: [W1, W2, b1, b2]
    """
    # Defining w0
    w0 = np.sqrt(6 / (input_size + hidden_size))
    
    # Random initialization of the parameters
    W1 = np.random.uniform(-w0, w0, (hidden_size, input_size))
    W2 = np.random.uniform(-w0, w0, (input_size, hidden_size))
    b1 = np.random.uniform(-w0, w0, (hidden_size, 1))
    b2 = np.random.uniform(-w0, w0, (input_size, 1))

    Wb = [W1, W2, b1, b2]
    
    return Wb

def aeCost(W_e, data, params):
    """Calculates the error of the autoencoder network with parameters W_e, for the given data.

    Args:
        W_e (array): Array containing the network parameters: [W1, W2, b1, b2]
        data (np.array.ndarray): The input data. Must be in shape: (L_in, samples)
        params (array): Array containing the additional information: [L_in, L_hid, cost_lambda, cost_beta, cost_rho]

    Returns:
        [float, dict]: array containg the net cost and a dictionary containing the gradients of 
         cost with respect to network parameters and activations: [J, J_grad]
    """
    
    # Unpacking the network parameters
    W1, W2, b1, b2 = W_e
    
    batch_size = data.shape[1]
    
    L_in = params[0]
    L_hid = params[1]
    cost_lambda = params[2]
    cost_beta = params[3]
    cost_rho = params[4]
    
    # Forward propagation calculations
    A0 = data
    Z1 = np.matmul(W1, A0) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    # Calculating the loss
    J_tto = (1 / (2 * batch_size)) * np.sum( (A2 - A0)**2 )
    J_Tykhonov = (cost_lambda / 2) * (np.sum( (W1)**2 ) + np.sum( (W2)**2 ))
    J_KL = cost_beta * np.sum(KL_div_ber(cost_rho, A1)) / batch_size
    J = J_tto + J_Tykhonov + J_KL
    
    # Calculating the derivatives
    dA2 = (A2 - A0) #/ batch_size
    dZ2 = dA2 * sigmoid_backward(Z2)
    dW2 = np.matmul(dZ2, A1.T) / batch_size + (cost_lambda * W2)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / batch_size
    dA1 = np.matmul(W2.T, dZ2) + (cost_beta * grad_KL_div_ber(cost_rho, A1) / batch_size)
    dZ1 = dA1 * sigmoid_backward(Z1)
    dW1 = np.matmul(dZ1, A0.T) / batch_size + (cost_lambda * W1)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / batch_size
    
    J_grad = {
        'dA2':dA2, 
        'dZ2':dZ2, 
        'dW2':dW2, 
        'db2':db2, 
        'dA1':dA1, 
        'dZ1':dZ1, 
        'dW1':dW1, 
        'db1':db1, 
    }
    return [J, J_grad]

def solver(data, W_e, params, lr, batch_size, epochs, beta1=0.9, beta2=0.999, epsilon=1e-8, verbose=True):
    """Trains the network with parameters W_e with the given data using the Adam algorithm.

    Args:
        data (np.array.ndarray): Input data to be used for training. Shape must be (L_in, samples)
        W_e (array): Array with the initial network parameters: [W1, W2, b1, b2]
        params (array): Array containing the additional information: [L_in, L_hid, cost_lambda, cost_beta, cost_rho]
        lr (float): Learning rate for training. Advised to be below 1e-3
        batch_size (int): Number of samples used per parameter updating
        epochs (int): Number of iterations over the data
        beta1 (float, optional): Beta1 value for Adam optimizer. Defaults to 0.9.
        beta2 (float, optional): Beta2 value for Adam optimizer. Defaults to 0.999.
        epsilon (float, optional): Epsilon value for Adam optimizer. Defaults to 1e-8.
        verbose (bool, optional): For verbosing the training progress. Defaults to True.

    Returns:
        [[W1, W2, b1, b2], history]: The trained network parameters and the history of loss value for progress tracking
    """
    # Unpackinng the parameters
    W1, W2, b1, b2 = W_e
    
    # Initializing the momentum and velocity for Adam
    m_W1, m_W2, m_b1, m_b2 = 0, 0, 0, 0
    v_W1, v_W2, v_b1, v_b2 = 0, 0, 0, 0
    
    # Iteration  number for Adam
    t = 0
    
    sample_number = data.shape[1]
    passes_per_epoch = sample_number // batch_size
    
    # Initializing the loss history
    loss_history = np.zeros((epochs, 1))
    
    print('Starting training...\n')
    
    for i in range(epochs):
        
        epoch_loss = 0
        
        if verbose:
            print(f'Epoch: [{i+1}/{epochs}] -> ', end='')
        
        for j in range(passes_per_epoch):
            t += 1
            
            # Calculating the cost and the gradients for the batch
            J, J_grad = aeCost(W_e, data[:, j*batch_size:(j+1)*batch_size], params)
            epoch_loss += J
            
            # Updating the moemntum values
            m_W1 = beta1 * m_W1 + (1 - beta1) * J_grad['dW1']
            m_W2 = beta1 * m_W2 + (1 - beta1) * J_grad['dW2']
            m_b1 = beta1 * m_b1 + (1 - beta1) * J_grad['db1']
            m_b2 = beta1 * m_b2 + (1 - beta1) * J_grad['db2']
            
            # Updating the velocity values
            v_W1 = beta2 * v_W1 + (1 - beta2) * (J_grad['dW1'] ** 2)
            v_W2 = beta2 * v_W2 + (1 - beta2) * (J_grad['dW2'] ** 2)
            v_b1 = beta2 * v_b1 + (1 - beta2) * (J_grad['db1'] ** 2)
            v_b2 = beta2 * v_b2 + (1 - beta2) * (J_grad['db2'] ** 2)
            
            # Normalizing the momentum
            m_W1_hat = m_W1 / (1 - beta1 ** t)
            m_W2_hat = m_W2 / (1 - beta1 ** t)
            m_b1_hat = m_b1 / (1 - beta1 ** t)
            m_b2_hat = m_b2 / (1 - beta1 ** t)
            
            # Normalizing the velocity
            v_W1_hat = v_W1 / (1 - beta2 ** t)
            v_W2_hat = v_W2 / (1 - beta2 ** t)
            v_b1_hat = v_b1 / (1 - beta2 ** t)
            v_b2_hat = v_b2 / (1 - beta2 ** t)
            
            # Updating the weights and biases
            W1 -= lr * m_W1_hat / (np.sqrt(v_W1_hat) + epsilon)
            W2 -= lr * m_W2_hat / (np.sqrt(v_W2_hat) + epsilon)
            b1 -= lr * m_b1_hat / (np.sqrt(v_b1_hat) + epsilon)
            b2 -= lr * m_b2_hat / (np.sqrt(v_b2_hat) + epsilon)
            
            # Updating W_e for the next batch of training
            W_e = [W1, W2, b1, b2]
        
        # Same training loop for the left-over samples that are left to the last batch
        if sample_number % batch_size != 0:    
            t += 1
            
            J, J_grad = aeCost(W_e, data[:, (passes_per_epoch*batch_size):], params)
            epoch_loss += J
            
            m_W1 = beta1 * m_W1 + (1 - beta1) * J_grad['dW1']
            m_W2 = beta1 * m_W2 + (1 - beta1) * J_grad['dW2']
            m_b1 = beta1 * m_b1 + (1 - beta1) * J_grad['db1']
            m_b2 = beta1 * m_b2 + (1 - beta1) * J_grad['db2']
            
            v_W1 = beta2 * v_W1 + (1 - beta2) * (J_grad['dW1'] ** 2)
            v_W2 = beta2 * v_W2 + (1 - beta2) * (J_grad['dW2'] ** 2)
            v_b1 = beta2 * v_b1 + (1 - beta2) * (J_grad['db1'] ** 2)
            v_b2 = beta2 * v_b2 + (1 - beta2) * (J_grad['db2'] ** 2)
            
            m_W1_hat = m_W1 / (1 - beta1 ** t)
            m_W2_hat = m_W2 / (1 - beta1 ** t)
            m_b1_hat = m_b1 / (1 - beta1 ** t)
            m_b2_hat = m_b2 / (1 - beta1 ** t)
            
            v_W1_hat = v_W1 / (1 - beta2 ** t)
            v_W2_hat = v_W2 / (1 - beta2 ** t)
            v_b1_hat = v_b1 / (1 - beta2 ** t)
            v_b2_hat = v_b2 / (1 - beta2 ** t)
            
            W1 -= lr * m_W1_hat / (np.sqrt(v_W1_hat) + epsilon)
            W2 -= lr * m_W2_hat / (np.sqrt(v_W2_hat) + epsilon)
            b1 -= lr * m_b1_hat / (np.sqrt(v_b1_hat) + epsilon)
            b2 -= lr * m_b2_hat / (np.sqrt(v_b2_hat) + epsilon)
            
            W_e = [W1, W2, b1, b2]
        
        epoch_loss = (epoch_loss * batch_size) / sample_number
        loss_history[i] = epoch_loss  
        
        if verbose:
                print(f'Epoch Loss: {epoch_loss:.4f}')

    print('\nTraining completed.')
    
    return [[W1, W2, b1, b2], loss_history]

def predict(W_e, data):
    """Make predictions for the data with the network with parameters W_e

    Args:
        W_e (array): Array with the initial network parameters: [W1, W2, b1, b2]
        data (np.array.ndarray): Data in shape: (L_in, samples)

    Returns:
        np.array.ndarray: The network predictions of shape: (L_out, samples)
    """
    W1, W2, b1, b2 = W_e
    
    Z1 = np.matmul(W1, data) + b1
    A1 = sigmoid(Z1)
    
    Z2 = np.matmul(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    return A2


### PART - 1.A
############################################
# extracting the data from the data1.h5 file
print('\nBegining Question-1 Part-A...\n')

with h5py.File('data1.h5', 'r') as file:
    data_key = list(file.keys())[0]
    data = np.array(file[data_key])

print('\nInitial Data Shape:', np.shape(data))
    
# Carrying the channels to last
data_channels_last = np.transpose(data, (0, 2, 3, 1))
data_channels_last_scaled = (data_channels_last - data_channels_last.min()) / ((data_channels_last - data_channels_last.min()).max())

print('Data With Channels-Last:', data_channels_last_scaled.shape)
print('Scaled Data With Channels Max:', data_channels_last_scaled.max())
print('Scaled Data With Channels Min:', data_channels_last_scaled.min())

# scaling the data for the lumousity level
data_gs = np.average(data, axis=1, weights=(0.2126, 0.7152, 0.0722))

print('\nData Shape After Lumousity Scaling:', data_gs.shape)
print('Gray-Scale Data Maximum Value:', data_gs.max())
print('Gray-Scale Minumum Value:', data_gs.min())


# Normalize the data by clipping the beyond three standart deviations
data_gs_mean = np.mean(data_gs)
data_gs_std = np.std(data_gs)

print('\nData Average Value:', data_gs_mean)
print('Data Standard Deviation:', data_gs_std)

data_clip_limit_below = data_gs_mean - 3 * data_gs_std
data_clip_limit_above = data_gs_mean + 3 * data_gs_std

data_gs_normalized = np.clip(data_gs, data_clip_limit_below, data_clip_limit_above)

print('\nClipped Data Shape:', data_gs_normalized.shape)
print('Clipped Data Maximum Value:', data_gs_normalized.max())
print('Clipped Data Minumum Value:', data_gs_normalized.min())


# Scale the data to [0.1, 0.9] interval
data_gs_norm_max = data_gs_normalized.max()
data_gs_norm_min = data_gs_normalized.min()

data_gs_scaled = (data_gs_normalized /((data_gs_norm_max - data_gs_norm_min) / 0.8)) \
- ((data_gs_normalized /((data_gs_norm_max - data_gs_norm_min) / 0.8)).min() - 0.1)

print('\nScaled Data Shape:', data_gs_scaled.shape)
print('Scaled Maximum Value:', data_gs_scaled.max())
print('Scaled Minumum Value:', data_gs_scaled.min())


# Displaying the data
sample_count = data_gs_scaled.shape[0]

# Selecting the random samples
random_samples = np.random.randint(sample_count, size=200)

rows = 10
columns = 20

# Displaying the colored samples
fig_gs = plt.figure(1, figsize=(16, 8))
plt.title('Selected Gray-Scale Data Samples')

for i in range(1, rows * columns + 1):
    fig_gs.add_subplot(rows, columns, i)
    plt.imshow(data_gs_scaled[random_samples[i-1]], cmap='gray')
    plt.axis('off')

plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.gca().set_axis_off() 
plt.savefig('Q1_A_gray_scale_images.png')  
    
# Displaying the gray-scale, normalized samples
fig_colored = plt.figure(2, figsize=(16, 8))
plt.title('Selected Colored Data Samples')

for i in range(1, rows * columns + 1):
    fig_colored.add_subplot(rows, columns, i)
    plt.imshow(data_channels_last_scaled[random_samples[i-1]])
    plt.axis('off')

plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.gca().set_axis_off() 
plt.savefig('Q1_A_colored_images.png')  

print('-'*50)

## PART - 1.B
#######################################
# Shuffling the data samples
print('\nBegining Question-1 Part-B...\n')

np.random.shuffle(data_gs_scaled)
print(f'\nData shape after shuffle: {data_gs_scaled.shape}')

# Carrying the samples dimention to last for network dimentions
data_gs_scaled = np.transpose(data_gs_scaled, (1, 2, 0))
print(f'Data shape after reshaping: {data_gs_scaled.shape}')

# Flattening the images for proper shape.
model_input_train = img_to_flat(data_gs_scaled)

# Network sizes
input_size = 256
hidden_size = 64

# Initializing the weights and biases
Wb = init_AE_Wb(input_size, hidden_size)

# Assigning the cost parameters
# Best values for cost_beta and cost_rho are 0.1 and 0.5 respectively.
cost_lambda = 5e-4
cost_beta = 0.1
cost_rho = 0.5
params = [input_size, hidden_size, cost_lambda, cost_beta, cost_rho]

# Batch size and epoch number
batch_size = 32
epochs = 50

# Training the network for the training data
Wb_trained, history = solver(model_input_train, Wb, params, 1e-3, batch_size, epochs)

# x-axis for the loss history
history_x_axis = np.arange(1, history.size+1)

# Plotting the training losses
fig_error = plt.figure(3, figsize=(5,5))
plt.title('Training Error Values')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.plot(history_x_axis, history)
plt.savefig('Q1_B_loss_history.png')  


### PART - 1.C
###################################################
# Plotting the first layer connection weights
print('\nBegining Question-1 Part-C...\n')

W1 = Wb_trained[0].transpose()

# Converting the connection weights from 1-D to 2-D array
model_W1_img = flat_to_img(W1, (16, 16))
model_W1_img = np.transpose(model_W1_img, (2, 0, 1))

figure_latent = plt.figure(figsize=(12,12))
plt.title(f'lambda = {cost_lambda}, L_hid = {hidden_size}')
rows_latent = 8
columns_latent = 8

# Plotting for every neuron in the hidden layer
for i in range(1, rows_latent * columns_latent + 1):
    figure_latent.add_subplot(rows_latent, columns_latent, i)
    plt.imshow(model_W1_img[i-1], cmap='gray')
    plt.axis('off')
    
plt.subplots_adjust(hspace=0.1, wspace=0.1)
plt.gca().set_axis_off() 
plt.savefig('Q1_C_connection_weights.png')  
        

### PART - 1.D
###################################################
print('\nBegining Question-1 Part-D...\n')

L_hid_values = [16, 49, 81]
lambda_values = [1e-5, 1e-4, 1e-3]


params = [input_size, hidden_size, cost_lambda, cost_beta, cost_rho]

for lambda_value in lambda_values:
    for L_hid_value in L_hid_values:
        print(f'\nTraining for lambda={lambda_value} | L_hid={L_hid_value}\n')
        
        params = [input_size, L_hid_value, lambda_value, cost_beta, cost_rho]
        
        Wb = init_AE_Wb(input_size, L_hid_value)
        Wb_trained, history = solver(model_input_train, Wb, params, 1e-3, batch_size, epochs)

        W1 = Wb_trained[0].transpose()

        model_W1_img = flat_to_img(W1, (16, 16))
        model_W1_img = np.transpose(model_W1_img, (2, 0, 1))

        figure_latent = plt.figure(figsize=(12,12))
        plt.title(f'lambda={lambda_value} | L_hid = {L_hid_value}')
        rows_latent = int(np.sqrt(L_hid_value))
        columns_latent = int(np.sqrt(L_hid_value))

        for i in range(1, rows_latent * columns_latent + 1):
            figure_latent.add_subplot(rows_latent, columns_latent, i)
            plt.imshow(model_W1_img[i-1], cmap='gray')
            plt.axis('off')
            
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        plt.gca().set_axis_off() 
        plt.savefig(f'Q1_D_{lambda_value}_{L_hid_value}.png')  


plt.show()