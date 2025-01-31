import numpy as np
import matplotlib.pyplot as plt
import h5py


def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_backward(X):
    return np.exp(-X) / ((1 + np.exp(-X))**2)

def relu(X, alpha=0.01):
    return np.where(X>0, X, alpha * X)

def relu_backward(X, alpha=0.01):
    return np.where(X>0, 1, alpha)

def softmax(X, axis=0):
    max_vals = np.max(X, axis=axis, keepdims=True)
    e_X = np.exp(X - max_vals)
    Y = e_X / np.sum(e_X, axis=axis, keepdims=True)
    return Y



def init_NLP_Wb(dict_size, embed_size, hidden_size, std=0.01):
    
    W_embed = np.random.normal(0, std, (dict_size, embed_size))
    
    W1 = np.random.normal(0, std, (embed_size, hidden_size))
    W2 = np.random.normal(0, std, (hidden_size, dict_size))

    b1 = np.random.normal(0, std, (1, hidden_size))
    b2 = np.random.normal(0, std, (1, dict_size))
    
    params = {
        'W_embed': W_embed,
        'W1': W1,
        'W2': W2,
        'b1': b1,
        'b2': b2
    }

    return params
    
def NLP_forward_pass(params, data, dict_size):
    W_embed = params['W_embed']
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']
    
    batch_size, context_size = data.shape
    data = np.eye(dict_size)[data]
    
    A_pre = np.sum(data, axis=1) / context_size

    A0 = np.matmul(A_pre, W_embed)
    
    Z1 = np.matmul(A0, W1) + b1
    A1 = sigmoid(Z1)
    
    Z2 = np.matmul(A1, W2) + b2
    A2 = softmax(Z2, axis=1)
    
    cache = {
        'A_pre': A_pre,
        'A0': A0,
        'Z1': Z1,
        'A1': A2,
        'Z2': Z2,
        'A2': A2
    }
        
    return cache


def nlpCost(params, data, labels, dict_size):
    
    W_embed = params['W_embed']
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']
        
    batch_size, context_size = data.shape
    data = np.eye(dict_size)[data]
    labels = np.eye(dict_size)[labels]
     
    A_pre = np.sum(data, axis=1) #/ context_size

    A0 = np.matmul(A_pre, W_embed)
    
    Z1 = np.matmul(A0, W1) + b1
    A1 = sigmoid(Z1)
    
    Z2 = np.matmul(A1, W2) + b2
    A2 = softmax(Z2, axis=1)
    
    J = np.sum(-labels * np.log(A2)) / batch_size
    
    dZ2 = (A2 - labels)
    dW2 = np.matmul(A1.T, dZ2) / batch_size
    db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size

    dA1 = np.matmul(dZ2, W2.T)
    dZ1 = sigmoid_backward(Z1) * dA1
    dW1 = np.matmul(A0.T, dZ1) / batch_size
    db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size
    
    dA0 = np.matmul(dZ1, W1.T) # = dZ0
    dW_embed = np.matmul(A_pre.T, dA0) / batch_size
    
    grads = {
        'dZ2': dZ2,
        'dW2': dW2,
        'db2': db2,
        'dA1': dA1,
        'dZ1': dZ1,
        'dW1': dW1,
        'db1': db1,
        'dA0': dA0,
        'dW_embed': dW_embed
    }
    return [J, grads]
    
    
def SGD_Q2(params, dict_size, data, labels, val_data, val_labels, lr, batch_size, stop_loss=0.1, max_epochs=50, alpha=0, verbose=True):
        
    sample_number = data.shape[0]
    
    passes_per_epoch = sample_number // batch_size
    
    loss_history = np.ones((max_epochs * passes_per_epoch, 1)) * stop_loss
    
    print('Starting training...')
    print(f'Target Validation Loss: {stop_loss} | Maximum Number of Epochs: {max_epochs}.\n')
    
    epoch = 0
    val_loss = stop_loss + 1
    
    while val_loss > stop_loss and epoch < max_epochs:
        
        dW_embed, dW1, dW2, db1, db2 = 0, 0, 0, 0, 0
        
        perm = np.random.permutation(sample_number)
        data = data[perm, :]
        labels = labels[perm]
        
        epoch += 1
        epoch_loss = 0
        correct_labels = 0
        
        if verbose:
            print(f'Epoch: [{epoch}/{max_epochs}] -> ', end='')
        
        for batch in range(passes_per_epoch):
            
            batch_data = data[batch*batch_size:(batch+1)*batch_size, :]
            batch_labels = labels[batch*batch_size:(batch+1)*batch_size]
            
            J, J_grad = nlpCost(params, batch_data, batch_labels, dict_size)
            A = NLP_forward_pass(params, batch_data, dict_size)['A2']
            
            epoch_loss += J
            pred_labels = np.argmax(A, axis=1)
            # true_labels = np.argmax(batch_labels, axis=1)
            correct_labels += np.where(pred_labels == batch_labels, 1, 0).sum()
            
            
            dW_embed = alpha * dW_embed - lr * J_grad['dW_embed']
            dW1 = alpha * dW1 - lr * J_grad['dW1']
            dW2 = alpha * dW2 - lr * J_grad['dW2']
            db1 = alpha * db1 - lr * J_grad['db1']
            db2 = alpha * db2 - lr * J_grad['db2']
            
            params['W_embed'] += dW_embed
            params['W1'] += dW1
            params['W2'] += dW2
            params['b1'] += db1
            params['b2'] += db2
            
            
            loss_history[(epoch-1)*passes_per_epoch + batch] = J
            
        
        if sample_number % batch_size != 0:
            
            batch_data = data[(passes_per_epoch*batch_size):, :]
            batch_labels = labels[(passes_per_epoch*batch_size):]

            J, J_grad = nlpCost(params, batch_data, batch_labels, dict_size)
            A = NLP_forward_pass(params, batch_data, dict_size)['A2']
            
            epoch_loss += J
            pred_labels = np.argmax(A, axis=1)
            correct_labels += np.where(pred_labels == batch_labels, 1, 0).sum()
            
            dW_embed = alpha * dW_embed - lr * J_grad['dW_embed']
            dW1 = alpha * dW1 - lr * J_grad['dW1']
            dW2 = alpha * dW2 - lr * J_grad['dW2']
            db1 = alpha * db1 - lr * J_grad['db1']
            db2 = alpha * db2 - lr * J_grad['db2']
            
            params['W_embed'] += dW_embed
            params['W1'] += dW1
            params['W2'] += dW2
            params['b1'] += db1
            params['b2'] += db2
            
        val_loss, _ = nlpCost(params, val_data, val_labels, dict_size)
        val_A = NLP_forward_pass(params, val_data, dict_size)['A2']
        
        val_preds = np.argmax(val_A, axis=1)
        val_correct_labels = np.where(val_preds == val_labels, 1, 0).sum()
        val_total_labels = val_data.shape[0]
        
        train_acc_percent = (correct_labels / sample_number) * 100
        val_acc_percent = (val_correct_labels / val_total_labels) * 100
        epoch_loss = epoch_loss * batch_size / sample_number
        
        
        if verbose:
            print(f'Training Loss: {epoch_loss:.4f} | Training Accuracy: {train_acc_percent:.2f}% | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc_percent:.2f}%')
        
        
    print('\nTraining completed.')
    
    return [params, loss_history]




############################
# BEGINING OF WORKING CODE #
############################

with h5py.File('data2.h5', 'r') as File:
    file_keys = list(File.keys())
    
    testd_ind = np.array(File[file_keys[0]]) - 1
    testx_ind = np.array(File[file_keys[1]]) - 1
    traind_ind = np.array(File[file_keys[2]]) - 1
    trainx_ind = np.array(File[file_keys[3]]) - 1
    vald_ind = np.array(File[file_keys[4]]) - 1
    valx_ind = np.array(File[file_keys[5]]) - 1
    words = np.array(File[file_keys[6]])

print('Train x shape:', trainx_ind.shape)
print('Train d shape:', traind_ind.shape)
print('Test x shape:', testx_ind.shape)
print('Test d shape:', testd_ind.shape)
print('Validation x shape:', valx_ind.shape)
print('Validation d shape:', vald_ind.shape)

num_words = words.shape[0]

# Initializing the 
Wb_32_256 = init_NLP_Wb(num_words, 32, 256, 0.01)
Wb_16_128 = init_NLP_Wb(num_words, 16, 128, 0.01)
Wb_8_64 = init_NLP_Wb(num_words, 8, 64, 0.01)


# Training for the instructed layer sizes
print('\nTraining for -> Embedding Size = 32 | Hidden Size = 256...\n')
Wb_32_256_trained, _ = SGD_Q2(Wb_32_256, num_words, trainx_ind, traind_ind, valx_ind, vald_ind, 0.15, 200, 2., alpha=0.85, max_epochs=50, verbose=True)

print('\nTraining for -> Embedding Size = 16 | Hidden Size = 128...\n')
Wb_16_128_trained, _ = SGD_Q2(Wb_16_128, num_words, trainx_ind, traind_ind, valx_ind, vald_ind, 0.15, 200, 2., alpha=0.85, max_epochs=50, verbose=True)

print('\nTraining for -> Embedding Size = 8 | Hidden Size = 64...\n')
Wb_8_64_trained, _ = SGD_Q2(Wb_8_64, num_words, trainx_ind, traind_ind, valx_ind, vald_ind, 0.15, 200, 2., alpha=0.85, max_epochs=50, verbose=True)


# Making predictions on the test data
pred_32_256 = NLP_forward_pass(Wb_32_256_trained, testx_ind, num_words)['A2']
pred_16_128 = NLP_forward_pass(Wb_16_128_trained, testx_ind, num_words)['A2']
pred_8_64 = NLP_forward_pass(Wb_8_64_trained, testx_ind, num_words)['A2']

# Retriving the 10 most expected words
pred_32_256 = np.argsort(pred_32_256, axis=1)[:, -10:]
pred_16_128 = np.argsort(pred_16_128, axis=1)[:, -10:]
pred_8_64 = np.argsort(pred_8_64, axis=1)[:, -10:]

# Sampling 5 random trigrams from the test data set
test_size = testx_ind.shape[0]
number_of_samples = 5
random_samples = np.random.permutation(test_size)[:number_of_samples]

for i in range(number_of_samples):
    print(f'\nTrigram: {words[testx_ind[random_samples[i]]]} |', f'True Value: {words[testd_ind[random_samples[i]]]}')
    print(f'Top Predictions for Embedding Size = 32, Hidden Size = 256: ', [word for word in reversed(words[pred_32_256[i]])])
    print(f'Top Predictions for Embedding Size = 16, Hidden Size = 128: ', [word for word in reversed(words[pred_16_128[i]])])
    print(f'Top Predictions for Embedding Size = 8, Hidden Size = 64: ', [word for word in reversed(words[pred_8_64[i]])], '\n')