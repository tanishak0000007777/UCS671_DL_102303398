# Name: Tanishak  
# Roll No: 102303398  

# a = 8  
# b = 9  
# c = 3  

# Hidden width = 14  
# Hidden layers = 4  
# Learning rate (baseline) = 0.008  
# Activation = ReLU  
# Initialization range = [-1/9, +1/9]


import numpy as np


def relu(a):
    return np.maximum(0, a)

def relu_deriv(x):
    return (x > 0).astype(float)

def init_network(input_size, hidden_size, num_hidden, output_size, a):
    params = {}
    limit = 1 / (a + 1)
    layer_sizes = [input_size] + [hidden_size]*num_hidden + [output_size]
    
    for i in range(len(layer_sizes)-1):
        params['W'+str(i+1)] = np.random.uniform(
            -limit, limit,
            (layer_sizes[i], layer_sizes[i+1])
        )
        params['b'+str(i+1)] = np.zeros((1, layer_sizes[i+1]))
        
    return params

def forward(X, params, num_hidden):
    cache = {'A0': X}
    A = X
    
    for i in range(1, num_hidden+1):
        Z = A @ params['W'+str(i)] + params['b'+str(i)]
        A = relu(Z)
        cache['Z'+str(i)] = Z
        cache['A'+str(i)] = A
    
    # output layer (linear)
    Z = A @ params['W'+str(num_hidden+1)] + params['b'+str(num_hidden+1)]
    cache['Z'+str(num_hidden+1)] = Z
    cache['A'+str(num_hidden+1)] = Z
    
    return Z, cache

def loss_fn(y_pred, y):
    return np.mean((y_pred - y)**2)

def backward(y_pred, y, params, cache, num_hidden):
    grads = {}
    m = y.shape[0]
    
    dA = 2*(y_pred - y)/m
    
    # output layer
    L = num_hidden + 1
    grads['W'+str(L)] = cache['A'+str(L-1)].T @ dA
    grads['b'+str(L)] = np.sum(dA, axis=0, keepdims=True)
    
    dA_prev = dA @ params['W'+str(L)].T
    
    # hidden layers
    for i in reversed(range(1, num_hidden+1)):
        dZ = dA_prev * relu_deriv(cache['Z'+str(i)])
        grads['W'+str(i)] = cache['A'+str(i-1)].T @ dZ
        grads['b'+str(i)] = np.sum(dZ, axis=0, keepdims=True)
        if i > 1:
            dA_prev = dZ @ params['W'+str(i)].T
    
    return grads


def update(params, grads, lr):
    for key in params:
        params[key] -= lr * grads[key]
    return params


def gradient_norm(grads, layer):
    return np.linalg.norm(grads['W'+str(layer)])


def train(lr):
    input_size = 10
    hidden_size = 14
    num_hidden = 4
    output_size = 1
    
    X = np.random.randn(100, input_size)
    y = np.sum(X, axis=1, keepdims=True)
    
    params = init_network(input_size, hidden_size, num_hidden, output_size, a=8)
    
    losses = {}
    
    for epoch in range(1,401):
        y_pred, cache = forward(X, params, num_hidden)
        loss = loss_fn(y_pred, y)
        grads = backward(y_pred, y, params, cache, num_hidden)
        params = update(params, grads, lr)
        
        if epoch in [1,100,400]:
            losses[epoch] = loss
    
    g_first = gradient_norm(grads, 1)
    g_last = gradient_norm(grads, num_hidden)
    GRI = g_first / (g_last + 1e-8)
    
    return losses, g_first, g_last, GRI


baseline = train(lr=0.008)
print(baseline)

break_run = train(lr=0.16)
print(break_run)

b_loss, b_f, b_l, b_gri = baseline
br_loss, br_f, br_l, br_gri = break_run

print("\n----- Detailed Results -----")

print("\nBaseline:")
print("Epoch1:", b_loss[1])
print("Epoch100:", b_loss[100])
print("Epoch400:", b_loss[400])
print("First grad norm:", b_f)
print("Last grad norm:", b_l)
print("GRI:", b_gri)

print("\nBreak Run:")
print("Epoch1:", br_loss[1])
print("Epoch100:", br_loss[100])
print("Epoch400:", br_loss[400])
print("First grad norm:", br_f)
print("Last grad norm:", br_l)
print("GRI:", br_gri)
