
import numpy as np

np.random.seed(42)

# -----------------------------
# DATASET (given in lab)
# -----------------------------
X = np.random.uniform(-2, 2, (400, 3))
y = (
    np.sin(X[:,0]) +
    0.5*(X[:,1]**2) -
    0.8*X[:,2]
).reshape(-1,1)

X = X.T
y = y.T

# -----------------------------
# ACTIVATIONS + DERIVATIVES
# -----------------------------
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s*(1-s)

# -----------------------------
# INIT PARAMETERS
# -----------------------------
def init_params(layers):
    params = {}
    for i in range(len(layers)-1):
        params["W"+str(i+1)] = np.random.randn(layers[i+1], layers[i]) * 0.1
        params["b"+str(i+1)] = np.zeros((layers[i+1],1))
    return params

# -----------------------------
# FORWARD PASS
# -----------------------------
def forward(X, params, activation):
    cache = {"A0": X}
    L = len(params)//2

    for i in range(1, L+1):
        W = params["W"+str(i)]
        b = params["b"+str(i)]

        Z = W @ cache["A"+str(i-1)] + b
        cache["Z"+str(i)] = Z

        if i == L:
            A = Z  # output linear
        else:
            A = activation(Z)

        cache["A"+str(i)] = A

    return A, cache

# -----------------------------
# LOSS
# -----------------------------
def mse(y, yhat):
    return np.mean((y - yhat)**2)

# -----------------------------
# BACKPROP
# -----------------------------
def backward(y, params, cache, activation_deriv):
    grads = {}
    L = len(params)//2
    m = y.shape[1]

    dA = -2*(y - cache["A"+str(L)])

    for i in reversed(range(1, L+1)):
        Z = cache["Z"+str(i)]
        A_prev = cache["A"+str(i-1)]

        if i == L:
            dZ = dA
        else:
            dZ = dA * activation_deriv(Z)

        grads["dW"+str(i)] = (1/m)*(dZ @ A_prev.T)
        grads["db"+str(i)] = (1/m)*np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            W = params["W"+str(i)]
            dA = W.T @ dZ

    return grads

# -----------------------------
# UPDATE
# -----------------------------
def update(params, grads, lr):
    L = len(params)//2
    for i in range(1, L+1):
        params["W"+str(i)] -= lr*grads["dW"+str(i)]
        params["b"+str(i)] -= lr*grads["db"+str(i)]
    return params

# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train(layers, act_name="relu", epochs=1000, lr=0.01):

    if act_name=="relu":
        activation = relu
        activation_deriv = relu_deriv
    else:
        activation = sigmoid
        activation_deriv = sigmoid_deriv

    params = init_params(layers)

    for epoch in range(epochs):

        yhat, cache = forward(X, params, activation)
        loss = mse(y, yhat)

        grads = backward(y, params, cache, activation_deriv)
        params = update(params, grads, lr)

        if epoch==200:
            loss200 = loss

    # gradient norms
    first_grad = np.linalg.norm(grads["dW1"])
    last_grad = np.linalg.norm(grads["dW"+str(len(layers)-1)])

    print("\nArchitecture:", layers)
    print("Activation:", act_name)
    print("Final loss:", loss)
    print("Loss @200:", loss200)
    print("Grad norm first:", first_grad)
    print("Grad norm last:", last_grad)

print("Running Shallow Model")
train([3,4,1],"relu")
train([3,4,1],"sigmoid")

print("\nRunning Medium Model")
train([3,6,6,1],"relu")
train([3,6,6,1],"sigmoid")

print("\nRunning Deep Model")
train([3,8,8,8,8,1],"relu")
train([3,8,8,8,8,1],"sigmoid")

print("\nRunning Very Deep Model")
train([3,8,8,8,8,8,8,8,8,1],"relu")
train([3,8,8,8,8,8,8,8,8,1],"sigmoid")

