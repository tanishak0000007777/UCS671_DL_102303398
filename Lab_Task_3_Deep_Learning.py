

import pandas as pd
import numpy as np

# load dataset
df = pd.read_csv("/content/glass.csv")

# check shape
print(df.shape)

# see columns
print(df.columns)

# first rows
print(df.head())

# create binary label
df["y"] = (df["Type"] == 1).astype(int)

# drop useless columns
df = df.drop(columns=["Type","Id"], errors='ignore')

X = df.drop(columns=["y"]).values
y = df["y"].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w, b):
    z = np.dot(X, w) + b
    p = sigmoid(z)
    return p

def loss(y, p):
    return -np.mean(y*np.log(p + 1e-9) + (1-y)*np.log(1-p + 1e-9))

def update_weights(X, y, w, b, lr):

    p = predict_proba(X, w, b)
    error = p - y

    w = w - lr * (np.dot(X.T, error)) / len(y)
    b = b - lr * np.mean(error)

    return w, b

# initialize
w = np.zeros(X_train.shape[1])
b = 0.0
lr = 0.1
epochs = 200

for i in range(epochs):
    w, b = update_weights(X_train, y_train, w, b, lr)

    if i % 20 == 0:
        p = predict_proba(X_train, w, b)
        print("Epoch", i, "Loss:", loss(y_train, p))

print("\nFinal weights:", w)
print("Final bias:", b)

def predict_label(p, threshold=0.5):
    return (p >= threshold).astype(int)

# test predictions
prob = predict_proba(X_test, w, b)
pred = predict_label(prob, 0.5)

accuracy = np.mean(pred == y_test)
print("\nTest Accuracy:", accuracy)

pred2 = predict_label(prob, 0.7)
accuracy2 = np.mean(pred2 == y_test)
print("Accuracy with threshold 0.7:", accuracy2)

