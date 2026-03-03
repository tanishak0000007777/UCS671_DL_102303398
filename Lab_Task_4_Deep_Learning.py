import pandas as pd
import numpy as np

# load dataset
data = pd.read_csv("/content/Salary_Data.csv")

print(data.head())

# -----------------------------
# KEEP ONLY NUMERIC COLUMNS
# -----------------------------
data = data[["Age", "Years of Experience", "Salary"]]

# remove missing values
data = data.dropna()

# -----------------------------
# INPUT & OUTPUT
# -----------------------------
X = data[["Age", "Years of Experience"]].values
y = data["Salary"].values

# -----------------------------
# NORMALIZE DATA (VERY IMPORTANT)
# -----------------------------
X = X.astype(float)
y = y.astype(float)

X = X / np.max(X, axis=0)
y = y / np.max(y)

# -----------------------------
# INITIALIZE
# -----------------------------
w = np.zeros(X.shape[1])
b = 0.0

# -----------------------------
# FUNCTIONS
# -----------------------------
def predict(X, w, b):
    return np.dot(X, w) + b

def mse(y, y_hat):
    return np.mean((y_hat - y)**2)

def gradients(X, y, y_hat):
    N = len(y)
    dw = (2/N) * np.dot(X.T, (y_hat - y))
    db = (2/N) * np.sum(y_hat - y)
    return dw, db

# -----------------------------
# TRAINING
# -----------------------------
lr = 0.01
epochs = 1000

for epoch in range(epochs):

    y_hat = predict(X, w, b)
    loss = mse(y, y_hat)

    dw, db = gradients(X, y, y_hat)

    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        print("Epoch", epoch, "Loss:", loss)

print("\nFinal weights:", w)
print("Final bias:", b)

# -----------------------------
# PREDICT NEW SALARY
# -----------------------------
new_person = np.array([30, 5], dtype=float)
new_person = new_person / np.max(X, axis=0)

pred_scaled = np.dot(new_person, w) + b
pred_salary = pred_scaled * np.max(data["Salary"])

print("\nPredicted Salary:", pred_salary)

