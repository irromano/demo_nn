import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Generate dummy dataset (X: 2 features, y: binary labels)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1],
              [0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0], [0], [1], [1], [0]])  # XOR pattern (for example)

# Network architecture
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 10000

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.ones((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.ones((1, output_size))

# Activation functions
def relu(z) -> np.ndarray:
    return np.maximum(0, z)

def relu_derivative(z) -> np.ndarray:
    return (z > 0).astype(float)

def sigmoid(z) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)

    # Loss (Mean Squared Error)
    loss = np.mean((y - a2) ** 2)

    # error2 = da2_dz2 * dc_da2
    error2 = sigmoid_derivative(z2) * (a2 - y) * 2
    dc_dw2 = error2.T.dot(a1).T
    dc_db2 = np.sum(error2, axis=0, keepdims=True)

    # error1 = error2 dot w2 * da1_dz1
    error1 = error2.dot(W2.T) * sigmoid_derivative(z1)
    dc_dw1 = X.T.dot(error1)
    dc_db1 = np.sum(error1, axis=0, keepdims=True)

    # Update weights and biases
    W2 -= learning_rate * dc_dw2
    b2 -= learning_rate * dc_db2
    W1 -= learning_rate * dc_dw1
    b1 -= learning_rate * dc_db1

    # Print loss occasionally
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final predictions
predictions = (a2 > 0.5).astype(int)
print("Final predictions:")
print(predictions)
