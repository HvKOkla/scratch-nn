import numpy as np
from neural_network import NeuralNetwork

# 1. Input data (The XOR problem: inputs are 0 or 1)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# 2. Expected answers (The correct "classes")
y = np.array([
    [1, 0], # If input [0,0], answer is class 0
    [0, 1], # If input [0,1], answer is class 1
    [0, 1], # If input [1,0], answer is class 1
    [1, 0]  # If input [1,1], answer is class 0
])


network = NeuralNetwork(input_size=2, hidden_size=4, output_size=2)

# 4. Training loop (We repeat this 5000 times)
print("🚀 Starting training...")
for epoch in range(5000):
    # Forward pass (The network tries to guess)
    probs = network.forward(X)
    
    # Backward pass (The network learns from its mistakes)
    loss = network.backward(X, y, learning_rate=0.1)
    
    # Print the error every 1000 steps to see it going down
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss : {loss:.4f}")

# 5. Final test (Let's see if the network is smart now)
print("\n🧠 Results after training:")
probs = network.forward(X)
for i in range(4):
    input_val = X[i]
    prediction = np.argmax(probs[i])
    print(f"Input : {input_val} -> Prediction : {prediction} (Confidence : {probs[i][prediction]*100:.1f}%)")