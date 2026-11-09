
# 🧠 Neural Networks & RL From Scratch

> Implementing the core mathematics of Deep Learning and Reinforcement Learning using only `NumPy`. No PyTorch, no TensorFlow.

## Overview

This project is a pedagogical deep-dive into the algorithms that power modern AI. By building these networks brick by brick, I aim to demystify the "black box" of libraries like PyTorch.

### What's implemented:
- **Multi-Layer Perceptron (MLP)**: Forward pass, ReLU, Softmax, and manual Backpropagation.
- **Orthogonal Initialization**: To solve the symmetry-breaking problem in neural networks.
- **Q-Learning**: A basic value-based RL algorithm.
- **Deep Q-Network (DQN)**: Combining Q-Learning with a neural network, featuring Epsilon-Greedy exploration.

## 📂 Project Structure

```text
scratch-nn/
├── neural_network.py   # The MLP class (Forward/Backward math)
├── test_xor.py         # Training the network on the XOR logic problem
├── q_learning.py       # Basic Q-Learning implementation
├── dqn.py              # Deep Q-Network with Epsilon-Greedy
└── README.md
```

## Usage

Make sure you have `numpy` installed:
```bash
pip install numpy
```

**1. Test the Neural Network on XOR:**
```bash
python test_xor.py
```
*(The network learns to predict `[0,1]` and `[1,0]` outputs, proving it understands non-linear logic).*

**2. Run the DQN:**
```bash
python dqn.py
```

## 🧠 Key Learnings
- Implementing the chain rule manually for backpropagation.
- Understanding why initializing weights to zero breaks learning (and why orthogonal init works).
- Seeing how RL agents learn through trial and error using TD-Errors.

## 📝 License
MIT License.
```