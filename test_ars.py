import numpy as np

# Le même réseau qu'avant, mais SANS backward !
class SimpleNet:
    def __init__(self):
        # On met tous les poids dans un seul gros vecteur
        self.weights = np.random.randn(2 * 4 + 4 + 4 * 2 + 2) * 0.1

    def get_weights(self):
        return self.weights.copy()

    def set_weights(self, w):
        self.weights = w.copy()

    def forward(self, X):
        # On découpe le vecteur pour retrouver nos matrices
        W1 = self.weights[0:8].reshape(2, 4)
        b1 = self.weights[8:12].reshape(1, 4)
        W2 = self.weights[12:20].reshape(4, 2)
        b2 = self.weights[20:22].reshape(1, 2)
        
        Z1 = np.dot(X, W1) + b1
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(A1, W2) + b2
        
        exp_x = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Le problème XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[1,0], [0,1], [0,1], [1,0]])

net = SimpleNet()
best_weights = net.get_weights()
best_score = -999
learning_rate = 0.1
noise = 0.1

print("Starting ARS (Augmented Random Search)...")
for step in range(1000):
    noise_vec = np.random.randn(len(best_weights)) * noise
    w_plus = best_weights + noise_vec
    w_minus = best_weights - noise_vec
    
    # 2. On teste les deux versions
    net.set_weights(w_plus)
    score_plus = np.sum(np.argmax(net.forward(X), axis=1) == np.argmax(y, axis=1))
    
    net.set_weights(w_minus)
    score_minus = np.sum(np.argmax(net.forward(X), axis=1) == np.argmax(y, axis=1))
    
    # 3. Si l'une des deux versions est meilleure, on met à jour les poids
    if score_plus > score_minus:
        best_weights = best_weights + learning_rate * noise_vec
    else:
        best_weights = best_weights - learning_rate * noise_vec
        
    if step % 100 == 0:
        net.set_weights(best_weights)
        current_score = np.sum(np.argmax(net.forward(X), axis=1) == np.argmax(y, axis=1))
        print(f"Step {step} | Score parfait : {current_score}/4")

net.set_weights(best_weights)
print("\n🧠 Résultats finaux :")
probs = net.forward(X)
for i in range(4):
    print(f"Entrée : {X[i]} -> Prédiction : {np.argmax(probs[i])}")