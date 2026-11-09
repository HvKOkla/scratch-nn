import numpy as np

class QNetwork:
    def __init__(self):
        # Couche cachée (2 entrées -> 4 neurones cachés)
        self.W1 = np.random.randn(2, 4) * 0.1
        self.b1 = np.zeros((1, 4))
        
        # Couche de sortie (4 neurones cachés -> 2 actions)
        self.W2 = np.random.randn(4, 2) * 0.1
        self.b2 = np.zeros((1, 2))

    def forward(self, X):
        # Forward avec ReLU (la magie du non-linéaire)
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2

    def backward(self, X, action, error, lr=0.1):
        # Rétropropagation du gradient (seulement pour l'action choisie)
        dZ2 = np.zeros_like(self.Z2)
        dZ2[0, action] = -error # On corrige uniquement l'action qui a fait une erreur
        
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = dZ2
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0) # Dérivée de ReLU
        
        dW1 = np.dot(X.T, dZ1)
        db1 = dZ1
        
        # Mise à jour
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

# Données
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

net = QNetwork()
lr = 0.1
epsilon = 1.0 # Taux d'exploration (on va le réduire)

print("🚀 Starting Deep Q-Learning (DQN)...")
for episode in range(1000):
    for i in range(4):
        state = X[i].reshape(1, 2)
        target = y[i]
        
        # 1. Epsilon-Greedy : Parfois on explore au hasard !
        if np.random.rand() < epsilon:
            action = np.random.randint(2)
        else:
            q_values = net.forward(state)
            action = np.argmax(q_values)
            
        # 2. On calcule la récompense
        reward = 1 if action == target else -1
        
        # 3. On calcule l'erreur
        q_values = net.forward(state)
        target_q = q_values[0, action] + lr * (reward - q_values[0, action])
        error = target_q - q_values[0, action]
        
        # 4. On met à jour le réseau
        net.backward(state, action, error, lr)
        
    # On réduit l'exploration au fil du temps
    epsilon = max(0.01, epsilon * 0.99)
    
    if episode % 200 == 0:
        scores = []
        for i in range(4):
            q = net.forward(X[i].reshape(1,2))
            scores.append(np.argmax(q))
        print(f"Epoch {episode} | Prédictions : {scores}")

print("\n🧠 Résultats finaux :")
for i in range(4):
    q = net.forward(X[i].reshape(1,2))
    print(f"Entrée : {X[i]} -> Action choisie : {np.argmax(q)}")