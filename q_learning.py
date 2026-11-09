import numpy as np

# 1. Le réseau Q (il prend l'état et sort les Q-valeurs pour chaque action)
class QNetwork:
    def __init__(self):
        # 2 entrées (le XOR), 2 sorties (les Q-valeurs pour l'action 0 et l'action 1)
        self.W = np.random.randn(2, 2) * 0.1
        self.b = np.zeros((1, 2))

    def forward(self, X):
        # Pas de Softmax ici ! On veut des scores bruts
        return np.dot(X, self.W) + self.b

# 2. Les données (XOR)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0]) # La bonne action à choisir (0 ou 1)

net = QNetwork()
lr = 0.1

print("Starting Q-Learning...")
for episode in range(1000):
    for i in range(4):
        state = X[i].reshape(1, 2)
        target = y[i]
        
        # 1. Forward (Le réseau donne un score pour chaque action)
        q_values = net.forward(state)
        
        # 2. On choisit l'action avec le meilleur score
        action = np.argmax(q_values)
        
        # 3. On calcule la récompense (1 si bonne action, -1 si mauvaise)
        reward = 1 if action == target else -1
        
        # 4. La cible du Q-Learning
        target_q = q_values[0, action] + lr * (reward - q_values[0, action])
        
        # 5. Mise à jour des poids (Gradient très simple)
        error = target_q - q_values[0, action]
        net.W[:, action] += lr * error * state[0]
        net.b[0, action] += lr * error
        
    if episode % 200 == 0:
        # Test
        scores = [np.argmax(net.forward(X[i].reshape(1,2))) for i in range(4)]
        print(f"Epoch {episode} | Prédictions : {scores}")

print("\n🧠 Résultats finaux :")
for i in range(4):
    q = net.forward(X[i].reshape(1,2))
    print(f"Entrée : {X[i]} -> Action choisie : {np.argmax(q)} (Q-valeurs : {q[0]})")