import numpy as np
X = np.load("train_X.npy")
print(np.min(X), np.max(X))
print(np.mean(X, axis=0))
print(np.std(X, axis=0))
