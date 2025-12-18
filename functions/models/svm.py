import numpy as np
import pandas as pd
from functions.utils import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# PREDIKSI
class svr:
    def __init__(
        self,
        C=1.0,
        epsilon=0.1,
        gamma=None,
        max_iter=500,
        lr=0.01
    ):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.max_iter = max_iter
        self.lr = lr

        # learned params
        self.alpha = None
        self.alpha_star = None
        self.X = None
        self.y = None
        self.b = 0.0

    # rbf-kernel
    def _kernel(self, X1, X2):
        sqdist = (
            np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            + np.sum(X2 ** 2, axis=1)
            - 2 * X1 @ X2.T
        )
        return np.exp(-self.gamma * sqdist)

    # Training
    def fit(self, X, y):
        self.X = X
        self.y = y

        n_samples, n_features = X.shape
        if self.gamma is None:
            self.gamma = 1.0 / n_features

        self.alpha = np.zeros(n_samples)
        self.alpha_star = np.zeros(n_samples)

        K = self._kernel(X, X)

        # loss
        self.loss_history = []

        for _ in range(self.max_iter):
            f = K @ (self.alpha - self.alpha_star)
            errors = f - y

            # mse loss
            mse = np.mean(errors ** 2)
            self.loss_history.append(mse)

            outside = np.abs(errors) > self.epsilon
            grad = np.sign(errors[outside])

            self.alpha[outside] -= self.lr * grad
            self.alpha_star[outside] += self.lr * grad

            self.alpha = np.clip(self.alpha, 0, self.C)
            self.alpha_star = np.clip(self.alpha_star, 0, self.C)

        # Bias
        sv_mask = (self.alpha > 1e-5) | (self.alpha_star > 1e-5)
        if np.any(sv_mask):
            self.b = np.mean(
                y[sv_mask]
                - K[sv_mask] @ (self.alpha - self.alpha_star)
            )
        else:
            self.b = 0.0

        return self


    def predict(self, X):
        K_test = self._kernel(X, self.X)
        return K_test @ (self.alpha - self.alpha_star) + self.b

    def save(self, path):
        joblib.dump({
            "C": self.C,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "max_iter": self.max_iter,
            "lr": self.lr,
            "alpha": self.alpha,
            "alpha_star": self.alpha_star,
            "b": self.b,
            "X": self.X,
            "y": self.y
        }, path)

    def load(self, path):
        data = joblib.load(path)
        self.C = data["C"]
        self.epsilon = data["epsilon"]
        self.gamma = data["gamma"]
        self.max_iter = data["max_iter"]
        self.lr = data["lr"]
        self.alpha = data["alpha"]
        self.alpha_star = data["alpha_star"]
        self.b = data["b"]
        self.X = data["X"]
        self.y = data["y"]

# KLASIFIKASI
class svm_rbf_klasifikasi:
    def __init__(self, C=1.0, gamma=0.1, tol=1e-3, max_passes=5):
        self.C = C
        self.gamma = gamma
        self.tol = tol # toleransi utk KKT violation
        self.max_passes = max_passes
        self.alphas = None
        self.b = 0
        self.X = None
        self.y = None
        self.kernel = None

    def compute_kernel_matrix(self, X):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = rbf_kernel(X[i], X[j], self.gamma)
        return K

    def fit(self, X, y):
        n, d = X.shape
        self.X = X
        self.y = np.where(y == 0, -1, 1)  # 0→-1
        self.alphas = np.zeros(n)

        # Pre-komputasi kernel matrix
        K = self.compute_kernel_matrix(X)

        passes = 0
        while passes < self.max_passes:
            alpha_changed = 0

            for i in range(n):

                # Komputasi Ei = f(xi) - yi
                f_i = np.sum(self.alphas * self.y * K[:, i]) + self.b
                E_i = f_i - self.y[i]

                # Cek KKT violation
                if (self.y[i] * E_i < -self.tol and self.alphas[i] < self.C) or \
                (self.y[i] * E_i > self.tol and self.alphas[i] > 0):

                    # Pilih j != i
                    j = np.random.randint(0, n)
                    while j == i:
                        j = np.random.randint(0, n)

                    f_j = np.sum(self.alphas * self.y * K[:, j]) + self.b
                    E_j = f_j - self.y[j]

                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    # Komputasi batas L & H
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])

                    if L == H:
                        continue

                    # Komputasi eta (turunan kedua)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    self.alphas[j] -= self.y[j] * (E_i - E_j) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)

                    # Jika perubahan kecil, skip
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    self.alphas[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alphas[j])

                    # Update bias term b
                    b1 = self.b - E_i \
                        - self.y[i] * (self.alphas[i] - alpha_i_old) * K[i, i] \
                        - self.y[j] * (self.alphas[j] - alpha_j_old) * K[i, j]

                    b2 = self.b - E_j \
                        - self.y[i] * (self.alphas[i] - alpha_i_old) * K[i, j] \
                        - self.y[j] * (self.alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    alpha_changed += 1

            if alpha_changed == 0:
                passes += 1
            else:
                passes = 0  # reset jika progress terjadi

    def predict(self, X):
        preds = []
        for x in X:
            s = 0
            for alpha, y_i, x_i in zip(self.alphas, self.y, self.X):
                if alpha > 1e-6:  # only use support vectors
                    s += alpha * y_i * rbf_kernel(x_i, x, self.gamma)
            preds.append(np.sign(s + self.b))
        return np.where(np.array(preds) == -1, 0, 1)
    
    def save(self, path):
        joblib.dump({
        'C': self.C,
        'gamma': self.gamma,
        'tol': self.tol,
        'max_passes': self.max_passes,
        'alphas': self.alphas,
        'b': self.b,
        'X': self.X,
        'y': self.y
        }, path)


    def load(self, path):
        data = joblib.load(path)
        self.C = data['C']
        self.gamma = data['gamma']
        self.tol = data['tol']
        self.max_passes = data['max_passes']
        self.alphas = data['alphas']
        self.b = data['b']
        self.X = data['X']
        self.y = data['y']