import numpy as np

class MyRidgeGD:
    def __init__(self, lr=0.1, n_iter=300, alpha=1.0, batch_size=256, random_state=42):
        self.lr = lr
        self.n_iter = n_iter
        self.alpha = alpha
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        Xb = np.c_[np.ones(n), X]
        if X_val is not None:
            Xvb = np.c_[np.ones(len(X_val)), np.asarray(X_val, dtype=float)]
            yv = np.asarray(y_val, dtype=float)
        w = np.zeros(d + 1)
        ht = []
        hv = []
        bs = int(self.batch_size)
        for _ in range(int(self.n_iter)):
            perm = rng.permutation(n)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                m = len(idx)
                err = Xb[idx] @ w - y[idx]
                g = (Xb[idx].T @ err) / m
                g[1:] = g[1:] + self.alpha * w[1:] / m
                w = w - self.lr * g
            pt = Xb @ w
            ht.append(float(np.mean(np.abs(pt - y))))
            if X_val is not None:
                pv = Xvb @ w
                hv.append(float(np.mean(np.abs(pv - yv))))
        self.w_ = w
        self.hist_train_ = ht
        self.hist_val_ = hv
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.c_[np.ones(len(X)), X] @ self.w_
