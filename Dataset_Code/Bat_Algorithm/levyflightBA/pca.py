import sklearn.decomposition
import numpy as np

class PCA:
    def __init__(self):
        self.m = None
        self.V = None

    def _number_of_components(self, contribution_rate, eigenvalues):
        total = np.sum(eigenvalues)
        increment = 0.0
        for m, value in enumerate(eigenvalues):
            increment += value
            if increment / total > contribution_rate:
                # Starts at 0
                return m + 1

    def replace_principal_individuals(self, X, lb, ub):
        self._custom_pca(X, return_pca=False)
        V = self.V
        for i in range(self.m):
            X[i] = np.clip(np.dot(V[i], X[i]), lb, ub)
        return X

    def _custom_pca(self, matrix, return_pca=True):
        if len(matrix.shape) >= 3:
            Exception("Dimensions >=3 not supported in custom_pca.")
        # calculate the mean of each column
        M = np.mean(matrix.T, axis=1)
        # center columns by subtracting column means
        C = matrix - M
        # calculate covariance matrix of centered matrix
        self.V = np.cov(C.T)
        # eigendecomposition of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(self.V)
        self.m = self._number_of_components(0.85, eigenvalues)
        if return_pca:
            # project data
            P = eigenvectors.T.dot(C.T)
            return P.T[:self.m]
        return None

    def _industry_pca(self, matrix):
        if len(matrix.shape > 3):
            Exception("Input dimensions >=3 not supported in industry_pca.")
        if len(matrix.shape) == 3:
            samples, nx, ny = matrix.shape
            matrix = matrix.reshape(samples, nx*ny)
        pca = sklearn.decomposition.PCA()
        pca = pca.fit(matrix)
        return pca.transform(matrix)
