'''
MODULE: sparse_sensing.py
@Authors:
    A. Procacci [1]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
@Contacts:
    alberto.procacci@ulb.be
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: alberto.procacci@ulb.be
'''

import os
import numpy as np
import scipy.linalg as la

class SPR():
    
    def __init__(self, X, n_features):
        self.X = X
        self.n_features = n_features
        
    
    def optimal_placement(self):
        n = self.X.shape[0]
        m = self.X.shape[1]

        n_points = n // self.n_features
        
        mean_matrix = np.zeros((self.n_features, m))
        std_matrix = np.zeros((self.n_features, m))
        X0 = np.zeros_like(self.X)
        
        # Scale the matrix to unitary variance
        print('Scaling the matrix...')
        for i in range(self.n_features):
            for j in range(m):
                x = self.X[i*n_points:(i+1)*n_points,j]
                mean_matrix[i, j] = np.average(x)
                std_matrix[i, j] = np.std(x)
                X0[i*n_points:(i+1)*n_points,j] = (x - mean_matrix[i,j])/std_matrix[i, j]
        
        # Compute the SVD of the scaled dataset
        print('Computing the SVD...')
        U, S, Vt = np.linalg.svd(X0, full_matrices=False)
        L = S**2    # Compute the eigenvalues
        exp_variance = 100*np.cumsum(L)/np.sum(L)
        
        # The r-order truncation is selected based on the amount of variance recovered
        for r in range(exp_variance.size):
            if exp_variance[r]>99.5:
                break
        
        # Reduce the dimensionality
        Ur = U[:,:r]
        
        # Calculate the QRCP 
        print('Computing the QRCP and calculating C...')
        Q, R, P = la.qr(Ur.T, pivoting=True, mode='economic')
        p = r
        C = np.zeros((p, n))
        for j in range(p):
            C[j,P[j]] = 1
        
        return C

