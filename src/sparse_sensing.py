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

    '''
    Class used for Sparse Placement for Reconstruction (SPR)
    
    Attributes
    ---
    X : numpy array
        data matrix of dimensions (n,p) where n = n_features * n_points and p
        is the number of operating conditions.
    
    n_features : int
        the number of features in the dataset (temperature, velocity, etc.).
        
    Methods
    ---
    scale_data(scale_type='standard')
        Scale the data.
    
    unscale_data(x0, scale_type)
        Unscale the data.
    
    decomposition(X0, decomp_type='POD')
        Finds the taylored basis.
        
    reduction(U, exp_variance, select_modes, n_modes)
        Perform the dimensionality reduction.

    optimal_placement(scale_type='standard', select_modes='variance', n_modes=99)
        Calculates the C matrix using QRCP decomposition.
    
    fit_predict(C, y, scale_type='standard', select_modes='variance', n_modes=99)
        Calculates the Theta matrix, then predicts x.
    
    predict(y, scale_type='standard'):
        Predicts x.
    
    '''

    def __init__(self, X, n_features):
        '''    
        Parameters
        ----------
        X : numpy array
            Data matrix of dimensions (nxm) where n = n_features * n_points and m
            is the number of operating conditions.
        n_features : int
            The number of features in the dataset (temperature, velocity, etc.).

        Returns
        -------
        None.

        '''
        self.X = X
        self.n_features = n_features

    def scale_data(self, scale_type='standard'):
        '''
        Return the scaled data matrix. The default is to scale the data to 
        unitary variance.
        
        Parameters
        ----------
        scale_type : str, optional
            Type of scaling. The default is 'standard'.

        Returns
        -------
        X0 : numpy array
            Scaled data matrix.

        '''
        
        n = self.X.shape[0]
        m = self.X.shape[1]
        X0 = np.zeros_like(self.X)
        n_points = n // self.n_features

        if scale_type == 'standard':
            # Scale the matrix to unitary variance
            mean_matrix = np.zeros((self.n_features, m))
            std_matrix = np.zeros((self.n_features, m))
            print('Scaling the matrix...')
            for i in range(self.n_features):
                for j in range(m):
                    x = self.X[i*n_points:(i+1)*n_points, j]
                    mean_matrix[i, j] = np.average(x)
                    std_matrix[i, j] = np.std(x)
                    X0[i*n_points:(i+1)*n_points, j] = (x -
                                                        mean_matrix[i, j])/std_matrix[i, j]

            self.mean_matrix = mean_matrix
            self.std_matrix = std_matrix
        else:
            'raise exception'

        return X0

    def unscale_data(self, x0, scale_type):
        '''
        Return the unscaled vector.
        
        Parameters
        ----------
        x0 : numpy array
            Scaled vector to unscale, size (n,).
        scale_type : str
            Type of scaling.

        Returns
        -------
        x : numpy array
            The unscaled vector.

        '''
        x = np.zeros_like(x0)
        
        if scale_type == 'standard':
            for i in range(self.n_features):
                x[i*n_points:(i+1)*n_points] = np.average(self.std_matrix[i,:]) * x0[i*n_points:(i+1)*n_points] + np.average(self.mean_matrix[i,:])

        return x

    def decomposition(self, X0, decomp_type='POD'):
        '''
        Return the taylored basis and the amount of variance of the modes.

        Parameters
        ----------
        X0 : numpy array
            The scaled data matrix to be decomposed, size (n,p).
        decomp_type : str, optional
            Type of decomposition. The default is 'POD'.

        Returns
        -------
        U : numpy array
            The taylored basis used for SPR, size (n,p).
        exp_variance : numpy array
            Array containing the explained variance of the modes, size (p,).

        '''
        if decomp_type == 'POD':
            # Compute the SVD of the scaled dataset
            print('Computing the SVD...')
            U, S, Vt = np.linalg.svd(X0, full_matrices=False)
            L = S**2    # Compute the eigenvalues
            exp_variance = 100*np.cumsum(L)/np.sum(L)

        return U, exp_variance

    def reduction(self, U, exp_variance, select_modes, n_modes):
        '''
        Return the reduced taylored basis.

        Parameters
        ----------
        U : numpy array
            The taylored basis to be reduced, size (n,p).
        exp_variance : numpy array
            The array containing the explained variance of the modes, size (p,).
        select_modes : str
            Method of modes selection.
        n_modes : int or float
            Parameter that controls the number of modes to be retained.

        Returns
        -------
        Ur : numpy array
            Reduced taylored basis, size (n,r).

        '''
        if select_modes == 'variance':
            # The r-order truncation is selected based on the amount of variance recovered
            for r in range(exp_variance.size):
                if exp_variance[r] > n_modes:
                    break
        elif select_modes == 'number':
            r = n_modes

        # Reduce the dimensionality
        Ur = U[:, :r]

        return Ur

    def optimal_placement(self, scale_type='standard', select_modes='variance', n_modes=99):
        '''
        Return the matrix C containing the optimal placement of the sensors.

        Parameters
        ----------
        scale_type : str, optional
            Type of scaling. The default is 'standard'.
        select_modes : str, optional
            Type of mode selection. The default is 'variance'.
        n_modes : int or float, optional
            Parameters that control the amount of modes retained. The default is 99.

        Returns
        -------
        C : numpy array
            The measurement matrix C obtained using QRCP decomposition, 
            size (s,n).

        '''
        n = self.X.shape[0]

        X0 = SPR.scale_data(self, scale_type)
        U, exp_variance = SPR.decomposition(self, X0)
        Ur = SPR.reduction(self, U, exp_variance, select_modes, n_modes)
        r = Ur.shape[1]

        # Calculate the QRCP
        print('Computing the QRCP and calculating C...')
        Q, R, P = la.qr(Ur.T, pivoting=True, mode='economic')
        p = r
        C = np.zeros((p, n))
        for j in range(p):
            C[j, P[j]] = 1

        return C

    def fit_predict(self, C, y, scale_type='standard', select_modes='variance', n_modes=99):
        '''
        Fit the taylored basis and the measurement matrix.
        Return the prediction vector.

        Parameters
        ----------
        C : numpy array
            The measurement matrix, size (s,n).
        y : numpy array
            The measurement vector, size (s,).
        scale_type : str, optional
            Type of scaling method. The default is 'standard'.
        select_modes : str, optional
            Type of mode selection. The default is 'variance'.
        n_modes : str, optional
            Parameter that controls the amount of modes retained. The default is 99.

        Returns
        -------
        x_rec : numpy array
            The predicted state of the system, size (n,).

        '''
        X0 = SPR.scale_data(self, scale_type)
        U, exp_variance = SPR.decomposition(self, X0)
        Ur = SPR.reduction(self, U, exp_variance, select_modes, n_modes)
        self.Ur = Ur
        
        Theta = C @ Ur
        self.Theta = Theta
        ar, res, rank, s = la.lstsq(Theta, y)
        x0_rec = Ur @ ar
        
        x_rec = SPR.unscale_data(self, x0_rec, scale_type)
        return x_rec
    
    def predict(self, y, scale_type='standard'):
        '''
        Return the prediction vector. 
        This method has to be used after fit_predict.

        Parameters
        ----------
        y : numpy array
            The measurement vector, size (s,).
        scale_type : int, optional
            The type of scaling. The default is 'standard'.

        Returns
        -------
        x_rec : numpy array
            The reconstructed error, size (n,).

        '''
        ar, res, rank, s = la.lstsq(self.Theta, y)
        x0_rec = self.Ur @ ar
        
        x_rec = SPR.unscale_data(self, x0_rec, scale_type)
        return x_rec


