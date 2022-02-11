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
    ----------
    X : numpy array
        data matrix of dimensions (n,p) where n = n_features * n_points and p
        is the number of operating conditions.
    
    n_features : int
        the number of features in the dataset (temperature, velocity, etc.).
        
    Methods
    ----------
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
        if type(X) is not np.ndarray:
            raise TypeError('The matrix X is not a numpy array.')
        elif type(n_features) is not int:
             raise TypeError('The parameter n_features is not an integer.')
        else:
           self.X = X
           self.n_features = n_features

    def scale_data(self, scale_type='standard'):
        '''
        Return the scaled data matrix. The default is to scale the data to 
        unitary variance.
        
        Parameters
        ----------
        scale_type : str, optional
            Type of scaling. The default is 'standard'. For now, it is the only
            method implemented.

        Returns
        -------
        X0 : numpy array
            Scaled data matrix.

        '''
        
        n = self.X.shape[0]
        self.n_points = n // self.n_features
        if n % self.n_features != 0:
            raise Exception('The number of rows of X is not a multiple of n_features')
            exit()
        
        X0 = np.zeros_like(self.X)

        if scale_type == 'standard':
            # Scale the matrix to unitary variance
            mean_vector = np.zeros((self.n_features))
            std_vector = np.zeros((self.n_features))
            for i in range(self.n_features):
                x = self.X[i*self.n_points:(i+1)*self.n_points, :]
                mean_vector[i] = np.average(x)
                std_vector[i] = np.std(x)
                X0[i*self.n_points:(i+1)*self.n_points, :] = (x -
                                                    mean_vector[i])/std_vector[i]

            self.mean_vector = mean_vector
            self.std_vector = std_vector
        else:
            raise NotImplementedError('The scaling method selected has not been '\
                                      'implemented yet')

        return X0

    def scale_vector(self, y, scale_type):
        '''
        Return the scaled measurement vector.

        Parameters
        ----------
        y : numpy array
            Measurement vector to scale, size (s,2). The first column contains
            the measurements, the second column contains which feature is 
            measured.
        scale_type : str
            Type of scaling.

        Returns
        -------
        y0: numpy array
            The scaled measurement vector.

        '''
        
        y0 = np.zeros((y.shape[0],))
        if scale_type == 'standard':
            for i in range(y0.shape[0]):
                y0[i] = (y[i,0] - self.mean_vector[int(y[i,1])]) / self.std_vector[int(y[i,1])]
        else:
            raise NotImplementedError('The scaling method selected has not been '\
                                      'implemented yet')
        
        return y0
            
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
                x[i*self.n_points:(i+1)*self.n_points] = self.std_vector[i] * \
                x0[i*self.n_points:(i+1)*self.n_points] + self.mean_vector[i]
        else:
            raise NotImplementedError('The scaling method selected has not been '\
                                      'implemented yet')  
        
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
            U, S, Vt = np.linalg.svd(X0, full_matrices=False)
            L = S**2    # Compute the eigenvalues
            exp_variance = 100*np.cumsum(L)/np.sum(L)
        else:
            raise NotImplementedError('The decomposition method selected has not been '\
                                      'implemented yet')
                
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
            Parameter that controls the number of modes to be retained. If 
            select_modes = 'variance', n_modes can be a float between 0 and 100. 
            If select_modes = 'number', n_modes can be an integer between 1 and m.
            

        Returns
        -------
        Ur : numpy array
            Reduced taylored basis, size (n,r).

        '''
        if select_modes == 'variance':
            if not 0 <= n_modes <= 100: 
                raise ValueError('The parameter n_modes is outside the[0-100] range.')
                
            # The r-order truncation is selected based on the amount of variance recovered
            for r in range(exp_variance.size):
                if exp_variance[r] > n_modes:
                    break
        elif select_modes == 'number':
            if not type(n_modes) is int:
                raise TypeError('The parameter n_modes is not an integer.')
            if not 1 <= n_modes <= U.shape[1]: 
                raise ValueError('The parameter n_modes is outside the [1-m] range.')
            r = n_modes
        else:
            raise ValueError('The select_mode value is wrong.')

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
        Q, R, P = la.qr(Ur.T, pivoting=True, mode='economic')
        s = r
        C = np.zeros((s, n))
        for j in range(s):
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
            The measurement vector, size (s,2). The first column contains
            the measurements, the second column contains which feature is 
            measured.
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
        if C.shape[0] != y.shape[0]:
            raise ValueError('The number of rows of C does not match the number' \
                             ' of rows of y.')
        if C.shape[1] != self.X.shape[0]:
            raise ValueError('The number of columns of C does not match the number' \
                              ' of rows of X.')
        if y.shape[1] != 2:
            raise ValueError('The y array has the wrong number of columns. y has' \
                              ' to have dimensions (s,2).')
        
        self.scale_type = scale_type
        X0 = SPR.scale_data(self, scale_type)
        U, exp_variance = SPR.decomposition(self, X0)
        Ur = SPR.reduction(self, U, exp_variance, select_modes, n_modes)
        self.Ur = Ur
        
        Theta = C @ Ur
        self.Theta = Theta
        
        x_rec = SPR.predict(self, y)
        return x_rec
    
    def predict(self, y):
        '''
        Return the prediction vector. 
        This method has to be used after fit_predict.

        Parameters
        ----------
        y : numpy array
            The measurement vector, size (s,2). The first column contains
            the measurements, the second column contains which feature is 
            measured.
        scale_type : int, optional
            The type of scaling. The default is 'standard'.

        Returns
        -------
        x_rec : numpy array
            The reconstructed error, size (n,).

        '''
        if hasattr(self, 'Theta'):
            y0 = SPR.scale_vector(self, y, self.scale_type)
            ar, res, rank, s = la.lstsq(self.Theta, y0)
            x0_rec = self.Ur @ ar
        
            x_rec = SPR.unscale_data(self, x0_rec, self.scale_type)
        else:
            raise AttributeError('The function fit_predict has to be called '\
                                 'before calling predict.')
            
        return x_rec



if __name__ == '__main__':
    X = np.random.rand(15, 5)
    spr = SPR(X, 5)
    
    C = spr.optimal_placement()
    U, e = spr.decomposition(X, decomp_type='POD')
    U = spr.reduction(U, e, select_modes='number', n_modes=3)
    
    y = np.array([[1.2,0],
                  [1.1,0],
                  [3.1, 0],
                 [4.1, 1]])
    
    x_rec = spr.fit_predict(C, y)
    x_rec = spr.predict(y)


