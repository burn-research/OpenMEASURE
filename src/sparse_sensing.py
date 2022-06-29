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

import numpy as np
import scipy.linalg as la
from scipy.stats import qmc
import cvxpy as cp


class ROM():
    '''
    Class containing utilities for Reduced-Order-Models building.
    
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

    def decomposition(self, X0, decomp_type='POD', select_modes='variance', n_modes=99, 
                      solver='ECOS', abstol=1e-3, verbose=False):
        '''
        Return the taylored basis and the amount of variance of the modes.

        Parameters
        ----------
        X0 : numpy array
            The scaled data matrix to be decomposed, size (n,p).
        
        decomp_type : str, optional
            Type of decomposition. The default is 'POD'.
            If 'CPOD' it will calculate the constrained POD scores.
        
        select_modes : str, optional
            Method of modes selection.
        
        n_modes : int or float
            Parameter that controls the number of modes to be retained. If 
            select_modes = 'variance', n_modes can be a float between 0 and 100. 
            If select_modes = 'number', n_modes can be an integer between 1 and m.

        solver : str, optional
            Type of solver to use for solving the constrained minimization problem.
            Refer to the cvxpy documentation. The default is 'ECOS'.

        abstol : float, optional
            Absolute accuracy for the constrained solver used for CPOD. 
            Default is 1e-3.
            
        verbose : bool, optional
            If True, it prints the solver outputs. Default is False.

        Returns
        -------
        U : numpy array
            The taylored basis, size (n,p) or (n,r).
        
        A : numpy array
            The coefficient matrix, size (p,p) or or (p,r).
            
        exp_variance : numpy array
            Array containing the explained variance of the modes, size (p,) or (r,).

        '''
        if decomp_type == 'POD':
            # Compute the SVD of the scaled dataset
            U, S, Vt = np.linalg.svd(X0, full_matrices=False)
            A = np.matmul(np.diag(S), Vt).T
            L = S**2    # Compute the eigenvalues
            exp_variance = 100*np.cumsum(L)/np.sum(L)
            
            return U, A, exp_variance
        
        elif decomp_type == 'CPOD':
            # The constrained POD selects the POD scores by solving a constrained
            # minimization problem where the function to minimize is the
            # reconstruction error
            
            U, S, Vt = np.linalg.svd(X0, full_matrices=False)
            A = np.matmul(np.diag(S), Vt).T
            L = S**2    # Compute the eigenvalues
            exp_variance = 100*np.cumsum(L)/np.sum(L)
            Ur, Ar = self.reduction(U, A, exp_variance, select_modes, n_modes)
            r = Ar.shape[1]
            
            std_array = np.zeros(X0.shape[0])
            mean_array = np.zeros(X0.shape[0])
            for f in range(self.n_features):
                std_array[f*self.n_points:(f+1)*self.n_points] = self.std_vector[f]
                mean_array[f*self.n_points:(f+1)*self.n_points] = self.mean_vector[f]
            
            Gr = np.zeros_like(Ar)
            for i in range(Ar.shape[0]):
                g = cp.Variable(r)
                x0_tilde = Ur @ g
                x_tilde = cp.multiply(std_array,  (Ur @ g)) + mean_array
                
                objective = cp.Minimize(cp.pnorm(x0_tilde - X0[:,i], p=2))
                constrs = [x_tilde >= 0]
                prob = cp.Problem(objective, constrs)
                
                if verbose == True:
                    print(f'Calculating score {i+1}/{Ar.shape[0]}')
                    
                min_value = prob.solve(solver=solver, abstol=abstol, verbose=verbose)
                Gr[i,:] = g.value

            return Ur, Gr, exp_variance[:r]
            
        else:
            raise NotImplementedError('The decomposition method selected has not been '\
                                      'implemented yet')
                
        

    def reduction(self, U, A, exp_variance, select_modes, n_modes):
        '''
        Return the reduced taylored basis.

        Parameters
        ----------
        U : numpy array
            The taylored basis to be reduced, size (n,p).
        
        A : numpy array
            The coefficient matrix to be reduced, size (p,p).
        
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
        
        Ar : numpy array
            Reduced taylored basis, size (p,r)

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
        Ar = A[:, :r]

        return Ur, Ar

    def adaptive_sampling(self, P, scale_type='standard'):
        '''
        

        Parameters
        ----------
        P : numpy array
            The array contains the set of parameters used to build the X matrix, size (p,d).
        
        scale_type : str, optional
            Type of scaling. The default is 'standard'.

        Returns
        -------
        sample_new : numpy array
            The new sample, size (d,).

        '''
        
        X0 = self.scale_data(scale_type=scale_type)
        U, S, Vt = np.linalg.svd(X0, full_matrices=False)
        V = np.transpose(Vt)
        p = V.shape[0]
        
        Inf_basis = np.zeros((p,))
        Inf_relbasis = np.zeros((p,))
        for k in range(p):
            M = np.diag(S) @ (np.eye(p) - Vt[k,:] @ V[k,:])
            Un, Sn, Vnt = np.linalg.svd(M, full_matrices=False)
        
            Inf_ui_mj = np.zeros((p,)) 
            for i in range(p):        
                Inf_ui_mj[i] = 1/np.abs(Un[i,i]) - 1 # Influence of snapshot j on mode i
        
            Inf_basis[k] = np.sum(S * Inf_ui_mj) # Influence of snapshots on the basis
            
        for k in range(p):
            Inf_relbasis[k] = Inf_basis[k]/np.sum(Inf_basis) # Relative influence
       
        n_dim = P.shape[1]
        # LHS sampling with n=100*n_dim
        sampler = qmc.LatinHypercube(d=n_dim)
        q = 100*n_dim
        sample0 = sampler.random(n=q)
        
        sample = np.zeros_like(sample0)
        for d in range(n_dim):
            sample[:,d] = (P[:,d].max() - P[:,d].min()) * sample0[:,d] + P[:,d].min()
        
        Pot_basis = np.zeros((q,))
        for i in range(q):
            dist = np.linalg.norm(sample[i,:] - P, axis=1)
            j = np.argmin(dist)
            Pot_basis[i] = dist[j] * Inf_relbasis[j] # Potential of enrichment
        
        j_new = np.argmax(Pot_basis)
        sample_new = sample[j_new, :]
        return sample_new
        

class SPR(ROM):
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
    scale_vector(y, scale_type='standard')
        Scales the measurements matrix using the training information.

    optimal_placement(scale_type='standard', select_modes='variance', n_modes=99)
        Calculates the C matrix using QRCP decomposition.
    
    gem(Ur, n_sensors, verbose)
        Selects the best sensors' placement based on a greedy entropy maximization
        algorithm.
    
    fit_predict(C, y, scale_type='standard', select_modes='variance', 
                n_modes=99)
        Calculates the Theta matrix, then predicts ar and reconstructs x.
    
    predict(y, scale_type='standard'):
        Predicts ar and reconstructs x.
    
    '''

    def __init__(self, X, n_features):
        super().__init__(X, n_features)


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

    def gem(self, Ur, n_sensors, verbose):
        '''
        Selects the best sensors' placement based on a greedy entropy maximization
        algorithm.

        Parameters
        ----------
        Ur : numpy array
            Truncated basis used as target for entropy selection, size (n,r).
        
        n_sensors : int
            number of sensors.
        
        verbose : bool, optional
            If True, it will output relevant information on the entropy selection
            algorithm.

        Returns
        -------
        optimal_sensors : numpy array
            Array containing the index of the optimal sensors.

        '''
        
        # The scaling is used to ensure that the determinant of the covariance
        # matrix is greater than one.
        sigma = np.var(Ur, ddof=1, axis=1)
        coef = 1/np.sqrt(sigma.max())*2
        Ur_scl = Ur*coef

        sensor_list = []
        
        if verbose == True:
            header = ['# sensors', 'Hx', 'Hy', 'Htot', 'MI', 'det']
            print(f"{'-'*70} \n {header[0]:^10} {header[1]:^10} {header[2]:^10} {header[3]:^10} {header[4]:^10} {header[5]:^10} \n ")
    
        for s in range(n_sensors):
            hi = np.zeros((Ur.shape[0], ))
            for i, row in enumerate(Ur_scl):
                s_data = np.concatenate((Ur_scl[sensor_list], row[np.newaxis, :]), 
                                        axis=0)
                sigma = np.cov(s_data, rowvar=True)
                if s == 0:
                    det_sigma = sigma
                else:
                    det_sigma = np.linalg.det(sigma)

                if det_sigma <= 0:
                    hi[i] = -1e3

                else:
                    hi[i] = (s+1)/2 * (1 + np.log(2*np.pi)) + 0.5*np.log(det_sigma)

            sensor_list.append(np.argmax(hi))
            
            if verbose == True:
                sensor_counter = len(sensor_list)
                imax = np.argmax(hi)
                
                if sensor_counter == 1:
                    hx = hi.max()
                    print(f"{sensor_counter:^10} {hx:^10.2e} {'  -':^10} {'  -':^10} {'  -':^10} {'  -':^10}")
                
                elif len(sensor_list) == 2:
                    s_var_x = np.var(Ur_scl[sensor_list[0], :], ddof=1)
                    hx = 1/2 * (1 + np.log(2*np.pi)) + 0.5*np.log(s_var_x)
                    
                    s_var_y = np.var(Ur_scl[sensor_list[1], :], ddof=1)
                    hy = 1/2 * (1 + np.log(2*np.pi)) + 0.5*np.log(s_var_y)
                    
                    s_cov = np.cov(Ur_scl[sensor_list, :], ddof=1)
                    det_cov = np.linalg.det(s_cov)
                    htot = (s+1)/2 * (1 + np.log(2*np.pi)) + \
                        0.5*np.log(np.linalg.det(s_cov))
    
                    mi = 0.5 * np.log(s_var_x *
                                      s_var_y/np.linalg.det(s_cov))
                    
                    print(f"{sensor_counter:^10} {hx:^10.2e} {hy:^10.2e} {htot:^10.2e} {mi:^10.2e} {det_cov:^10.2e} ")

                elif len(sensor_list) > 2:
                    
                    s_var = np.var(Ur_scl[imax, :], ddof=1)
                    hy = 1/2 * (1 + np.log(2*np.pi)) + 0.5*np.log(s_var)
    
                    s_covx = np.cov(Ur_scl[sensor_list[:-1], :], ddof=1)
                    hx = (s)/2 * (1 + np.log(2*np.pi)) + \
                        0.5*np.log(np.linalg.det(s_covx))
    
                    s_cov = np.cov(Ur_scl[sensor_list, :], ddof=1)
                    det_cov = np.linalg.det(s_cov)
                    htot = (s+1)/2 * (1 + np.log(2*np.pi)) + \
                        0.5*np.log(np.linalg.det(s_cov))
    
                    mi = 0.5 * np.log(np.linalg.det(s_covx) *
                                      s_var/np.linalg.det(s_cov))
                    
                    print(f"{sensor_counter:^10} {hx:^10.2e} {hy:^10.2e} {htot:^10.2e} {mi:^10.2e} {det_cov:^10.2e}")
                    
        optimal_sensors = np.array(sensor_list)
        return optimal_sensors

    def optimal_placement(self, calc_type='qr', n_sensors=10, mask=None, scale_type='standard', 
                          select_modes='variance', n_modes=99, verbose=False):
        '''
        Return the matrix C containing the optimal placement of the sensors.

        Parameters
        ----------
        calc_type : str, optional
            Type of algorithm used to compute the C matrix. The available methods
            are 'qr' and 'gem'. The default is 'qr'.
        
        n_sensors : int, optional
            Number of sensors to calculate. Only used if the algorithm is 'gem'.
            Default is 10.
        
        mask : numpy array, optional
            A mask that indicates the region where to search for the optimal sensors.
            Only used if the algorithm is 'gem'. Default is None, meaning that 
            the entire volume is searched.
        
        scale_type : str, optional
            Type of scaling. The default is 'standard'.
        
        select_modes : str, optional
            Type of mode selection. The default is 'variance'. The available 
            options are 'variance' or 'number'.
        
        n_modes : int or float, optional
            Parameters that control the amount of modes retained. The default is 
            99, which represents 99% of the variance. If select_modes='number',
            n_modes represents the number of modes retained.
        
        verbose : bool, optional.
            If True, it will output the results of the computation used for the
            gem algorithm. Default is False.

        Returns
        -------
        C : numpy array
            The measurement matrix C obtained using QRCP decomposition, 
            size (s,n).

        '''
        n = self.X.shape[0]

        X0 = SPR.scale_data(self, scale_type)
        U, A, exp_variance = SPR.decomposition(self, X0)
        Ur, Ar = SPR.reduction(self, U, A, exp_variance, select_modes, n_modes)
        r = Ur.shape[1]

        if calc_type == 'qr':
            # Calculate the QRCP
            Q, R, P = la.qr(Ur.T, pivoting=True, mode='economic')
            s = r
            C = np.zeros((s, n))
            for j in range(s):
                C[j, P[j]] = 1
                
        elif calc_type == 'gem':
            if mask is not None:
                Ur[~mask, :] = 1
            
            P = SPR.gem(self, Ur, n_sensors, verbose)
            s = P.size
            C = np.zeros((s, n))
            for j in range(s):
                C[j, P[j]] = 1
        else:
            raise NotImplementedError('The sensor selection method has not been '\
                                      'implemented yet')
        
        return C

    def fit_predict(self, C, y, scale_type='standard', select_modes='variance', 
                    n_modes=99, method='OLS', solver='ECOS', abstol=1e-3, 
                    verbose=False):
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
            Type of scaling method. The default is 'standard'. Standard scaling is the 
            only scaling implemented for the 'COLS' method.
        
        select_modes : str, optional
            Type of mode selection. The default is 'variance'. The available 
            options are 'variance' or 'number'.
        
        n_modes : int or float, optional
            Parameters that control the amount of modes retained. The default is 
            99, which represents 99% of the variance. If select_modes='number',
            n_modes represents the number of modes retained.

        method : str, optional
            Method used to comupte the solution of the y0 = Theta * a linear 
            system. The choice are 'OLS' or 'COLS' which is constrained OLS. The
            default is 'OLS'.
        solver : str, optional
            Solver used for the constrained optimization problem. Only used if 
            the method selected is 'COLS'. The default is 'ECOS'.
        abstol : float, optional
            The absolute tolerance used to terminate the constrained optimization
            problem. The default is 1e-3.
        verbose : bool, optional
            If True, it displays the output from the constrained optimiziation 
            solver. The default is False

        Returns
        -------
        
        ar : numpy array
            The low-dimensional projection of the state of the system, size (r,)
        
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
        U, A, exp_variance = SPR.decomposition(self, X0)
        Ur, Ar = SPR.reduction(self, U, A, exp_variance, select_modes, n_modes)
        self.Ur = Ur
        
        Theta = C @ Ur
        self.Theta = Theta
        self.method = method
        self.solver = solver
        self.abstol = abstol
        self.verbose = verbose
        
        # calculate the condition number
        if Theta.shape[0] == Theta.shape[1]:
            U_theta, S_theta, V_thetat = np.linalg.svd(Theta)
            self.k = S_theta[0]/S_theta[-1]
        else:
            Theta_pinv = np.linalg.pinv(Theta)
            U_theta, S_theta, V_thetat = np.linalg.svd(Theta_pinv)
            self.k = S_theta[0]/S_theta[-1]
        
        ar, x_rec = SPR.predict(self, y)
        return ar, x_rec
    
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

        Returns
        -------
        ar : numpy array
            The low-dimensional projection of the state of the system, size (r,)
        x_rec : numpy array
            The reconstructed error, size (n,).

        '''
        if hasattr(self, 'Theta'):
            y0 = SPR.scale_vector(self, y, self.scale_type)
            
            if self.method == 'OLS':
                
                ar, res, rank, s = la.lstsq(self.Theta, y0)
                x0_rec = self.Ur @ ar
                
            elif self.method == 'COLS':
                r = self.Theta.shape[1]
                g = cp.Variable(r)
                
                std_array = np.zeros(self.Ur.shape[0])
                mean_array = np.zeros(self.Ur.shape[0])
                for f in range(self.n_features):
                    std_array[f*self.n_points:(f+1)*self.n_points] = self.std_vector[f]
                    mean_array[f*self.n_points:(f+1)*self.n_points] = self.mean_vector[f]
                
                g = cp.Variable(r)
                x_tilde = cp.multiply(std_array,  (self.Ur @ g)) + mean_array
                
                objective = cp.Minimize(cp.pnorm(y0 - self.Theta @ g, p=2))
                constrs = [x_tilde >= 0]
                prob = cp.Problem(objective, constrs)
                min_value = prob.solve(solver=self.solver, abstol=self.abstol, 
                                        verbose=self.verbose)
                ar = g.value
                x0_rec = self.Ur @ ar
            
            else:
                raise NotImplementedError('The prediction method selected has not been '\
                                          'implemented yet')
            
        
            x_rec = SPR.unscale_data(self, x0_rec, self.scale_type)
        else:
            raise AttributeError('The function fit_predict has to be called '\
                                 'before calling predict.')
            
        return ar, x_rec


if __name__ == '__main__':
    path = '../data/ROM/'
    X = np.load(path + 'X_2D_train.npy')
    P = np.genfromtxt(path + 'parameters_train.csv', delimiter=',', skip_header=1)
    xz = np.load(path + 'xz.npy')
    features = ['T', 'CH4', 'O2', 'CO2', 'H2O', 'H2', 'OH', 'CO', 'NOx']
    n_features = len(features)
    n_points = xz.shape[0]
    
    rom = ROM(X, n_features)
    X0 = rom.scale_data()
    U, A, exp_variance = rom.decomposition(X0, decomp_type='POD')
    Ur, Ar = rom.reduction(U, A, exp_variance, select_modes='variance', n_modes=99.5)
    
    spr = SPR(X, n_features)
    C = spr.optimal_placement(n_modes=99.5)

    x_test = X[:,0]
    
    y = np.zeros((C.shape[0],2))
    y[:,0] = C @ x_test

    for i in range(C.shape[0]):
        y[i,1] = np.argmax(C[i,:]) // n_points

    ap, x_rec_test = spr.fit_predict(C, y, n_modes=99.5, method='COLS', verbose=True)
    
    def NRMSE(prediction, observation):
        RMSE = np.sqrt(np.sum((prediction-observation)**2))/observation.size
        
        return RMSE/np.average(observation)
    
    error = NRMSE(x_rec_test, x_test)
    print(f'The NRMSE is {error:.5f}')

