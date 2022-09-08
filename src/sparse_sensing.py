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
        
    xyz : numpy array
        3D position of the data in X, size (nx3).
        
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

    def __init__(self, X, n_features, xyz):
        '''    
        Parameters
        ----------
        X : numpy array
            Data matrix of dimensions (nxm) where n = n_features * n_points and m
            is the number of operating conditions.
        
        n_features : int
            The number of features in the dataset (temperature, velocity, etc.).

        xyz : numpy array
            3D position of the data in X, size (nx3).

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
           self.xyz = xyz
           

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
        
    xyz : numpy array
        3D position of the data in X, size (nx3).
        
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

    def __init__(self, X, n_features, xyz):
        super().__init__(X, n_features, xyz)


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

    def gem(self, Ur, n_sensors, mask, d_min, verbose):
        '''
        Selects the best sensors' placement based on a greedy entropy maximization
        algorithm.

        Parameters
        ----------
        Ur : numpy array
            Truncated basis used as target for entropy selection, size (n,r).
        
        n_sensors : int
            number of sensors.
        
        mask : numpy array
            A mask that indicates the region where to search for the optimal sensors.
        
        d_min : float
            Minimum distance between sensors. Only used if 'gem' method is selected.
        
        verbose : bool
            If True, it will output relevant information on the entropy selection
            algorithm.

        Returns
        -------
        optimal_sensors : numpy array
            Array containing the index of the optimal sensors.

        '''
        if mask is None:
            mask = np.ones((Ur.shape[0],), dtype=bool)
        
        index_org = np.arange(0,Ur.shape[0]) # original index before masking

        # The scaling is used to ensure that the determinant of the covariance
        # matrix is greater than one.
        sigma = np.var(Ur[mask], ddof=1, axis=1)
        coef = 1/np.sqrt(sigma.max())*2
        Ur_msk = Ur[mask]*coef
        Ur_scl = Ur*coef

        xyz_msk = np.tile(self.xyz, (self.n_features,1))[mask]
        index_msk = index_org[mask]

        sensor_list_glb = []
        sensor_list_loc = []
        
        if verbose == True:
            header = ['# sensors', 'sigma^2 y', 'sigma^2 y|a', 'Htot']
            print(f"{'-'*70} \n {header[0]:^10} {header[1]:^10} {header[2]:^10} {header[3]:^10} \n ")
        
        H_tot = 0
        sigma_coef = np.var(Ur_msk, ddof=1, axis=1)
        for s in range(n_sensors):
            if s == 0:
                # the first sensor is the one with maximum variance
                i_sensor = np.argmax(sigma_coef)
                sensor_list_loc.append(i_sensor)
                sensor_list_glb.append(index_msk[i_sensor])
                
                p_sensor = xyz_msk[i_sensor,:]
                d_sensor = np.linalg.norm(p_sensor-xyz_msk, axis=1)
                # ensure no sensor placed close to the others
                mask_d = d_sensor >= d_min
            
                if verbose == True:
                    print(f"{s+1:^10} {sigma_coef[sensor_list_glb[s]]:^10.2e} {'  -':^10} {'  -':^10}")
            
            else:
                Ur_msk = Ur_msk[mask_d]
                xyz_msk = xyz_msk[mask_d]
                index_msk = index_msk[mask_d]
                
                temp = np.zeros((index_msk.shape[0]))
                Sigma_aa = np.cov(Ur_scl[sensor_list_glb, :], ddof=1)
                
                if s == 1:
                    Sigma_aa_inv = 1/Sigma_aa
                else:
                    # the noise is added to ensure the singularity of the 
                    # covariance matrix
                    noise = 1e-5
                    Sigma_aa_inv = np.linalg.inv(Sigma_aa + noise*np.eye(s))
                
                for j in range(index_msk.size):
                    Sigma = np.cov(Ur_scl[sensor_list_glb, :], Ur_msk[j, :], ddof=1)
                    Sigma_ya = Sigma[-1, :-1]
                    Sigma_ay = Sigma[:-1, -1]
                    sigma2y = Sigma[-1,-1]
                    
                    # conditional variance
                    sigma2y_cond = sigma2y - np.dot(np.dot(Sigma_ya,Sigma_aa_inv),Sigma_ay)
                    temp[j] = sigma2y_cond
            
                # the sensor with the highest conditional variance is selected
                i_sensor = np.argmax(temp)
                sensor_list_loc.append(i_sensor)
                sensor_list_glb.append(index_msk[i_sensor])
                
                p_sensor = xyz_msk[i_sensor,:]
                d_sensor = np.linalg.norm(p_sensor-xyz_msk, axis=1)
                
                mask_d = d_sensor >= d_min
                
                # calculate the total conditional entropy
                H_tot += 0.5*np.log(temp[i_sensor]) + 0.5*(np.log(2*np.pi) + 1)
                
                if verbose == True:
                    print(f"{s+1:^10} {sigma_coef[sensor_list_glb[s]]:^10.2e} {temp[i_sensor]:^10.2e} {H_tot:^10.2e}")
            
            
        optimal_sensors = np.array(sensor_list_glb)
        return optimal_sensors

    def optimal_placement(self, calc_type='qr', n_sensors=10, mask=None, d_min=0., 
                          scale_type='standard', select_modes='variance', n_modes=99, 
                          verbose=False):
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
            Default is None, meaning that the entire volume is searched.
        
        d_min : float, optional
            Minimum distance between sensors. Only used if 'gem' method is selected.
            Default is 0.0.
        
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
            # Calculate the QRCP placement
            if mask is not None:
                Ur[~mask, :] = 0
            Q, R, P = la.qr(Ur.T, pivoting=True, mode='economic')
            s = r
            C = np.zeros((s, n))
            for j in range(s):
                C[j, P[j]] = 1
                
        elif calc_type == 'gem':
            # Calculate the GEM placement
            P = SPR.gem(self, Ur, n_sensors, mask, d_min, verbose)
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
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
#---------------------------------Plotting utilities--------------------------------------------------
    def sample_cmap(x):
        return plt.cm.jet((np.clip(x,0,1)))

    def plot_sensors(xz_sensors, features, mesh_outline):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(mesh_outline[:,0], mesh_outline[:,1], c='k', lw=0.5, zorder=1)
        
        features_unique = np.unique(xz_sensors[:,2])
        colors = np.zeros((features_unique.size,4))
        for i in range(colors.shape[0]):
            colors[i,:] = sample_cmap(features_unique[i]/len(features))
            
        for i, f in enumerate(features_unique):
            mask = xz_sensors[:,2] == f
            ax.scatter(xz_sensors[:,0][mask], xz_sensors[:,1][mask], color=colors[i,:], 
                       marker='x', s=15, lw=0.5, label=features[int(f)], zorder=2)

        
        ax.set_xlabel('$x (\mathrm{m})$', fontsize=8)
        ax.set_ylabel('$z (\mathrm{m})$', fontsize=8)
        eps = 1e-2
        ax.set_xlim(-eps, 0.35)
        ax.set_ylim(-0.15,0.7+eps)
        ax.set_aspect('equal')
        ax.legend(fontsize=8, frameon=False, loc='center right')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        wid = 0.3
        ax.xaxis.set_tick_params(width=wid)
        ax.yaxis.set_tick_params(width=wid)
        ax.set_xticks([0., 0.18, 0.35])
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.show()

    def plot_contours_tri(x, y, zs, cbar_label=''):
        triang = tri.Triangulation(x, y)
        triang_mirror = tri.Triangulation(-x, y)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6,6))
        
        z_min = np.min(zs)
        z_max = np.max(zs)
       
        n_levels = 12
        levels = np.linspace(z_min, z_max, n_levels)
        cmap_name= 'inferno'
        titles=['Original CFD','Predicted']
        
        for i, ax in enumerate(axs):
            if i == 0:
                ax.tricontourf(triang_mirror, zs[i], levels, vmin=z_min, vmax=z_max, cmap=cmap_name)
            else:
                ax.tricontourf(triang, zs[i], levels, vmin=z_min, vmax=z_max, cmap=cmap_name)
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False) 
            
            ax.set_aspect('equal')
            ax.set_title(titles[i])
            ax.set_xlabel('$x (\mathrm{m})$')
            if i == 0:
                ax.set_ylabel('$z (\mathrm{m})$')
        
        fig.subplots_adjust(bottom=0., top=1., left=0., right=0.85, wspace=0.02, hspace=0.02)
        start = axs[1].get_position().bounds[1]
        height = axs[1].get_position().bounds[3]
        
        cb_ax = fig.add_axes([0.9, start, 0.05, height])
        cmap = mpl.cm.get_cmap(cmap_name, n_levels)
        norm = mpl.colors.Normalize(vmin=z_min, vmax=z_max)
        
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, 
                    orientation='vertical', label=cbar_label)
        
        plt.show()
    
    path = '../data/ROM/'
    X = np.load(path + 'X_2D_train.npy')
    X_test = np.load(path + 'X_2D_test.npy')
    P = np.genfromtxt(path + 'parameters_train.csv', delimiter=',', skip_header=1)
    xz = np.load(path + 'xz.npy')
    xyz = np.zeros((xz.shape[0], 3))
    xyz[:,0] = xz[:,0]
    xyz[:,2] = xz[:,1]
    features = ['T', 'CH4', 'O2', 'CO2', 'H2O', 'H2', 'OH', 'CO', 'NOx']
    n_features = len(features)
    n_points = xz.shape[0]
    mesh_outline = np.genfromtxt(path + 'mesh_outline.csv', delimiter=',', skip_header=1)
    
    # rom = ROM(X, n_features, xyz)
    # X0 = rom.scale_data()
    # U, A, exp_variance = rom.decomposition(X0, decomp_type='POD')
    # Ur, Ar = rom.reduction(U, A, exp_variance, select_modes='variance', n_modes=99.5)
    
    spr = SPR(X, n_features, xyz)
    # C = spr.optimal_placement(n_modes=99.5)

    # x_test = X[:,0]
    
    # y = np.zeros((C.shape[0],2))
    # y[:,0] = C @ x_test

    # for i in range(C.shape[0]):
    #     y[i,1] = np.argmax(C[i,:]) // n_points

    # ap, x_rec_test = spr.fit_predict(C, y, n_modes=99.5, method='COLS', verbose=True)
    
    # def NRMSE(prediction, observation):
    #     RMSE = np.sqrt(np.sum((prediction-observation)**2))/observation.size
        
    #     return RMSE/np.average(observation)
    
    # error = NRMSE(x_rec_test, x_test)
    # print(f'The NRMSE is {error:.5f}')

    mask_pos = xz[:,0] < 0.1
    mask = np.ones((X.shape[0],), dtype=bool)
    keep = ['T', 'CO2']

    for f in features:
        ind = features.index(f)
        mask[ind*n_points:(ind+1)*n_points][mask_pos] = False

        if f not in keep:
            mask[ind*n_points:(ind+1)*n_points] = False

    for i in range(n_points):
        if np.any(X[i,:] > 1500):
            mask[i] = False

    C_qr = spr.optimal_placement(n_modes=99.5, mask=mask)
        
    # Get the sensors positions and features
    n_sensors = C_qr.shape[0]
    xz_sensors = np.zeros((n_sensors, 4))
    for i in range(n_sensors):
        index = np.argmax(C_qr[i,:])
        xz_sensors[i,:2] = xz[index % n_points, :]
        xz_sensors[i,2] = index // n_points

    plot_sensors(xz_sensors, features, mesh_outline)

    # Sample a test simulation using the optimal qr matrix
    y_qr = np.zeros((n_sensors,2))
    y_qr[:,0] = C_qr @ X_test[:,0]

    for i in range(n_sensors):
        y_qr[i,1] = np.argmax(C_qr[i,:]) // n_points

    # Fit the model and predict the low-dim vector (ap) and the high-dim solution (xp)
    ap, xp = spr.fit_predict(C_qr, y_qr)

    # Select the feature to plot
    str_ind = 'OH'
    ind = features.index(str_ind)

    plot_contours_tri(xz[:,0], xz[:,1], [X_test[ind*n_points:(ind+1)*n_points, 0], 
                    xp[ind*n_points:(ind+1)*n_points]], cbar_label=str_ind)

    C_gem = spr.optimal_placement(calc_type='gem', n_sensors=20, mask=None, d_min=0., 
                          scale_type='standard', select_modes='variance', n_modes=99.5, 
                          verbose=True)
    
    n_sensors = C_gem.shape[0]
    xz_sensors = np.zeros((n_sensors, 4))
    for i in range(n_sensors):
        index = np.argmax(C_gem[i,:])
        xz_sensors[i,:2] = xz[index % n_points, :]
        xz_sensors[i,2] = index // n_points

    plot_sensors(xz_sensors, features, mesh_outline)

    y_gem = np.zeros((n_sensors,2))
    y_gem[:,0] = C_gem @ X_test[:,0]

    for i in range(n_sensors):
        y_gem[i,1] = np.argmax(C_gem[i,:]) // n_points

    # Fit the model and predict the low-dim vector (ap) and the high-dim solution (xp)
    ap, xp = spr.fit_predict(C_gem, y_gem)

    # Select the feature to plot
    str_ind = 'OH'
    ind = features.index(str_ind)

    plot_contours_tri(xz[:,0], xz[:,1], [X_test[ind*n_points:(ind+1)*n_points, 0], 
                    xp[ind*n_points:(ind+1)*n_points]], cbar_label=str_ind)