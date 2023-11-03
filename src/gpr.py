'''
MODULE: gpr.py
@Authors:
    A. Procacci [1]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
@Contacts:
    alberto.procacci@ulb.be
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: alberto.procacci@ulb.be
'''

import copy
import numpy as np
import torch
import gpytorch
from scipy.stats import kurtosis
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
import sparse_sensing as sps

class ExactGPModel(gpytorch.models.ExactGP):
    '''
    Subclass used to build an exact GP Model inheriting from the ExactGP model.
    
    Attributes
    ----------
    X_train : numpy array
        matrix of dimensions (p,d) where p is the number of operating conditions
        and d is the number of design features.
    
    Y_train : numpy array
        matrix of dimensions (p,q) where p is the number of operating conditions
        and q is the number of retained coefficients.

    likelihood : gpytorch.likelihoods
        one dimensional likelihood from gpytorch.likelihoods.
        
    mean : gpytorch.means
        mean function gpytorch.means.
    
    kernel : gpytorch.kernels
        kernel function from gpytorch.kernels.
        
    Methods
    ----------
    forward(x)
        Return the multivariate distribution given the input x.
    
    '''
    
    def __init__(self, X_train, Y_train, likelihood, mean, kernel):
        super().__init__(X_train, Y_train, likelihood)
        
        self.mean_module = mean
        self.covar_module = kernel
            
    def forward(self, x):
        mean_x = self.mean_module(x)
        kernel_x = self.covar_module(x)
        return MultivariateNormal(mean_x, kernel_x)

class BatchIndipendentMultitaskGPModel(gpytorch.models.ExactGP):
    '''
    Subclass used to build a Multitask GP Model inheriting from the ExactGP model. 
    A Multitask model is needed when the target variable has multiple components.
    
    Attributes
    ----------
    X_train : numpy array
        matrix of dimensions (p,d) where p is the number of operating conditions
        and d is the number of design features.
    
    Y_train : numpy array
        matrix of dimensions (p,q) where p is the number of operating conditions
        and q is the number of retained coefficients.
        
    likelihood : gpytorch.likelihoods.MultitaskGaussianLikelihood
        multi-dimensional likelihood from gpytorch.likelihoods.MultitaskGaussianLikelihood.
        
    mean : gpytorch.means
        mean function gpytorch.means.
    
    kernel : gpytorch.kernels
        kernel function from gpytorch.kernels.    
        
    Methods
    ----------
    forward(x)
        Return the multivariate distribution given the input x.
    
    '''
    
    def __init__(self, X_train, Y_train, likelihood, mean, kernel):
        super().__init__(X_train, Y_train, likelihood)
    
        self.mean_module = mean
        self.covar_module = kernel
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        kernel_x = self.covar_module(x)
        return MultitaskMultivariateNormal.from_batch_mvn(MultivariateNormal(mean_x, 
                                                                             kernel_x))

class ConstrainedMultitaskGPModel(gpytorch.models.ExactGP):
    '''
    Subclass used to build a Multitask GP Model inheriting from the ExactGP model. 
    The added loss is used to introduce a physics-informed loss.
    
    Attributes
    ----------
    X_train : numpy array
        matrix of dimensions (p,d) where p is the number of operating conditions
        and d is the number of design features.
    
    Y_train : numpy array
        matrix of dimensions (p,q) where p is the number of operating conditions
        and q is the number of retained coefficients.
        
    likelihood : gpytorch.likelihoods.MultitaskGaussianLikelihood
        multi-dimensional likelihood from gpytorch.likelihoods.MultitaskGaussianLikelihood.
        
    mean : gpytorch.means
        mean function gpytorch.means.
    
    kernel : gpytorch.kernels
        kernel function from gpytorch.kernels.  

    AddedLoss : class inheriting from gpytorch.mlls.AddedLossTerm  
        class used to define the added loss.

    Methods
    ----------
    forward(x)
        Return the multivariate distribution given the input x.
    
    '''
    
    def __init__(self, X_train, Y_train, likelihood, mean, kernel, AddedLoss):
        super().__init__(X_train, Y_train, likelihood)
        
        self.X_train = X_train
        self.Y_train = Y_train
         
        self.mean_module = mean
        self.covar_module = kernel

        self.AddedLoss = AddedLoss
        self.register_added_loss_term('added_loss')

    def forward(self, inputs, **kwargs):
        mean_x = self.mean_module(inputs)
        kernel_x = self.covar_module(inputs)

        if self.training and 'output' in kwargs:        
            added_loss_term = self.AddedLoss(kwargs)
            self.update_added_loss_term('added_loss', added_loss_term)

        return MultitaskMultivariateNormal.from_batch_mvn(MultivariateNormal(mean_x, 
                                                                             kernel_x))

class GPR(sps.ROM):
    '''
    Class used for building a GPR-based ROM.
    
    Attributes
    ----------
    X : numpy array
        data matrix of dimensions (n,p) where n = n_features * n_points and p
        is the number of operating conditions.
        
    n_features : int
        the number of features in the dataset (temperature, velocity, etc.).
    
    xyz : numpy array
        3D position of the data in X, size (nx3).
    
    P : numpy array
        Design features matrix of dimensions (p,d) where p is the number of 
        operating conditions.
    
    gpr_type : str, optional.
        If 'SingleTask', a GPR model is trained for each of the r latent dimensions.
        If 'MultiTask', the a single multitask GPR is trained for all the latent dimensions.
        The default is 'SingleTask'.     
        
    Methods
    ----------
    scale_GPR_data(P, scale_type)
        Scales the input parameters.
        
    unscale_coefficients(scale_type)
        Unscale the coefficients.
    
    fit(scaleX_type='std', scaleP_type='std', select_modes='variance', decomp_type='POD', 
            n_modes=99, max_iter=1000, rel_error=1e-5, lr=0.1, solver='ECOS', abstol=1e-3, verbose=False)
        Trains the GPR model.
    
    predict(P_star)
        Predicts the low-dimensional array of vectors Ar and its uncertainty Ar_sigma.

    update(P_new, A_new, A_sigma_new=None, retrain=False, verbose=False)
        Updates the model with new data.
    '''

    def __init__(self, X, n_features, xyz, P, gpr_type='SingleTask'):
        super().__init__(X, n_features, xyz)
        self.P = P
        self.gpr_type = gpr_type
        
        if P.shape[0] != X.shape[1]:
            raise Exception(f'The number of parameters ({P.shape[0]}) is different' \
                            f' from the number of columns of X ({X.shape[1]})')
            exit()


    def _train_loop(self, model, likelihood, P0_torch, Vr_torch, i):
        model.train()
        likelihood.train()
    
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        mll = ExactMarginalLogLikelihood(likelihood, model)
        loss_old = 1e10
        e = 1e10
        j = 0
        while (e > self.rel_error) and (j < self.max_iter):
            optimizer.zero_grad()
            output = model(P0_torch)
            loss = -mll(output, Vr_torch)
            loss.backward()
            e = torch.abs(loss - loss_old).item()
            loss_old = loss
            if self.verbose == True:
                if self.gpr_type == 'SingleTask':
                    noise_avg = np.mean(model.likelihood.noise.detach().numpy())
                    print(f'Iter {j+1:d}/{self.max_iter:d} - Mode: {i+1:d}/{self.r:d} - Loss: {loss.item():.2e} - ' \
                    f'Mean noise: {noise_avg:.2e}')
                else:
                    print(f'Iter {j+1:d}/{self.max_iter:d} - Loss: {loss.item():.2e} - ' \
                         f'Noise: {model.likelihood.noise.item():.2e}')
                    
            optimizer.step()
            j += 1

        Vr_sigma = output.stddev.detach().numpy()
        
        return model, likelihood, Vr_sigma

    def scale_GPR_data(self, P, scale_type):
        '''
        Return the scaled input and target for the GPR model.

        Parameters
        ----------
        P : numpy array
            Data matrix to scale, size (p,d). 
        
        scale_type : str
            Type of scaling.

        Returns
        -------
        P0: numpy array
            The scaled measurement vector.

        '''
        
        P_cnt = np.zeros_like(P)
        P_scl = np.zeros_like(P)
        
        for i in range(P.shape[1]):
            x = P[:,i]
            P_cnt[:,i] = np.mean(x)
            
            if scale_type == 'std':
                P_scl[:,i] = np.std(x)
            
            elif scale_type == 'none':
                P_scl[:,i] = 1.
            
            elif scale_type == 'pareto':
                P_scl[:,i] = np.sqrt(np.std(x))
            
            elif scale_type == 'vast':
                scl_factor = np.std(x)**2/np.average(x)
                P_scl[:,i] = scl_factor
            
            elif scale_type == 'range':
                scl_factor = np.max(x) - np.min(x)
                P_scl[:,i] = scl_factor
                
            elif scale_type == 'level':
                P_scl[:,i] = np.average(x)
                
            elif scale_type == 'max':
                P_scl[:,i] = np.max(x)
            
            elif scale_type == 'variance':
                P_scl[:,i] = np.var(x)
            
            elif scale_type == 'median':
                P_scl[:,i] = np.median(x)
            
            elif scale_type == 'poisson':
                scl_factor = np.sqrt(np.average(x))
                P_scl[:,i] = scl_factor
            
            elif scale_type == 'vast_2':
                scl_factor = (np.std(x)**2 * kurtosis(x, None)**2)/np.average(x)
                P_scl[:,i] = scl_factor
            
            elif scale_type == 'vast_3':
                scl_factor = (np.std(x)**2 * kurtosis(x, None)**2)/np.max(x)
                P_scl[:,i] = scl_factor
            
            elif scale_type == 'vast_4':
                scl_factor = (np.std(x)**2 * kurtosis(x, None)**2)/(np.max(x)-np.min(x))
                P_scl[:,i] = scl_factor
            
            elif scale_type == 'l2-norm':
                scl_factor = np.linalg.norm(x.flatten())
                P_scl[:,i] = scl_factor
            
            else:
                raise NotImplementedError('The scaling method selected has not been '\
                                          'implemented yet')
                    
        self.P_cnt = P_cnt
        self.P_scl = P_scl
        P0 = (P - P_cnt)/P_scl
        return P0    

    def fit(self, scaleX_type='std', scaleP_type='std', decomp_type='POD', limits=None, select_modes='variance', 
            n_modes=99, solver='ECOS', abstol=1e-3, verbose=False, basis=None):
        '''
        Fit the GPR model.
        Return the model and likelihood.

        Parameters
        ----------
        scaleX_type : str, optional
            Type of scaling method for the data matrix. The default is 'std'.
        
        scaleP_type : str, optional
            Type of scaling method for the parameters. The default is 'std'.
        
        decomp_type : str, optional
            Type of decomposition. The default is 'POD'. The available options
            are 'POD' and 'CPOD'
        
        limits : list, optional
            List of minimum and maximum constraints. Required if decomp_type is 'CPOD'.
            The default is None.
        
        select_modes : str, optional
            Type of mode selection. The default is 'variance'. The available 
            options are 'variance' or 'number'.
            
        n_modes : int or float, optional
            Parameters that control the amount of modes retained. The default is 
            99, which represents 99% of the variance. If select_modes='number',
            n_modes represents the number of modes retained.
        
        solver : str, optional
            Type of solver to use for solving the constrained minimization problem.
            Refer to the cvxpy documentation. The default is 'ECOS'.

        abstol : float, optional
            Absolute accuracy for the constrained solver used for CPOD. 
            Default is 1e-3.

        verbose : bool, optional
            If True, it will print informations on the training of the hyperparameters.
            The default is False.
        
        basis : tuple, optional
            If it is not None, the tuple contains the matrices Ur and Ar, wich are
            set avoiding the computation of the decomposition.
            
        '''
        
        self.scaleX_type = scaleX_type
        self.scaleP_type = scaleP_type
        self.select_modes = select_modes
        self.decomp_type = decomp_type, 
        self.n_modes = n_modes
        self.solver = solver
        self.abstol = abstol
        self.verbose = verbose

        X0 = self.scale_data(scaleX_type)
        
        if basis is None:
            Ur, Ar, exp_variance_r = self.decomposition(X0, decomp_type, limits, select_modes, n_modes, 
                                                    solver, abstol, verbose)
        else:
            Ur = basis[0]
            Ar = basis[1]

        self.Ur = Ur
        self.Ar = Ar
        self.r = Ar.shape[1]
        self.d = self.P.shape[1]
        
        # Get the singular values and the orthonormal basis
        Vr = np.zeros_like(Ar)
        Sigma_r = np.zeros((self.r,))
        for i in range(self.r):
            Sigma_r[i] = np.linalg.norm(Ar[:,i])
            Vr[:,i] = Ar[:,i]/Sigma_r[i]
        
        self.Sigma_r = Sigma_r
        P0 = GPR.scale_GPR_data(self, self.P, scaleP_type)
        
        self.P0 = P0
        self.Vr = Vr
    
    def train(self, mean=None, kernel=None, likelihood=None, max_iter=1000, 
              rel_error=1e-5, lr=0.1, verbose=False):
        '''
        Train the GPR model.
        Return the model and likelihood.

        Parameters
        ----------
        mean : gpytorch.means, optional.
            The mean passed to the GPR model. The default is means.ConstantMean.

        kernel : gpytorch.kernels, optional.
            The kernel used for the computation of the covariance matrix. The default
            is the Matern kernel.

        likelihood : gpytorch.likelihoods, optional
            The likelihood passed to the GPR model. If gpr_type='SingleTask', the default 
            is GaussianLikelihood(). If gpr_type='MultiTask', the MultitaskGaussianLikelihood()
            is the only option.
        
        max_iter : int, optional
            Maximum number of iterations to train the hyperparameters. The default
            is 1000.
            
        rel_error : float, optional
            Minimum relative error below which the training of hyperparameters is
            stopped. The default is 1e-5.
        
        lr : float, optional
            Learning rate of the Adam optimizer used for minimizing the negative log 
            likelihood. The default is 0.1.

        verbose : bool, optional
            If True, it will print informations on the training of the hyperparameters.
            The default is False.
            

        Returns
        -------
        model : gpytorch.models
            The trained gpr model.

        likelihood : gpytorch.likelihoods.
            The trained likelihood.
        
        '''
        
        self.max_iter = max_iter
        self.rel_error = rel_error 
        self.lr = lr
        self.verbose = verbose

        P0_torch = torch.from_numpy(self.P0).contiguous().double()
        Vr_torch = torch.from_numpy(self.Vr).contiguous().double()
            
        models = []
        likelihoods = []

        self.mean = mean
        self.kernel = kernel
        self.likelihood = likelihood

        if self.gpr_type == 'MultiTask':
            
            if mean is None:
                self.mean = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.r]))
            
            if kernel is None:
                self.kernel = gpytorch.kernels.MaternKernel(batch_shape=torch.Size([self.r]))
            
            if likelihood is None:
                self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.r)
            
            likelihoods.append(self.likelihood)
            models.append(BatchIndipendentMultitaskGPModel(P0_torch, Vr_torch, likelihoods[0], 
                                                           self.mean, self.kernel))
            models[0].double()
            likelihoods[0].double()
            
            models[0], likelihoods[0], Vr_sigma = self._train_loop(models[0], likelihoods[0], P0_torch, Vr_torch, 0)
        
        else:
            if mean is None:
                self.mean = gpytorch.means.ConstantMean()
            
            if kernel is None:
                self.kernel = gpytorch.kernels.MaternKernel(2.5)
            
            if likelihood is None:
                self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

            Vr_sigma = np.zeros_like(self.Vr)

            for i in range(self.r):
                likelihood = copy.deepcopy(self.likelihood)
                mean = copy.deepcopy(self.mean)
                kernel = copy.deepcopy(self.kernel)
                model = ExactGPModel(P0_torch, Vr_torch[:,i], likelihood, mean, kernel)
                
                model.double()
                likelihood.double()

                model, likelihood, Vr_sigma[:, i] = self._train_loop(model, likelihood, P0_torch, Vr_torch[:,i], i)

                models.append(model)
                likelihoods.append(likelihood)

        self.Vr_sigma = Vr_sigma
        self.models = models
        self.likelihoods = likelihoods
        
        return models, likelihoods
    
    def predict(self, P_star):
        '''
        Return the prediction vector. 
        This method has to be used after fit.

        Parameters
        ----------
        P_star : numpy array
            The set of design features to evaluate the prediction, size (n_p,d).

        Returns
        -------
        A_pred : numpy array
            The low-dimensional projection of the state of the system, size (n_p,r)
        
        A_sigma : numpy array
            Uncertainty in the prediction, size (n_p,r)

        '''
        
        if hasattr(self, 'models'):
            if P_star.ndim < 2:
                P_star = P_star[np.newaxis, :]

            n_p = P_star.shape[0]

            P0_star = np.zeros_like(P_star)
            for i in range(P_star.shape[1]):
                P0_star[:,i] = (P_star[:,i] - self.P_cnt[0,i]) / self.P_scl[0,i]
            
            P0_star_torch = torch.from_numpy(P0_star).contiguous().double()
            
            if self.gpr_type == 'MultiTask':    

                # Set into eval mode
                self.models[0].eval()
                self.likelihoods[0].eval()

                observed_pred = self.likelihoods[0](self.models[0](P0_star_torch))
                V_pred = observed_pred.mean.detach().numpy()
                V_sigma = observed_pred.stddev.detach().numpy()
                
            else:
                V_pred = np.zeros((n_p, self.r))
                V_sigma = np.zeros((n_p, self.r))

                for i in range(self.r):
                    # Set into eval mode
                    self.models[i].eval()
                    self.likelihoods[i].eval()

                    observed_pred = self.likelihoods[i](self.models[i](P0_star_torch))
                    V_pred[:,i] = observed_pred.mean.detach().numpy()
                    V_sigma[:,i] = observed_pred.stddev.detach().numpy()
                    
        else:
            raise AttributeError('The function fit has to be called '\
                                  'before calling predict.')
        
        A_pred = np.zeros_like(V_pred)
        A_sigma = np.zeros_like(V_sigma)
        for i in range(self.r):
            A_pred[:,i] = self.Sigma_r[i] * V_pred[:,i]
            A_sigma[:,i] = self.Sigma_r[i] * V_sigma[:,i]
        
        return A_pred, A_sigma
    
    def update(self, P_new, A_new, A_sigma_new=None, retrain=False, verbose=False):     
        '''
        Updates the model with new data.

        Parameters
        ----------
        P_new : numpy array
            The set of design features of the new data, size (n_p_new, d).

        A_new : numpy array
            The set of new data in the low dimensional space, size (n_p_new, r).

        A_sigma_new : numpy array, optional
            The uncertainty of the new data. The default is None.
        
        retrain : bool, optional
            If True, the hyperparameters are retrained. The default is False.

        verbose : bool, optional
            If True, it will print informations on the training of the hyperparameters.
            The default is False.

        '''
        
        self.verbose = verbose
        
        # Create new set of parameters
        P0_new = np.zeros_like(P_new)
        for i in range(P_new.shape[1]):
            P0_new[:,i] = (P_new[:,i] - self.P_cnt[0,i]) / self.P_scl[0,i]
            
        P0_tot = np.concatenate([self.P0, P0_new], axis=0)
        P0_tot_torch = torch.from_numpy(P0_tot).contiguous().double()
        
        # Create new set of observations
        Vr_new = np.zeros_like(A_new)
        for i in range(self.r):
            # V_sigma_train[:,i] = A_sigma_train[:,i]/self.Sigma_r[i]
            Vr_new[:,i] = A_new[:,i]/self.Sigma_r[i]
        
        Vr_tot = np.concatenate([self.Vr, Vr_new], axis=0)
        Vr_tot_torch = torch.from_numpy(Vr_tot).contiguous().double()
        
        # If the uncertainty is passed, create new set of uncertainties
        if A_sigma_new is not None:
            Vr_sigma_new = np.zeros_like(A_sigma_new)
            for i in range(self.r):
                Vr_sigma_new[:,i] = A_sigma_new[:,i]/self.Sigma_r[i]

            Vr_sigma_tot = np.concatenate([self.Vr_sigma, Vr_sigma_new], axis=0)
            Vr_sigma_tot_torch = torch.from_numpy(Vr_sigma_tot).contiguous().double()
            self.Vr_sigma = np.zeros_like(Vr_sigma_tot)
                
        
        if self.gpr_type == 'MultiTask':

                self.models[0].set_train_data(P0_tot_torch, Vr_tot_torch, strict=False)
                
                if retrain:
                    temp = self._train_loop(self.models[0], self.likelihoods[0], P0_tot_torch, Vr_tot_torch, 0)
                    self.models[0], self.likelihoods[0], self.Vr_sigma = temp

        else:
            for i in range(self.r):

                self.models[i].set_train_data(P0_tot_torch, Vr_tot_torch[:,i], strict=False)
                
                if retrain:
                    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(Vr_sigma_tot_torch[:, i]**2)
                    self.models[i].likelihood = likelihood
                    
                    temp = self._train_loop(self.models[i], self.likelihoods[i], P0_tot_torch, Vr_tot_torch[:,i], i)
                    self.models[i], self.likelihoods[i], self.Vr_sigma[:,i] = temp


class CGPR(GPR):
    
    def __init__(self, X, n_features, xyz, P, P_cstr, AddedLoss):
        
        super().__init__(X, n_features, xyz, P, 'MultiTask')
        self.P_cstr = P_cstr
        self.AddedLoss = AddedLoss
    
    def train(self, mean=None, kernel=None, likelihood=None, max_iter=1000, 
              rel_error=1e-5, lr=0.1, verbose=False):
        '''
        Train the GPR model.
        Return the model and likelihood.

        Parameters
        ----------
        mean : gpytorch.means, optional.
            The mean passed to the GPR model. The default is means.ConstantMean.

        kernel : gpytorch.kernels, optional.
            The kernel used for the computation of the covariance matrix. The default
            is the Matern kernel.

        likelihood : gpytorch.likelihoods, optional
            The likelihood passed to the GPR model. If gpr_type='SingleTask', the default 
            is GaussianLikelihood(). If gpr_type='MultiTask', the MultitaskGaussianLikelihood()
            is the only option.
        
        max_iter : int, optional
            Maximum number of iterations to train the hyperparameters. The default
            is 1000.
            
        rel_error : float, optional
            Minimum relative error below which the training of hyperparameters is
            stopped. The default is 1e-5.
        
        lr : float, optional
            Learning rate of the Adam optimizer used for minimizing the negative log 
            likelihood. The default is 0.1.

        verbose : bool, optional
            If True, it will print informations on the training of the hyperparameters.
            The default is False.
            

        Returns
        -------
        model : gpytorch.models
            The trained gpr model.

        likelihood : gpytorch.likelihoods.
            The trained likelihood.
        
        '''
        
        self.max_iter = max_iter
        self.rel_error = rel_error 
        self.lr = lr
        self.verbose = verbose

        P0_torch = torch.from_numpy(self.P0).contiguous().double()
        Vr_torch = torch.from_numpy(self.Vr).contiguous().double()
        
        P0_cstr = np.zeros_like(self.P_cstr)
        for i in range(P0_cstr.shape[1]):
            P0_cstr[:,i] = (self.P_cstr[:,i] - self.P_cnt[0,i]) / self.P_scl[0,i]

        P0_tot = np.concatenate([self.P0, P0_cstr], axis=0)
        self.P0_tot = P0_tot
        P0_tot_torch = torch.from_numpy(P0_tot).contiguous().double()

        models = []
        likelihoods = []

        self.mean = mean
        self.kernel = kernel
        self.likelihood = likelihood
    
        if mean is None:
            self.mean = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.r]))
        
        if kernel is None:
            self.kernel = gpytorch.kernels.MaternKernel(batch_shape=torch.Size([self.r]))
        
        if likelihood is None:
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.r)
        
        likelihoods.append(self.likelihood)
        models.append(ConstrainedMultitaskGPModel(P0_torch, Vr_torch, likelihoods[0], 
                                                  self.mean, self.kernel, self.AddedLoss))
        
        models[0].double()
        likelihoods[0].double()
        
        models[0], likelihoods[0], Vr_sigma = self._train_loop(models[0], likelihoods[0], P0_torch, Vr_torch, P0_tot_torch)
        
        self.Vr_sigma = Vr_sigma
        self.models = models
        self.likelihoods = likelihoods
        
        return models, likelihoods

    def compute_mll(self, mean=None, kernel=None, likelihood=None):
        '''
        Computes the mll loss and the prediction on the training data.
        This is useful to set the coefficients of the added loss.

        Parameters
        ----------
        mean : gpytorch.means, optional.
            The mean passed to the GPR model. The default is means.ConstantMean.

        kernel : gpytorch.kernels, optional.
            The kernel used for the computation of the covariance matrix. The default
            is the Matern kernel.

        likelihood : gpytorch.likelihoods, optional
            The likelihood passed to the GPR model. If gpr_type='SingleTask', the default 
            is GaussianLikelihood(). If gpr_type='MultiTask', the MultitaskGaussianLikelihood()
            is the only option.

        Returns
        -------
        loss_mll : numpy array
            The marginal log likelihood of the training data.

        Vr_pred_train : torch tensor
            The prediction on the training data.
        
        '''
        P0_torch = torch.from_numpy(self.P0).contiguous().double()
        Vr_torch = torch.from_numpy(self.Vr).contiguous().double()
        
        P0_cstr = np.zeros_like(self.P_cstr)
        for i in range(P0_cstr.shape[1]):
            P0_cstr[:,i] = (self.P_cstr[:,i] - self.P_cnt[0,i]) / self.P_scl[0,i]

        P0_tot = np.concatenate([self.P0, P0_cstr], axis=0)
        P0_tot_torch = torch.from_numpy(P0_tot).contiguous().double()

        if mean is None:
            mean = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.r]))
        
        if kernel is None:
            kernel = gpytorch.kernels.MaternKernel(batch_shape=torch.Size([self.r]))
        
        if likelihood is None:
            likelihood = MultitaskGaussianLikelihood(num_tasks=self.r)
        
        model = (ConstrainedMultitaskGPModel(P0_torch, Vr_torch, likelihood, 
                                             mean, kernel, self.AddedLoss))

        model.double()
        likelihood.double()

        model.train()
        likelihood.train()

        loss_mll = likelihood(model(P0_torch)).log_prob(Vr_torch).detach().numpy()

        model.eval()
        model.likelihood.eval()
        
        output = model.likelihood(model(P0_tot_torch))
        Vr_pred_train = output.mean.detach()

        return loss_mll, Vr_pred_train

    def _train_loop(self, model, likelihood, P0_torch, Vr_torch, P0_tot_torch):
        
        model.train()
        likelihood.train()
    
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        mll = ExactMarginalLogLikelihood(likelihood, model)
        loss_old = 1e10
        e = 1e10
        j = 0
        while (e > self.rel_error) and (j < self.max_iter):
            optimizer.zero_grad()
            
            model.eval()
            likelihood.eval()
        
            output_cstr = likelihood(model(P0_tot_torch))
        
            model.train()
            likelihood.train()
            
            loss_ml = model.likelihood(model(P0_torch)).log_prob(model.Y_train).detach()
            output = model(P0_torch, output=output_cstr, loss_ml=loss_ml, verbose=self.verbose)
            
            loss = -mll(output, Vr_torch)
            loss.backward()
            
            e = torch.abs(loss - loss_old).item()
            loss_old = loss
            if self.verbose == True:
                
                print(f'Iter {j+1:d}/{self.max_iter:d} - Loss: {loss.item():.2e} - ' \
                    f'Noise: {model.likelihood.noise.item():.2e}')
                
            optimizer.step()
            j += 1

            Vr_sigma = output.stddev.detach().numpy()

        return model, likelihood, Vr_sigma

if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri


    # Replace this with the path where you saved the data directory
    path = '../data/ROM/'

    # This is a n x m matrix where n = 165258 is the number of cells times the number of features
    # and m = 41 is the number of simulations.
    X_train = np.load(path + 'X_2D_train.npy')

    # This is a n x 4 matrix containing the 4 testing simulations
    X_test = np.load(path + 'X_2D_test.npy')

    features = ['T', 'CH4', 'O2', 'CO2', 'H2O', 'H2', 'OH', 'CO', 'NOx']
    n_features = len(features)

    # This is the file containing the x,z positions of the cells
    xz = np.load(path + 'xz.npy')
    n_cells = xz.shape[0]
    
    # Create the x,y,z array
    xyz = np.zeros((n_cells, 3))
    xyz[:,0] = xz[:,0]
    xyz[:,2] = xz[:,1]

    # This reads the files containing the parameters (D, H2, phi) with which 
    # the simulation were computed
    P_train = np.genfromtxt(path + 'parameters_train.csv', delimiter=',', skip_header=1)
    P_test = np.genfromtxt(path + 'parameters_test.csv', delimiter=',', skip_header=1)

    # Load the outline the mesh (for plotting)
    mesh_outline = np.genfromtxt(path + 'mesh_outline.csv', delimiter=',', skip_header=1)

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
        cmap_name = 'inferno'
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
        cmap = mpl.cm.get_cmap(cmap_name)
        norm = mpl.colors.Normalize(vmin=z_min, vmax=z_max)
        
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, 
                    orientation='vertical', label=cbar_label)
        
        plt.show()
    #------------------------------------GPR ROM--------------------------------------------------
    # Create the gpr object
    
    gpr = GPR(X_train, n_features, xyz, P_train, gpr_type='MultiTask')
    
    # Calculates the POD coefficients ap and the uncertainty for the test simulations
    gpr.fit()
    models, likelihoods = gpr.train(max_iter=10, verbose=True, lr=0.01)
    Ap, Sigmap = gpr.predict(P_test)
    
    # Reconstruct the high-dimensional state from the POD coefficients
    Xp = gpr.reconstruct(Ap)

    # Select the feature to plot
    str_ind = 'OH'
    ind = features.index(str_ind)

    x_test = X_test[ind*n_cells:(ind+1)*n_cells,3]
    xp_test = Xp[ind*n_cells:(ind+1)*n_cells, 3]

    # plot_contours_tri(xz[:,0], xz[:,1], [x_test, xp_test], cbar_label=str_ind)

    # gpr.update(P_test, Ap, Sigmap, retrain=True, verbose=True)
    # Ap, Sigmap = gpr.predict(P_test)
    # Xp = gpr.reconstruct(Ap)

    # x_test = X_test[ind*n_cells:(ind+1)*n_cells,3]
    # xp_test = Xp[ind*n_cells:(ind+1)*n_cells, 3]

    # plot_contours_tri(xz[:,0], xz[:,1], [x_test, xp_test], cbar_label=str_ind)

    #------------------------------------CGPR ROM--------------------------------------------------
    # Create the gpr object
    
    class AddedLoss(gpytorch.mlls.AddedLossTerm):
        def __init__(self, kwargs):
            self.output = kwargs.get('output')
            self.loss_ml = kwargs.get('loss_ml')
            self.verbose = kwargs.get('verbose')
            
        def loss(self):
            loss_l = limit_loss(self.output.mean) 
                        
            return loss_l

    def g(x):
        return torch.maximum(torch.tensor(0),x)

    def scale_limits(limits):
        X_cnt = torch.from_numpy(cgpr.X_cnt)
        X_scl = torch.from_numpy(cgpr.X_scl)
        n_points = cgpr.n_points
        
        limits0 = []
        for limit in limits:
            limit_torch = torch.from_numpy(limit)
            limit0 = torch.zeros((X_cnt.shape[0],))

            for i in range(cgpr.n_features):
                limit0[i*n_points:(i+1)*n_points] = ((limit[i] - X_cnt[i*n_points:(i+1)*n_points, 0])
                                                    /X_scl[i*n_points:(i+1)*n_points, 0])

            limits0.append(limit0)

        return limits0

    def limit_loss(output):
        X0_output = torch.linalg.multi_dot([Ur, Sigma_r, output.T])

        limit_error = torch.zeros_like(X0_output)

        for j in range(limit_error.shape[1]):
            limit_error[:,j] = (g(limits0[0] - X0_output[:,j]) + 
                                g(X0_output[:,j] - limits0[1]))
        
        return -alpha_l*torch.linalg.matrix_norm(limit_error)
    
    def compute_alpha_l(loss_mll, Vr_pred_train):
        loss_l = limit_loss(Vr_pred_train).detach()

        if torch.abs(loss_l) < 1e-3:
            alpha_l = torch.tensor(1)
        else:
            alpha_l = 0.5*torch.abs(loss_mll/loss_l)

        return alpha_l

    n_cstr = 2
    d_array = np.linspace(P_train[:,0].min(), P_train[:,0].max(), n_cstr)
    h2_array = np.linspace(P_train[:,1].min(), P_train[:,1].max(), n_cstr)
    phi_array = np.linspace(P_train[:,2].min(), P_train[:,2].max(), n_cstr)

    d_grid, h2_grid, phi_grid = np.meshgrid(d_array, h2_array, phi_array)
    P_cstr = np.zeros((n_cstr**3,3))
    P_cstr[:,0] = d_grid.flatten()
    P_cstr[:,1] = h2_grid.flatten()
    P_cstr[:,2] = phi_grid.flatten()
    
    cgpr = CGPR(X_train, n_features, xyz, P_train, P_cstr, AddedLoss)
    cgpr.fit()
    Ur = torch.from_numpy(cgpr.Ur).contiguous().double()
    Sigma_r = torch.from_numpy(np.diag(cgpr.Sigma_r)).contiguous().double()
    
    limit_min = np.array([200, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float')
    limit_max = np.array([5000, 1, 1, 1, 1, 1, 1, 1, 1], dtype='float')
    limits0 = scale_limits([limit_min, limit_max])

    alpha_l = 0.
    loss_mll, Vr_pred_train = cgpr.compute_mll()
    # alpha_l = compute_alpha_l(loss_mll, Vr_pred_train)
    cgpr.train(likelihood=likelihoods[0], mean=models[0].mean_module, 
               kernel=models[0].covar_module, max_iter=10, verbose=True, lr=0.01)
    Ap, Sigmap = cgpr.predict(P_test)
    Xp = cgpr.reconstruct(Ap)

    # Select the feature to plot
    str_ind = 'OH'
    ind = features.index(str_ind)

    xp_test = Xp[ind*n_cells:(ind+1)*n_cells, 3]

    plot_contours_tri(xz[:,0], xz[:,1], [x_test, xp_test], cbar_label=str_ind)