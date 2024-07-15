'''
MODULE: dgpr.py
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
import torch
import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy 
import numpy as np
from .sparse_sensing import ROM
from .gpr import ExactGPModel
from scipy.integrate import BDF
from torch.utils.data import DataLoader, TensorDataset
class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, mean, kernel):
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(VariationalGPModel, self).__init__(variational_strategy)
        
        self.mean_module = mean
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        kernel_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, kernel_x)

class dGPR(ROM):
    '''
    Class used for building a GPR-based ROM of a dynamical system.
    
    Attributes
    ----------
    X : numpy array
        data matrix of dimensions (n,p) where n = n_features * n_points and p
        is the number of snapshots.
        
    X_dot : numpy array
        data matrix of dimensions (n,p) containing the time derivative of the
        matrix X.

    dt = float
        time interval between the snapshots in seconds.

    n_features : int
        the number of features in the dataset (temperature, velocity, etc.).
    
    xyz : numpy array
        3D position of the data in X, size (nx3).
         
        
    Methods
    ----------
    
    fit()
        Fit the ROM model.
    
    '''
    def __init__(self, X, X_dot, dt, n_features, xyz):
        super().__init__(X, n_features, xyz)
        
        self.X_dot = X_dot
        self.dt = dt

    def _reduction(self, Sigma, select_modes, n_modes):
        exp_variance = 100*np.cumsum(Sigma**2)/np.sum(Sigma**2)

        if select_modes == 'variance':
            if not 0 <= n_modes <= 100: 
                raise ValueError('The parameter n_modes is outside the[0-100] range.')
                
            # The r-order truncation is selected based on the amount of variance recovered
            if n_modes == 100:
                r = Sigma.size
            else:
                r = 1
                while exp_variance[r-1] < n_modes:
                    r += 1
    
        elif select_modes == 'number':
            if not type(n_modes) is int:
                raise TypeError('The parameter n_modes is not an integer.')
            if not 1 <= n_modes <= Sigma.size: 
                raise ValueError('The parameter n_modes is outside the [1-m] range.')
            r = n_modes
        else:
            raise ValueError('The select_mode value is wrong.')

        self.r = int(r)

        return int(r)
    
    def _train_loop(self, model, likelihood, Vr_torch, Vr_dot_torch, i):
        model.train()
        likelihood.train()
    
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        loss_old = 1e10
        e = 1e10
        j = 0
        while (e > self.rel_error) and (j < self.max_iter):
            optimizer.zero_grad()
            output = model(Vr_torch)
            loss = -mll(output, Vr_dot_torch)
            loss.backward()
            e = torch.abs(loss - loss_old).item()
            loss_old = loss
            if self.verbose == True:
                noise_avg = np.mean(model.likelihood.noise.detach().numpy())
                print(f'Iter {j+1:d}/{self.max_iter:d} - Mode: {i+1:d}/{self.r:d} - Loss: {loss.item():.2e} - ' \
                f'Mean noise: {noise_avg:.2e}')
                    
            optimizer.step()
            j += 1

        Vr_sigma = output.stddev.detach().numpy()
        
        return model, likelihood, Vr_sigma


    def _enkf(self, Q, R, X, d):
        n_samples = X.shape[1]
        mvn_d = gpytorch.distributions.MultivariateNormal(torch.from_numpy(d), torch.from_numpy(R))
    
        D = mvn_d.sample(torch.Size([n_samples])).T
        D = D.detach().numpy()

        Xp = X + Q @ np.linalg.inv(Q + R) @ (D - X)
        
        return Xp.T
    
    def _fun(self, t, v):

        v_torch = torch.from_numpy(v[np.newaxis, :]).contiguous().double()
        v_dot_mean = np.zeros((1, self.r))
        
        for i in range(self.r):
            # Set into eval mode
            self.models[i].eval()
            self.likelihoods[i].eval()

            observed_pred = self.likelihoods[i](self.models[i](v_torch))
            v_dot_mean[:,i] = observed_pred.mean.detach().numpy()
            
        return v_dot_mean.flatten()/self.scale_cst

    def fit(self, scale_type='std', axis_cnt=1, select_modes='variance', n_modes=99):
        '''
        Fit the ROM model.

        Parameters
        ----------
        scale_type : str, optional
            Type of scaling method for the data matrix. The default is 'std'.
        
        axis_cnt : int, optional
            Axis used to compute the centering coefficient. If None, the centering coefficient
            is a scalar. Default is 1.

        select_modes : str, optional
            Type of mode selection. The default is 'variance'. The available 
            options are 'variance' or 'number'.
            
        n_modes : int or float, optional
            Parameters that control the amount of modes retained. The default is 
            99, which represents 99% of the variance. If select_modes='number',
            n_modes represents the number of modes retained.
            
        '''

        self.scaleX_type = scale_type
        self.select_modes = select_modes
        self.n_modes = n_modes
        
        # Scale X and X_dot
        self.X0 = self.scale_data(scale_type, axis_cnt)
        self.X0_dot = self.X_dot/self.X_scl

        # Compute the POD      
        U, Sigma, Vt = np.linalg.svd(self.X0, full_matrices=False)
        
        # Reduce the dimensionality
        r = self._reduction(Sigma, select_modes, n_modes)
        Ur = U[:,:r]
        Sigmar = Sigma[:r]
        Vr = Vt[:r, :].T
        # Ar = np.transpose(np.diag(Sigma[:r]) @ Vt[:r, :])
        
        # Project X_dot
        Psi = np.transpose(Ur @ np.diag(1/Sigma[:r]))
        Vr_dot = np.transpose(Psi @ (self.X0 + self.dt*self.X0_dot) - Psi @ self.X0)/self.dt

        # Store the results
        self.Ur = Ur
        self.Sigmar = Sigmar
        self.Psi = Psi

        self.Vr = Vr
        self.Vr_dot = Vr_dot

    def train(self, n_samples=None, mean=None, kernel=None, likelihood=None, 
              max_iter=1000, rel_error=1e-5, lr=0.1, verbose=False):
        '''
        Train the GPR model.
        Return the model and likelihood.

        Parameters
        ----------
        n_samples : int, optional.
            If passed, the GPR is trained on a random subset with size n_samples.

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
        # self.scale_cst = self.Vr.shape[0]
        self.scale_cst = 1.

        rng_shuffle = np.random.default_rng()
        index_shuffle = np.arange(self.Vr.shape[0], dtype='int')
        rng_shuffle.shuffle(index_shuffle)
        
        if n_samples is not None:    
            index_shuffle = index_shuffle[:n_samples]
        else:
            n_samples = self.Vr.shape[0]

        Vr_torch = self.scale_cst*torch.from_numpy(self.Vr[index_shuffle, :]).contiguous().double()
        Vr_dot_torch = self.scale_cst*torch.from_numpy(self.Vr_dot[index_shuffle, :]).contiguous().double()
            
        models = []
        likelihoods = []

        self.mean = mean
        self.kernel = kernel
        self.likelihood = likelihood

        if mean is None:
            self.mean = gpytorch.means.ConstantMean()
        
        if kernel is None:
            self.kernel = gpytorch.kernels.ScaleKernel((gpytorch.kernels.RBFKernel()))
        
        if likelihood is None:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        Vr_dot_sigma = np.zeros((n_samples, self.r))

        for i in range(self.r):
            likelihood = copy.deepcopy(self.likelihood)
            mean = copy.deepcopy(self.mean)
            kernel = copy.deepcopy(self.kernel)
            model = ExactGPModel(Vr_torch, Vr_dot_torch[:,i], likelihood, mean, kernel)
            
            model.double()
            likelihood.double()

            model, likelihood, Vr_dot_sigma[:, i] = self._train_loop(model, likelihood, Vr_torch, Vr_dot_torch[:,i], i)

            models.append(model)
            likelihoods.append(likelihood)

        self.Vr_dot_sigma = Vr_dot_sigma
        self.models = models
        self.likelihoods = likelihoods
        
        return models, likelihoods
    
    def predict(self, Vr_star):
        '''
        Return the prediction vector. 
        This method has to be used after train.

        Parameters
        ----------
        Vr_star : numpy array
            The set of points where to evaluate the prediction, size (n_p,r).
        
        Returns
        -------
        Vr_dot_pred : numpy array
            The low-dimensional projection of the derivative, size (n_p,r)
        
        Vr_dot_sigma : numpy array
            Uncertainty in the prediction, size (n_p,r)

        '''
        
        if not hasattr(self, 'models'):
            raise AttributeError('The function fit has to be called '\
                                  'before calling predict.')
        
        n_p = Vr_star.shape[0]
        Vr_star_torch = self.scale_cst*torch.from_numpy(Vr_star).contiguous().double()
        
        Vr_dot_pred = np.zeros((n_p, self.r))
        Vr_dot_sigma = np.zeros((n_p, self.r))

        for i in range(self.r):
            # Set into eval mode
            self.models[i].eval()
            self.likelihoods[i].eval()

            # observed_pred = self.likelihoods[i](self.models[i](Vr_star_torch))
            observed_pred = self.models[i](Vr_star_torch)
            Vr_dot_pred[:,i] = observed_pred.mean.detach().numpy()
            Vr_dot_sigma[:,i] = observed_pred.stddev.detach().numpy()
                                
        return Vr_dot_pred/self.scale_cst, Vr_dot_sigma/self.scale_cst
    
    def forecast(self, Vr_start, n_timesteps, n_paths, dt_fc, 
                 method='euler-maruyama', noise_coeff=1.):
        '''
        Forecast an ensamble in the future. 
        
        Parameters
        ----------
        Vr_start : numpy array
            The starting point, size (r,).
        
        n_timesteps: int
            The number of timesteps to forecast.

        n_paths: int
            The number of members of the ensamble

        dt_fc: float
            The time delta between timesteps in seconds.
        
        method: str, optional
            The method used to integrate the SDE. The default is
            the Euler-Maruyama method.
        
        noise_coeff: float, optional
            Multiplicative coefficient for the noise level. 
            The default is 1.
            
        Returns
        -------
        Vr_fc : numpy array
            The forecasted ensamble, size (n_t, n_paths, r)
        
        '''

        Vr_fc = np.zeros((n_timesteps, n_paths, self.r))
        Vr_fc[0, :, :] = Vr_start

        rng = np.random.default_rng()        
        dW = noise_coeff*np.sqrt(dt_fc)*rng.normal(scale=1, size=(Vr_fc.shape))

        for i in range(n_timesteps-1):
            if method == 'euler-maruyama':
                Vr_dot_mean, Vr_dot_sigma = self.predict(Vr_fc[i, :, :])
                Vr_fc[i+1, :, :] = Vr_fc[i, :, :] + Vr_dot_mean*dt_fc + Vr_dot_sigma*dW[i, :, :]
        
            elif method == 'runge-kutta':
                Vr_dot_mean, Vr_dot_sigma = self.predict(Vr_fc[i, :, :])
                Vr_hat = Vr_fc[i, :, :] + Vr_dot_mean*dt_fc + Vr_dot_sigma*np.sqrt(dt_fc)
                _, Vr_hat_dot_sigma = self.predict(Vr_hat)
                Vr_fc[i+1, :, :] = (Vr_fc[i, :, :] + Vr_dot_mean*dt_fc + Vr_dot_sigma*dW[i, :, :] +
                                    0.5*(Vr_dot_sigma-Vr_hat_dot_sigma)*(dW[i, :, :]**2-dt_fc)/np.sqrt(dt_fc))
                
        return Vr_fc
    
    def forecast_enkf(self, Vr_start, t_start, n_timesteps, n_paths, dt_fc, 
                      Vr_meas_mean, Vr_meas_cov, t_meas, method='euler-maruyama'):
        
        '''
        Forecast an ensamble in the future and perform ensamble
        kalman filtering. 
        
        Parameters
        ----------
        Vr_start: numpy array
            The starting point, size (r,).

        t_start: float
            The starting time of the forecast.
        
        n_timesteps: int
            The number of timesteps to forecast.

        n_paths: int
            The number of members of the ensamble

        dt_fc: float
            The time delta between timesteps in seconds.

        Vr_meas_mean: numpy array
            The measured system's state, size (n_meas, r).

        Vr_meas_cov: numpy array
            The covariance of the measured system's state, size (n_meas, r, r).
        
        t_meas: numpy array
            The time of the measurements, size (n_meas,)

        method: str, optional
            The method used to integrate the SDE. The default is
            the Euler-Maruyama method.
        
        Returns
        -------
        Vr_fc : numpy array
            The forecasted ensamble, size (n_t, n_paths, r)
        
        '''

        Vr_fc = np.zeros((n_timesteps, n_paths, self.r))
        Vr_fc[0, :, :] = Vr_start

        t_fc = t_start + dt_fc*np.arange(n_timesteps)

        rng = np.random.default_rng()        
        dW = rng.normal(scale=np.sqrt(dt_fc), size=(Vr_fc.shape))

        for i in range(n_timesteps-1):
            
            t_diff = np.abs(t_fc[i]-t_meas)
            i_diff = np.argmin(t_diff)
            
            if t_diff[i_diff] <= dt_fc:
                Vr_cov = np.cov(Vr_fc[i, :, :], rowvar=False) + 1e-6*np.eye(self.r)

                Xp = self._enkf(Vr_cov, Vr_meas_cov[i_diff, :, :], 
                        Vr_fc[i, :, :].T, Vr_meas_mean[i_diff, :])
                
                Vr_fc[i, :, :] = Xp
        
            if method == 'euler-maruyama':
                Vr_dot_mean, Vr_dot_sigma = self.predict(Vr_fc[i, :, :])
                Vr_fc[i+1, :, :] = Vr_fc[i, :, :] + Vr_dot_mean*dt_fc + Vr_dot_sigma*dW[i, :, :]
        
            elif method == 'runge-kutta':
                Vr_dot_mean, Vr_dot_sigma = self.predict(Vr_fc[i, :, :])
                Vr_hat = Vr_fc[i, :, :] + Vr_dot_mean*dt_fc + Vr_dot_sigma*np.sqrt(dt_fc)
                _, Vr_hat_dot_sigma = self.predict(Vr_hat)
                Vr_fc[i+1, :, :] = (Vr_fc[i, :, :] + Vr_dot_mean*dt_fc + Vr_dot_sigma*dW[i, :, :] +
                                    0.5*(Vr_dot_sigma-Vr_hat_dot_sigma)*(dW[i, :, :]**2-dt_fc)/np.sqrt(dt_fc))

        return Vr_fc
    

    def forecast_bdf(self, t_start, t_end, Vr_start, max_step):

        bdf = BDF(self._fun, t_start, Vr_start, t_end, max_step=max_step)

        t_bdf = []
        v_bdf = []
        while bdf.status == 'running':
            bdf.step()
            t_bdf.append(bdf.t)
            v_bdf.append(bdf.y)
            
        return np.array(t_bdf), np.array(v_bdf)
    
class dGPR_vi(dGPR):
    '''
    Class used for building a GPR-based ROM of a dynamical system.
    
    Attributes
    ----------
    X : numpy array
        data matrix of dimensions (n,p) where n = n_features * n_points and p
        is the number of snapshots.
        
    X_dot : numpy array
        data matrix of dimensions (n,p) containing the time derivative of the
        matrix X.

    dt = float
        time interval between the snapshots in seconds.

    n_features : int
        the number of features in the dataset (temperature, velocity, etc.).
    
    xyz : numpy array
        3D position of the data in X, size (nx3).
         
        
    Methods
    ----------
    
    fit()
        Fit the ROM model.
    
    '''
    def __init__(self, X, X_dot, dt, n_features, xyz):
        super().__init__(X, X_dot, dt, n_features, xyz)
    
    
    def _train_loop(self, model, likelihood, i):
        model.train()
        likelihood.train()
    
        train_dataset = TensorDataset(self.Vr_torch, self.Vr_dot_torch[:,i])
        train_loader = DataLoader(train_dataset, batch_size=self.n_batch, shuffle=True)

        optimizer = torch.optim.Adam([{'params': model.parameters()},
                    {'params': likelihood.parameters()}], lr=self.lr)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=self.Vr_dot_torch.size(0))

        for e in range(self.n_epochs):
            for Vr_batch, Vr_dot_batch in train_loader:
        
                optimizer.zero_grad()
                output = model(Vr_batch)
                loss = -mll(output, Vr_dot_batch)
                loss.backward()
                if self.verbose == True:
                    noise_avg = np.mean(likelihood.noise.detach().numpy())
                    print(f'Epoch {e+1:d}/{self.n_epochs:d} - Mode: {i+1:d}/{self.r:d} - Loss: {loss.item():.2e} - ' \
                    f'Mean noise: {noise_avg:.2e}')
                        
                optimizer.step()
                
        Vr_sigma = model(self.Vr_torch).stddev.detach().numpy()
        
        return model, likelihood, Vr_sigma
    
    def train(self, V_dot_sigma=1, n_samples=None, n_batch=256, n_epochs=100, 
              mean=None, kernel=None, likelihood=None, 
              lr=0.1, verbose=False):
        '''
        Train the GPR model.
        Return the model and likelihood.

        Parameters
        ----------
        n_samples : int, optional.
            If passed, the GPR is trained on a random subset with size n_samples.

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
        
        self.lr = lr
        self.verbose = verbose
        self.scale_cst = 1.
        self.n_batch = n_batch
        self.n_epochs = n_epochs
        
        rng_shuffle = np.random.default_rng()
        index_shuffle = np.arange(self.Vr.shape[0], dtype='int')
        rng_shuffle.shuffle(index_shuffle)
        
        if n_samples is not None:    
            index_shuffle = index_shuffle[:n_samples]
        else:
            n_samples = self.Vr.shape[0]

        self.Vr_torch = torch.from_numpy(self.Vr).contiguous().double()
        # Vr_sigma_torch = V_dot_sigma*torch.ones_like(Vr_mean_torch).contiguous().double()
        
        self.Vr_dot_torch = torch.from_numpy(self.Vr_dot).contiguous().double()
            
        models = []
        likelihoods = []

        self.mean = mean
        self.kernel = kernel
        self.likelihood = likelihood

        if mean is None:
            self.mean = gpytorch.means.ConstantMean()
        
        if kernel is None:
            self.kernel = gpytorch.kernels.ScaleKernel((gpytorch.kernels.RBFKernel()))
        
        if likelihood is None:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        Vr_dot_sigma = np.zeros_like(self.Vr_dot)

        for i in range(self.r):
            likelihood = copy.deepcopy(self.likelihood)
            mean = copy.deepcopy(self.mean)
            kernel = copy.deepcopy(self.kernel)
            model = VariationalGPModel(self.Vr_torch[index_shuffle, :], mean, kernel)

            model.double()
            likelihood.double()

            model, likelihood, Vr_dot_sigma[:, i] = self._train_loop(model, likelihood, i)

            models.append(model)
            likelihoods.append(likelihood)

        self.Vr_dot_sigma = Vr_dot_sigma
        self.models = models
        self.likelihoods = likelihoods
        
        return models, likelihoods