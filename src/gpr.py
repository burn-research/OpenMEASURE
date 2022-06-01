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

import numpy as np
import torch
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
# from sparse_sensing import *
import sparse_sensing as sps

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
        
    mean : mean function from gpytorch
    
    kernel : kernel function from gpytorch    
        
    Methods
    ----------
    forward(x)
        Return the multivariate distribution given the input x.
    
    '''
    
    def __init__(self, X_train, Y_train, likelihood, mean=None, kernel=None):
        super().__init__(X_train, Y_train, likelihood)
        
        self.n_tasks = Y_train.shape[1]
        shape = torch.Size([self.n_tasks])
        
        if mean == None:
            self.mean_module = ConstantMean(batch_shape=shape)
        else:
            self.mean_module = mean
        
        if kernel == None:    
            self.covar_module = ScaleKernel(MaternKernel(nu=2.5, batch_shape=shape), 
                                      batch_shape=shape)
        else:
            self.covar_module = kernel
        

    def forward(self, x):
        mean_x = self.mean_module(x)
        kernel_x = self.covar_module(x)
        return MultitaskMultivariateNormal.from_batch_mvn(MultivariateNormal(mean_x, kernel_x))
    

class GPR(sps.ROM):
    '''
    Class used for building a GPR-based ROM.
    
    Attributes
    ----------
    X : numpy array
        data matrix of dimensions (n,p) where n = n_features * n_points and p
        is the number of operating conditions.
    
    P : numpy array
        Design features matrix of dimensions (p,d) where p is the number of 
        operating conditions.
        
    n_features : int
        the number of features in the dataset (temperature, velocity, etc.).
        
    Methods
    ----------
    scale_coefficients(A, scale_type)
        Scales the coefficients used for training the GPR model.
        
    unscale_coefficients(scale_type)
        Unscale the coefficients.
    
    fit_predict(xs, scale_type='standard', select_modes='variance', 
                n_modes=99)
        Trains the GPR model, then predicts ar and reconstructs x.
    
    predict(xs, scale_type='standard'):
        Predicts ar and reconstructs x.
    
    '''

    def __init__(self, X, P, n_features):
        super().__init__(X, n_features)
        self.P = P


    def scale_GPR_data(self, D, scale_type):
        '''
        Return the scaled input and target for the GPR model.

        Parameters
        ----------
        D : numpy array
            Data matrix to scale, size (p,d+q). The first d columns contain
            the design features, the second q columns contains the decomposition
            coefficients.
        scale_type : str
            Type of scaling.

        Returns
        -------
        D0: numpy array
            The scaled measurement vector.

        '''
        
        D0 = np.zeros_like(D)
        GPR_scale_mean = np.zeros((D.shape[1]))
        GPR_scale_std = np.zeros((D.shape[1]))
        if scale_type == 'standard':
            GPR_scale_mean = np.mean(D, axis=0)
            GPR_scale_std = np.std(D, axis=0)
            
            for i in range(D.shape[1]):
                D0[:,i] = (D[:,i] - GPR_scale_mean[i]) / GPR_scale_std[i]
        
            self.GPR_scale_mean = GPR_scale_mean
            self.GPR_scale_std = GPR_scale_std
        else:
            raise NotImplementedError('The scaling method selected has not been '\
                                      'implemented yet')
        
        return D0    


    def fit_predict(self, xs, scale_type='standard', select_modes='variance', 
                    n_modes=99, max_iter=1000, rel_error=1e-5, verbose=False, 
                    save_path=None):
        '''
        Fit the GPR model.
        Return the prediction vector.

        Parameters
        ----------
        xs : numpy array
            The set of design features to evaluate the prediction, size (n_p,d).
        scale_type : str, optional
            Type of scaling method. The default is 'standard'.
        select_modes : str, optional
            Type of mode selection. The default is 'variance'. The available 
            options are 'variance' or 'number'.
        n_modes : int or float, optional
            Parameters that control the amount of modes retained. The default is 
            99, which represents 99% of the variance. If select_modes='number',
            n_modes represents the number of modes retained.
        max_iter : int, optional
            Maximum number of iterations to train the hyperparameters. The default
            is 1000.
        rel_error : float, optional
            Minimum relative error below which the training of hyperparameters is
            stopped. The default is 1e-5.
        verbose : bool, optional
            If True, it will print informations on the training of the hyperparameters.
            The default is False.
        save_path : str, optional.
            String containing the path where to save the model (extension .pth).
            The default is None, and the model is not saved.

        Returns
        -------
        A_pred : numpy array
            The low-dimensional projection of the state of the system, size (n_p,q)
        Sigma : numpy array
            Uncertainty in the prediction, size (n_p,q)
        
        '''
        
        self.scale_type = scale_type
        X0 = self.scale_data(scale_type)
        U, A, exp_variance = self.decomposition(X0)
        Ur, Ar = self.reduction(U, A, exp_variance, select_modes, n_modes)
        self.Ur = Ur
        self.Ar = Ar
        self.r = Ar.shape[1]
        self.d = self.P.shape[1]
        p = Ar.shape[0]
        
        D = np.zeros((p, self.d + self.r))
        D[:, :self.d] = self.P
        D[:, self.d:] = Ar
        
        D0 = GPR.scale_GPR_data(self, D, scale_type)
        P0_torch = torch.from_numpy(D0[:, :self.d]).contiguous().float()
        A0_torch = torch.from_numpy(D0[:, self.d:]).contiguous().float() 
        
        likelihood = MultitaskGaussianLikelihood(num_tasks=self.r)
        model = BatchIndipendentMultitaskGPModel(P0_torch, A0_torch, likelihood)
        
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(likelihood, model)
        loss_old = 1e10
        e = 1e10
        i = 0
        while (e > rel_error) and (i < max_iter):
            optimizer.zero_grad()
            output = model(P0_torch)
            loss = -mll(output, A0_torch)
            loss.backward()
            e = torch.abs(loss - loss_old).item()
            loss_old = loss
            if verbose == True:
                print('Iter %d/%d - Loss: %.3f - Noise: %.3f - Lengthscale: %.3f'
                      % (i + 1, max_iter, loss.item(), model.likelihood.noise.item(),
                         model.covar_module.base_kernel.lengthscale[0]))

            optimizer.step()
            i += 1

        if save_path is not None:
            torch.save(model.state_dict(), save_path)
        
        self.model = model
        self.likelihood = likelihood
        
        A_pred, Sigma = self.predict(xs)
        
        return A_pred, Sigma
    
    def predict(self, xs):
        '''
        Return the prediction vector. 
        This method has to be used after fit_predict.

        Parameters
        ----------
        xs : numpy array
            The set of design features to evaluate the prediction, size (n_p,d).

        Returns
        -------
        A_pred : numpy array
            The low-dimensional projection of the state of the system, size (n_p,r)
        Sigma : numpy array
            Uncertainty in the prediction, size (n_p,r)

        '''
        
        if hasattr(self, 'model'):
            # Set into eval mode
            self.model.eval()
            self.likelihood.eval()

            if xs.ndim < 2:
                xs = xs[np.newaxis, :]
            
            n_p = xs.shape[0]
            
            xs0 = np.zeros_like(xs)
            for i in range(xs.shape[1]):
                xs0[:,i] = (xs[:,i] - self.GPR_scale_mean[i]) / self.GPR_scale_std[i]

            xs0_torch = torch.from_numpy(xs0).contiguous().float()
            observed_pred = self.likelihood(self.model(xs0_torch))
            A0_pred = observed_pred.mean.detach().numpy()
            
            A0_cov = observed_pred.lazy_covariance_matrix
            Sigma0 = np.sqrt(A0_cov.diag().detach().numpy())
            Sigma0 = Sigma0.reshape((n_p, self.r))
            
            A_pred = np.zeros_like(A0_pred)
            Sigma = np.zeros_like(Sigma0)
            
            for i in range(A_pred.shape[1]):
                A_pred[:,i] = self.GPR_scale_std[self.d+i] * A0_pred[:, i] + self.GPR_scale_mean[self.d+i]
                Sigma[:,i] = self.GPR_scale_std[self.d+i] * Sigma0[:, i] + self.GPR_scale_mean[self.d+i]
            
        else:
            raise AttributeError('The function fit_predict has to be called '\
                                  'before calling predict.')
            
        return A_pred, Sigma

    def reconstruct(self, Ar):
        '''
        Reconstruct the X matrix from the low-dimensional representation.

        Parameters
        ----------
        Ar : numpy array
            The matrix containing the low-dimensional coefficients, size (n_p,r).

        Returns
        -------
        X_rec : numpy array
            The high-dimensional representation of the state of the system, size (n,n_p)
            
        '''
        
        n_p = Ar.shape[0]
        n = self.Ur.shape[0]
        X_rec = np.zeros((n, n_p))
        for i in range(n_p):
            x0_rec = self.Ur @ Ar[i,:]
            X_rec[:,i] = self.unscale_data(x0_rec, self.scale_type)

        return X_rec
    
if __name__ == '__main__':
    X = np.random.rand(15, 5)
    P = np.random.rand(5, 2)
    gpr = GPR(X, P, 5)
    
    U, A, e = gpr.decomposition(X, decomp_type='POD')
    Ur, Ar = gpr.reduction(U, A, e, select_modes='number', n_modes=3)
    
    P_test = P[0:3,:] + 1
    Ap, Sigma = gpr.fit_predict(P_test)
    X_rec = gpr.reconstruct(Ap)