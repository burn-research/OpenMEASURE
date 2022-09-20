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
            self.covar_module = ScaleKernel(kernel, batch_shape=shape)
            
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

    def __init__(self, X, n_features, xyz, P):
        super().__init__(X, n_features, xyz)
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
                    n_modes=99, mean=None, kernel=None, max_iter=1000, 
                    rel_error=1e-5, verbose=False, save_path=None):
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
        model = BatchIndipendentMultitaskGPModel(P0_torch, A0_torch, likelihood,
                                                 mean=mean, kernel=kernel)
        
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
                print('Iter %d/%d - Loss: %.3f - Noise: %.3f'
                      % (i + 1, max_iter, loss.item(), model.likelihood.noise.item()))

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
    import numpy as np
    # from gpr import GPR
    # from sparse_sensing import SPR
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    # Replace this with the path where you saved the data directory
    path = './data/ROM/'

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
    #------------------------------------GPR ROM--------------------------------------------------
    # Create the gpr object
    gpr = GPR(X_train, n_features, xyz, P_train)

    # Calculates the POD coefficients ap and the uncertainty for the test simulations
    Ap, Sigmap = gpr.fit_predict(P_test, verbose=True)

    # Reconstruct the high-dimensional state from the POD coefficients
    Xp = gpr.reconstruct(Ap)

    # Select the feature to plot
    str_ind = 'OH'
    ind = features.index(str_ind)

    x_test = X_test[ind*n_cells:(ind+1)*n_cells,3]
    xp_test = Xp[ind*n_cells:(ind+1)*n_cells, 3]

    plot_contours_tri(xz[:,0], xz[:,1], [x_test, xp_test], cbar_label=str_ind)

