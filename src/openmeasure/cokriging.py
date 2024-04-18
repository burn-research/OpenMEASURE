
'''
MODULE: cokriging.py
@Authors:
    A. Procacci [1]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
@Contacts:
    alberto.procacci@ulb.be
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: alberto.procacci@ulb.be
'''

#%%
import numpy as np
from .sparse_sensing import ROM
from openmdao.surrogate_models.multifi_cokriging import MultiFiCoKriging

class CoKriging():
    def __init__(self, X_train_l, X_train_u, Y_train_lf_l, Y_train_lf_u, Y_train_hf_l,
                 xyz_lf, xyz_hf, n_features):
        self.X_train_l = X_train_l  # Linked parameters
        self.X_train_u = X_train_u  # Unlinked parameters
        self.Y_train_lf_l = Y_train_lf_l  # Linked LF data
        self.Y_train_lf_u = Y_train_lf_u  # Unlinked LF data
        self.Y_train_hf_l = Y_train_hf_l  # Linked HF data
        self.xyz_lf = xyz_lf # Position of the data in the space
        self.xyz_hf = xyz_hf
        self.n_features = n_features # Number of features for single data vector
        self.n_linked = X_train_l.shape[0]  # Number of linked conditions
        self.n_unlinked = X_train_u.shape[0] # Number of unlinked conditions
        self.n_latent = 0
        self.scale_type = 'std'  # Standard scaling is the default    
        self.regr_type = 'linear' # Regression function for cokriging
        self.rho_regr = 'constant'
        self.normalize = True
        self.theta = None
        self.theta0 = None
        self.thetaL = None
        self.thetaU = None
        self.initial_range = 0.3
        self.tol = 1e-6

        if (Y_train_lf_l.shape[1] != self.n_linked) or (Y_train_hf_l.shape[1] != self.n_linked):
            raise Exception(
            '''The number of linked conditions does not correspond to the number of columns of
            Y_train_lf_l or Y_train_hf_l''')
            exit()
        if (Y_train_lf_u.shape[1] != self.n_unlinked):
            raise Exception(
            '''The number of unlinked conditions does not correspond to the number of columns of
            Y_train_lf_u''')
            exit()

    def manifold_alignment(self, select_modes='variance', n_modes_hf=99, n_modes_lf=99):
        self.rom_hf = ROM(self.Y_train_hf_l, self.n_features, self.xyz_hf)   # Create ROM object for scaling
        self.rom_lf = ROM(np.concatenate((self.Y_train_lf_l, self.Y_train_lf_u), axis=1), self.n_features, self.xyz_lf)

        X0_hf = self.rom_hf.scale_data(self.scale_type) # Scale data
        X0_lf = self.rom_lf.scale_data(self.scale_type)

        U_hf, Sigma_hf, V_hf = np.linalg.svd(X0_hf, full_matrices=False) # SVD to find the HF and LF decomposition
        U_lf, Sigma_lf, V_lf = np.linalg.svd(X0_lf, full_matrices=False)

        self.Sigma_hf = Sigma_hf
        self.Sigma_lf = Sigma_lf

        Z_hf = np.diag(Sigma_hf) @ V_hf # Calculate the scores
        Z_lf = np.diag(Sigma_lf) @ V_lf

        # Reduction of dimensionality
        exp_variance_hf = 100*np.cumsum(Sigma_hf**2)/np.sum(Sigma_hf**2)
        exp_variance_lf = 100*np.cumsum(Sigma_lf**2)/np.sum(Sigma_lf**2)

        Ur_hf, Zr_hf_t = self.rom_hf.reduction(U_hf, Z_hf.T, exp_variance_hf, select_modes, n_modes_hf)
        Ur_lf, Zr_lf_t = self.rom_lf.reduction(U_lf, Z_lf.T, exp_variance_lf, select_modes, n_modes_lf)

        Zr_hf = Zr_hf_t.T
        Zr_lf = Zr_lf_t.T

        self.r_hf = Ur_hf.shape[1]
        self.r_lf = Ur_lf.shape[1]

        if self.r_lf < self.r_hf:
            padding = np.zeros((self.r_hf-self.r_lf, Zr_lf.shape[1]))
            Zr_lf = np.concatenate([Zr_lf, padding], axis=0)

        Zr_lf_l = Zr_lf[:, :self.n_linked]  # Split in linked and unlinked
        Zr_lf_u = Zr_lf[:, self.n_linked:]

        Z0r_hf = np.zeros_like(Zr_hf)  # Center the scores
        for i in range(Z0r_hf.shape[0]):
            Z0r_hf[i,:] = Zr_hf[i,:] - np.mean(Zr_hf[i,:])
        
        Z0r_lf_l = np.zeros_like(Zr_lf_l)
        for i in range(Z0r_lf_l.shape[0]):
            Z0r_lf_l[i,:] = Zr_lf_l[i,:] - np.mean(Zr_lf_l[i,:])
        
        Ur, Sigmar, Vr_t = np.linalg.svd(Z0r_lf_l @ Z0r_hf.T, full_matrices=False)  # Compute the SVD for the procrustes projection
        sr = np.sum(Sigmar)/np.trace(Z0r_lf_l @ Z0r_lf_l.T)
        Qr = np.transpose(Vr_t) @ Ur.T
        Zr_aligned = sr * Qr @ Zr_lf  # Compute the aligned LF scores

        self.n_latent = Zr_aligned.shape[0]
        self.Zr_aligned = Zr_aligned
        self.Ur_hf = Ur_hf
        self.Zr_hf = Zr_hf

    def fit(self):
        X_train = np.concatenate((self.X_train_u, self.X_train_l), axis=0)

        self.model_list = []
        for k in range(self.n_latent):
            # Create a list of cokriging models
            self.model_list.append(MultiFiCoKriging(regr=self.regr_type, rho_regr=self.rho_regr, theta=self.theta,
                                               theta0=self.theta0, thetaL=self.thetaL, thetaU=self.thetaU, normalize=self.normalize))
            # Fit the list of models
            self.model_list[k].fit([X_train , self.X_train_l], [self.Zr_aligned[k,:], self.Zr_hf[k,:]], 
                              initial_range=self.initial_range, tol=self.tol)
        

    def predict(self, X_test, n_truncated=None):
        n_test = X_test.shape[0]  # Number of testing conditions

        if n_truncated is None:
            n_truncated = self.n_latent
        
        Z_pred = np.zeros((n_truncated, n_test))  
        Z_mse = np.zeros((n_truncated, n_test))

        for i in range(n_truncated):
            Z_pred[i,:] = self.model_list[i].predict(X_test)[0].flatten()  # Compute the prediction in the latent space
            Z_mse[i,:] = self.model_list[i].predict(X_test)[1].flatten()  # Compute the MSE (?) of the prediction

        Y0_pred = self.Ur_hf @ Z_pred # Project in the original space
        Y0_mse = self.Ur_hf @ Z_mse 
        
        Y_pred = np.empty_like(Y0_pred)
        Y_mse = np.empty_like(Y0_mse)
        for i in range(n_test):
            Y_pred[:,i] = self.rom_hf.unscale_data(Y0_pred[:,i]) # Unscale the data
            Y_mse[:,i] = self.rom_hf.unscale_data(Y0_mse[:,i])
            
        return Y_pred, Y_mse
    