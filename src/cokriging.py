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
import sparse_sensing as sps
from openmdao.surrogate_models.multifi_cokriging import MultiFiCoKriging

class CoKriging():
    def __init__(self, X_train_l, X_train_u, Y_train_lf_l, Y_train_lf_u, Y_train_hf_l):
        self.X_train_l = X_train_l  # Linked parameters
        self.X_train_u = X_train_u  # Unlinked parameters
        self.Y_train_lf_l = Y_train_lf_l  # Linked LF data
        self.Y_train_lf_u = Y_train_lf_u  # Unlinked LF data
        self.Y_train_hf_l = Y_train_hf_l  # Linked HF data
        self.n_latent = 0
        self.scale_type = 'std'  # Standard scaling is the default    
        self.n_linked = Y_train_lf_l.shape[1]  # Number of linked conditions
        self.regr_type = 'linear' # Regression function for cokriging
        self.rho_regr = 'constant'
        self.normalize = True
        self.theta = None
        self.theta0 = None
        self.thetaL = None
        self.thetaU = None
        self.initial_range = 0.3
        self.tol = 1e-6

    def manifold_alignment(self):
        self.rom_hf = sps.ROM(self.Y_train_hf_l)   # Create ROM object for scaling
        self.rom_lf = sps.ROM(np.concatenate((self.Y_train_lf_l, self.Y_train_lf_u), axis=1))

        X0_hf = self.rom_hf.scale_data(self.scale_type) # Scale data
        X0_lf = self.rom_lf.scale_data(self.scale_type)

        U_hf, Sigma_hf, V_hf = np.linalg.svd(X0_hf, full_matrices=False) # SVD to find the HF and LF decomposition
        U_lf, Sigma_lf, V_lf = np.linalg.svd(X0_lf, full_matrices=False)

        Z_hf = np.diag(Sigma_hf) @ V_hf # Calculate the scores
        Z_lf = np.diag(Sigma_lf) @ V_lf

        Z_lf_l = Z_lf[:, :self.n_linked]  # Split in linked and unlinked
        Z_lf_u = Z_lf[:, self.n_linked:]

        Z0_hf = np.zeros_like(Z_hf)  # Center the scores
        for i in range(Z0_hf.shape[0]):
            Z0_hf[i,:] = Z_hf[i,:] - np.mean(Z_hf[i,:])
        
        Z0_lf_l = np.zeros_like(Z_lf_l)
        for i in range(Z0_lf_l.shape[0]):
            Z0_lf_l[i,:] = Z_lf_l[i,:] - np.mean(Z_lf_l[i,:])
        
        U, Sigma, V_t = np.linalg.svd(Z0_lf_l @ Z0_hf.T, full_matrices=False)  # Compute the SVD for the procrustes projection
        s = np.sum(Sigma)/np.trace(Z0_lf_l @ Z0_lf_l.T)
        Q = np.transpose(V_t) @ U.T
        Z_aligned = s * Q @ Z_lf  # Compute the aligned LF scores

        self.n_latent = Z_aligned.shape[0]
        self.Z_aligned = Z_aligned
        self.U_hf = U_hf
        self.Z_hf = Z_hf

    def fit(self):
        X_train = np.concatenate((self.X_train_u, self.X_train_l), axis=1)

        self.model_list = []
        for k in range(self.n_latent):
            # Create a list of cokriging models
            self.model_list.append(MultiFiCoKriging(regr=self.regr_type, rho_regr=self.rho_regr, theta=self.theta,
                                               theta0=self.theta0, thetaL=self.thetaL, thetaU=self.thetaU, normalize=self.normalize))
            # Fit the list of models
            self.model_list[k].fit([X_train , self.X_train_l], [self.Z_aligned[k,:], self.Z_hf[k,:]], 
                              initial_range=self.initial_range, tol=self.tol)
        

    def predict(self, X_test, n_truncated=None):
        n_test = X_test.shape[0]

        if n_truncated is None:
            n_truncated = self.n_latent
        
        Z_pred = np.zeros((n_truncated, n_test))
        Z_mse = np.zeros((n_truncated, n_test))

        for i in range(n_truncated):
            Z_pred[i,:] = self.model_list[i].predict(X_test)[0]
            Z_mse[i,:] = self.model_list[i].predict(X_test)[1]

        Y0_pred = self.U_hf @ Z_pred
        Y_pred = self.rom_hf.unscale_data(Y0_pred)

        Y0_mse = self.U_hf @ Z_mse
        Y_mse = self.rom_hf.unscale_data(Y0_mse)

        return Y_pred, Y_mse