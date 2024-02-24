
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
import sparse_sensing as sps
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
        self.rom_hf = sps.ROM(self.Y_train_hf_l, self.n_features, self.xyz_hf)   # Create ROM object for scaling
        self.rom_lf = sps.ROM(np.concatenate((self.Y_train_lf_l, self.Y_train_lf_u), axis=1), self.n_features, self.xyz_lf)

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
    
#%%
if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    from shapely.geometry import Polygon
    
    path_data = '/Users/alberto/Documents/Research/Co-kriging/data'

    hf_data = np.load(f'{path_data}/hf_data.npy')
    lf_data = np.load(f'{path_data}/lf_data.npy')
    xyzi_hf = np.load(f'{path_data}/xyzi_hf.npy')
    xyzi_lf = np.load(f'{path_data}/xyzi_lf.npy')
    par = np.load(f'{path_data}/parameters.npy')

    test_id = [0, 21, 27, 38]
    train_par = np.delete(par, test_id, axis=0)
    test_par = par[test_id,:]

    n_cells_hf = xyzi_hf.shape[0]
    n_cells_lf = xyzi_lf.shape[0]

    features = ['T', 'CH4', 'O2', 'CO2', 'H2O', 'H2', 'OH', 'CO']
    n_features = len(features)

    Y_hf_train = np.delete(hf_data, test_id, axis=1)
    Y_hf_test = hf_data[:, test_id]

    Y_lf_train = np.delete(lf_data, test_id, axis=1)
    Y_lf_test = lf_data[:, test_id]

    n_subset = 41
    np.random.seed(1)
    r_index = np.linspace(0, Y_hf_train.shape[1]-1, Y_hf_train.shape[1], dtype=np.int)
    np.random.shuffle(r_index)

    X_train_l = train_par[r_index[:n_subset], :]
    X_train_u = train_par[r_index[n_subset:], :]
    
    Y_train_hf_l = Y_hf_train[:, r_index[:n_subset]]

    Y_train_lf_l = Y_lf_train[:, r_index[:n_subset]]
    Y_train_lf_u = Y_lf_train[:, r_index[n_subset:]]
    #%%

    cokriging = CoKriging(X_train_l, X_train_u, Y_train_lf_l, Y_train_lf_u, Y_train_hf_l,
                          xyzi_lf, xyzi_hf, n_features)
    cokriging.manifold_alignment(select_modes='variance', n_modes_hf=99.9, n_modes_lf=99)
    cokriging.fit()
    
    Y_pred, Y_mse = cokriging.predict(test_par)

    #%%

    def contour_furnace():
        path_mesh = '/Users/alberto/Documents/Research/Digital_twin_ss/'
        mesh_data = np.genfromtxt(path_mesh + 'mesh/mesh_data.csv', delimiter=',',
                                skip_header=1)

        mask = mesh_data[:, 2] < 1e-17
        n_points = np.sum(mask)
        contour_data = np.zeros((n_points, 2))
        contour_data[:, 0] = mesh_data[:, 1][mask]
        contour_data[:, 1] = mesh_data[:, 3][mask]

        srt_contour_data = np.zeros_like(contour_data)
        temp = np.copy(contour_data)

        srt_contour_data[0, :] = contour_data[0, :]
        ind = 0

        for i in range(1, n_points):
            new_point = srt_contour_data[i-1, :]
            temp = np.delete(temp, ind, 0)

            dist = new_point-temp
            dist_norm = np.linalg.norm(dist, axis=1)
            ind = np.argmin(dist_norm)
            srt_contour_data[i, :] = temp[ind, :]

        return srt_contour_data

    def plot_contours_tri(xs, ys, zs, cbar_label='', filename=''):
        triang0 = tri.Triangulation(xs[0], ys[0])
        triang1 = tri.Triangulation(xs[1], ys[1])

        # contour = contour_furnace()
        # outline = Polygon(contour)
        
        # mask0 = [not outline.contains(Polygon(zip(xs[0][t], ys[0][t]))) 
        #         for t in triang0.get_masked_triangles()]
        
        # mask1 = [not outline.contains(Polygon(zip(xs[1][t], ys[1][t]))) 
        #         for t in triang1.get_masked_triangles()]

        # triang0.set_mask(mask0)
        # triang1.set_mask(mask1)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6,6))
        
        z_min = min([np.min(zs[0]), np.min(zs[1])])
        z_max = max([np.max(zs[0]), np.max(zs[1])])
    
        n_levels = 32
        levels = np.linspace(z_min, z_max, n_levels)
        cmap_name= 'inferno'
        
        for i, ax in enumerate(axs):
            if i == 0:
                ax.tricontourf(triang0, zs[i], levels, vmin=z_min, vmax=z_max, cmap=cmap_name)
                # ax.plot(contour[:,0], contour[:,1], c='k')
                ax.invert_xaxis()
            else:
                ax.tricontourf(triang1, zs[i], levels, vmin=z_min, vmax=z_max, cmap=cmap_name)
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False) 
                # ax.plot(contour[:,0], contour[:,1], c='k')

            ax.set_aspect('equal')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            
        fig.subplots_adjust(bottom=0., top=1., left=0., right=0.85, wspace=0.02, hspace=0.02)
        start = axs[1].get_position().bounds[1]
        height = axs[1].get_position().bounds[3]
        
        cb_ax = fig.add_axes([0.9, start, 0.05, height])
        cmap = mpl.cm.get_cmap(cmap_name, n_levels)
        norm = mpl.colors.Normalize(vmin=z_min, vmax=z_max)
        
        text = cb_ax.yaxis.label
        font = mpl.font_manager.FontProperties(size=16)
        text.set_font_properties(font)

        cbformat = mpl.ticker.ScalarFormatter(useMathText=True)   # create the formatter
        cbformat.set_powerlimits((-3, 4))

        cb_ax.tick_params(labelsize=14)
        cb_ax.yaxis.offsetText.set_fontsize(14)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, 
                    orientation='vertical', label=cbar_label, format=cbformat)
        
        if filename != '':
            fig.savefig(filename, transparent=True, dpi=600, bbox_inches='tight')    
        
        plt.show()

    str_ind = 'T'
    ind = features.index(str_ind)

    eps = 5e-2
    mask_plot = (xyzi_hf[:,1] < eps) & (xyzi_hf[:,2] > 0)

    x_plot_test = Y_hf_test[ind*n_cells_hf:(ind+1)*n_cells_hf,3][mask_plot]
    x_plot_pred = Y_pred[ind*n_cells_hf:(ind+1)*n_cells_hf,3][mask_plot]
    plot_contours_tri([xyzi_hf[:,0][mask_plot], xyzi_hf[:,0][mask_plot]], [xyzi_hf[:,2][mask_plot],xyzi_hf[:,2][mask_plot]], 
                        [x_plot_test, x_plot_pred])
    