import pytest
import src.openmeasure.sparse_sensing as sps
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri

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

    
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('z', fontsize=8)
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
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('z')
    
    fig.subplots_adjust(bottom=0., top=1., left=0., right=0.85, wspace=0.02, hspace=0.02)
    start = axs[1].get_position().bounds[1]
    height = axs[1].get_position().bounds[3]
    
    cb_ax = fig.add_axes([0.9, start, 0.05, height])
    cmap = mpl.colormaps[cmap_name]
    norm = mpl.colors.Normalize(vmin=z_min, vmax=z_max)
    
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, 
                orientation='vertical', label=cbar_label)
    
    plt.show()

class TestSPR:

    def setup_method(self, method):
        path = '../data/ROM/'
        self.mesh_outline = np.genfromtxt(path + 'mesh_outline.csv', delimiter=',', skip_header=1)

        # This is a n x m matrix where n = 165258 is the number of cells times the number of features
        # and m = 41 is the number of simulations.
        X_train = np.load(path + 'X_2D_train.npy')

        # This is a n x 4 matrix containing the 4 testing simulations
        self.X_test = np.load(path + 'X_2D_test.npy')

        self.features = ['T', 'CH4', 'O2', 'CO2', 'H2O', 'H2', 'OH', 'CO', 'NOx']
        n_features = len(self.features)

        # This is the file containing the x,z positions of the cells
        self.xz = np.load(path + 'xz.npy')
        self.n_cells = self.xz.shape[0]
        
        # Create the x,y,z array
        xyz = np.zeros((self.n_cells, 3))
        xyz[:,0] = self.xz[:,0]
        xyz[:,2] = self.xz[:,1]

        # This reads the files containing the parameters (D, H2, phi) with which 
        # the simulation were computed
        self.P_train = np.genfromtxt(path + 'parameters_train.csv', delimiter=',', skip_header=1)
        self.P_test = np.genfromtxt(path + 'parameters_test.csv', delimiter=',', skip_header=1)

        self.spr = sps.SPR(X_train, n_features, xyz) # Create the spr object

    def teardown_method(self, method):
        pass

    def test_optimal_placement_qr(self):
        self.spr.fit(scale_type='std', select_modes='number', n_modes=5)
        n_sensors = 5
        C_qr = self.spr.optimal_placement()
    
        # Get the sensors positions and features
        xz_sensors = np.zeros((n_sensors, 4))
        for i in range(n_sensors):
            index = np.argmax(C_qr[i,:])
            xz_sensors[i,:2] = self.xz[index % self.n_cells, :]
            xz_sensors[i,2] = index // self.n_cells

        plot_sensors(xz_sensors, self.features, self.mesh_outline)

    def test_prediction_optimal_placement_ols(self):
        self.spr.fit(scale_type='std', select_modes='number', n_modes=5)
        n_sensors = 5
        C_qr = self.spr.optimal_placement()

        y_qr = np.zeros((n_sensors,3))
        y_qr[:,0] = C_qr @ self.X_test[:,3]

        for i in range(n_sensors):
            y_qr[i,2] = np.argmax(C_qr[i,:]) //self.n_cells

        self.spr.train(C_qr)
        Ap, _ = self.spr.predict(y_qr)
        Xp = self.spr.reconstruct(Ap)

        # Select the feature to plot
        str_ind = 'OH'
        ind = self.features.index(str_ind)

        plot_contours_tri(self.xz[:,0], self.xz[:,1], [self.X_test[ind*self.n_cells:(ind+1)*self.n_cells, 3], 
                        Xp[ind*self.n_cells:(ind+1)*self.n_cells, 0]], cbar_label=str_ind)


    def test_prediction_optimal_placement_cols(self):
        self.spr.fit(scale_type='std', select_modes='number', n_modes=5, axis_cnt=1)
        n_sensors = 5
        C_qr = self.spr.optimal_placement()

        y_qr = np.zeros((n_sensors,3))
        y_qr[:,0] = C_qr @ self.X_test[:,3]

        for i in range(n_sensors):
            y_qr[i,2] = np.argmax(C_qr[i,:]) //self.n_cells

        limit_min = np.array([200., 0., 0., 0., 0., 0., 0., 0., 0.])
        limit_max = np.array([3000., 1., 1., 1., 1., 1., 1., 1., 1.])

        self.spr.train(C_qr, method='COLS', limits=[limit_min, limit_max])
        Ap, _ = self.spr.predict(y_qr)
        Xp = self.spr.reconstruct(Ap)

        # Select the feature to plot
        str_ind = 'OH'
        ind = self.features.index(str_ind)

        plot_contours_tri(self.xz[:,0], self.xz[:,1], [self.X_test[ind*self.n_cells:(ind+1)*self.n_cells, 3], 
                        Xp[ind*self.n_cells:(ind+1)*self.n_cells, 0]], cbar_label=str_ind)        
