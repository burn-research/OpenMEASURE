import pytest
import src.openmeasure.cokriging as ckg
import numpy as np
import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from tests.test_spr_data import plot_contours_tri


class TestGPR:

    def setup_method(self, method):
        path = '../data/ROM/'
        self.mesh_outline = np.genfromtxt(path + 'mesh_outline.csv', delimiter=',', skip_header=1)

        # This is a n x m matrix where n = 165258 is the number of cells times the number of features
        # and m = 41 is the number of simulations.
        X_train = np.load(path + 'X_2D_train.npy')
        n_linked = 30
        X_train_l = X_train[:, :n_linked]
        X_train_u = X_train[:, n_linked:]

        # This is a n x 4 matrix containing the 4 testing simulations
        self.X_test = np.load(path + 'X_2D_test.npy')

        self.features = ['T', 'CH4', 'O2', 'CO2', 'H2O', 'H2', 'OH', 'CO', 'NOx']
        n_features = len(self.features)

        # Select the feature to plot
        self.str_ind = 'OH'
        self.ind = self.features.index(self.str_ind)

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

        self.P_train_l = self.P_train[:n_linked, :]
        self.P_train_u = self.P_train[n_linked:, :]

        self.cokriging = ckg.CoKriging(self.P_train_l, self.P_train_u, X_train_l, X_train_u, X_train_l,
                                       xyz, xyz, n_features) 
        

    def teardown_method(self, method):
        pass

    
    def test_Cokriging(self):
        self.cokriging.manifold_alignment()
        self.cokriging.fit()
        Xp, _ = self.cokriging.predict(self.P_test)
            
        x_test = self.X_test[self.ind*self.n_cells:(self.ind+1)*self.n_cells,3]
        xp_test = Xp[self.ind*self.n_cells:(self.ind+1)*self.n_cells, 3]

        plot_contours_tri(self.xz[:,0], self.xz[:,1], [x_test, xp_test], cbar_label=self.str_ind)
