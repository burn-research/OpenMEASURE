import pytest
import src.openmeasure.gpr as gpr
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

        self.gpr = gpr.GPR(X_train, n_features, xyz, self.P_train, gpr_type='MultiTask') 

    def teardown_method(self, method):
        pass

    
    def test_GPR(self):
        self.gpr.fit()
        self.gpr.train()
        Ap, _ = self.gpr.predict(self.P_test)
        
        # Reconstruct the high-dimensional state from the POD coefficients
        Xp = self.gpr.reconstruct(Ap)
    
        x_test = self.X_test[self.ind*self.n_cells:(self.ind+1)*self.n_cells,3]
        xp_test = Xp[self.ind*self.n_cells:(self.ind+1)*self.n_cells, 3]

        plot_contours_tri(self.xz[:,0], self.xz[:,1], [x_test, xp_test], cbar_label=self.str_ind)

    def test_GPR_update(self):
        self.gpr.fit()
        self.gpr.train()
        Ap, Sigmap = self.gpr.predict(self.P_test)

        self.gpr.update(self.P_test, Ap, Sigmap, retrain=True, verbose=True)
        Ap, _ = self.gpr.predict(self.P_test)
        Xp = self.gpr.reconstruct(Ap)

        # Select the feature to plot
        str_ind = 'OH'
        ind = self.features.index(str_ind)

        x_test = self.X_test[self.ind*self.n_cells:(self.ind+1)*self.n_cells,3]
        xp_test = Xp[self.ind*self.n_cells:(self.ind+1)*self.n_cells, 3]

        plot_contours_tri(self.xz[:,0], self.xz[:,1], [x_test, xp_test], cbar_label=self.str_ind)


    def test_GPR_constained_prediction(self):
        self.gpr.fit()
        self.gpr.train()

        v = cp.Variable((self.gpr.r,1))
        mean = cp.Parameter((self.gpr.r,1))
        cov = cp.Parameter((self.gpr.r, self.gpr.r))
        objective = cp.Maximize(-cp.matrix_frac(v-mean, cov))

        limit_min = np.array([200, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float')
        limit_max = np.array([3000, 1, 1, 1, 1, 1, 1, 1, 1], dtype='float')
        limits0 = self.gpr.scale_limits([limit_min, limit_max])

        x0 = cp.matmul(self.gpr.Ur, cp.multiply(self.gpr.Sigma_r[:, np.newaxis], v))
        constraints = [x0 >= limits0[0][:, np.newaxis], 
                    x0 <= limits0[1][:, np.newaxis]]

        problem = cp.Problem(objective, constraints)
        problem_dict = {'problem': problem,
                        'v': v,
                        'mean': mean,
                        'cov': cov}

        Ap, _ = self.gpr.predict(self.P_test, problem_dict, solver='CLARABEL', verbose=True)
        
        # Reconstruct the high-dimensional state from the POD coefficients
        Xp = self.gpr.reconstruct(Ap)

        x_test = self.X_test[self.ind*self.n_cells:(self.ind+1)*self.n_cells,3]
        xp_test = Xp[self.ind*self.n_cells:(self.ind+1)*self.n_cells, 3]

        plot_contours_tri(self.xz[:,0], self.xz[:,1], [x_test, xp_test], cbar_label=self.str_ind)