import pytest
import src.openmeasure.sparse_sensing as sps
import numpy as np

class TestROM:

    def setup_method(self, method):
        rng = np.random.default_rng()
        self.n_points = 10
        self.n_features = 2
        self.m = 5
        X = rng.random(size=(self.n_points*self.n_features, self.m))   
        xyz = rng.random(size=(self.n_points, 3))
        self.rom = sps.ROM(X, self.n_features, xyz)

    def teardown_method(self, method):
        pass

    def test_centering_axis_one(self):
        self.rom.scale_data()
        np.testing.assert_array_equal(self.rom.X_cnt, np.mean(self.rom.X, axis=1)[:, np.newaxis])

    def test_scaling(self):
        self.rom.scale_data()
        X_scl = np.zeros((self.rom.X.shape[0], 1))
        Xc = self.rom.X - self.rom.X_cnt
        for i_f in range(self.rom.n_features):
            X_scl[i_f*self.rom.n_points:(i_f+1)*self.rom.n_points] = np.std(Xc[i_f*self.rom.n_points:(i_f+1)*self.rom.n_points])
        
        np.testing.assert_array_equal(self.rom.X_scl, X_scl)

    def test_centering_and_scaling(self):
        X0 = self.rom.scale_data()
        X_scl = np.zeros((self.rom.X.shape[0], 1))
        Xc = self.rom.X - self.rom.X_cnt
        for i_f in range(self.rom.n_features):
            X_scl[i_f*self.rom.n_points:(i_f+1)*self.rom.n_points] = np.std(Xc[i_f*self.rom.n_points:(i_f+1)*self.rom.n_points])
        
        X0_check = (self.rom.X - np.mean(self.rom.X, axis=1)[:, np.newaxis])/X_scl
        np.testing.assert_array_equal(X0, X0_check)

    def test_decomposition_svd(self):
        X0 = self.rom.scale_data()
        U, Sigma, Vt = np.linalg.svd(X0, full_matrices=False)
        A = np.dot(np.diag(Sigma), Vt).T

        Ur, Ar, _ = self.rom.decomposition(X0, n_modes=100)
        np.testing.assert_array_equal(U, Ur)
        np.testing.assert_array_equal(A, Ar)

    def test_reduction_number(self):
        X0 = self.rom.scale_data()
        self.rom.decomposition(X0, select_modes='number', n_modes=self.m-1)
        assert self.rom.r == self.m-1

    def test_reduction_variance(self):
        X0 = self.rom.scale_data()
        self.rom.decomposition(X0, select_modes='variance', n_modes=100)
        assert self.rom.r == self.m

    def test_fit(self):
        X0 = self.rom.scale_data()
        _, Sigma, Vt = np.linalg.svd(X0, full_matrices=False)
        V = Vt.T
        
        self.rom.fit(n_modes=100)
        np.testing.assert_allclose(self.rom.Vr, V)
        np.testing.assert_allclose(self.rom.Sigma_r, Sigma)

    def test_unscaling(self):
        X0 = self.rom.scale_data()
        self.rom.fit(n_modes=100)

        np.testing.assert_allclose(self.rom.unscale_data(X0[:,0]), self.rom.X[:, 0])
        
    def test_reconstruction(self):
        self.rom.fit(n_modes=100)
        x_rec = self.rom.reconstruct(self.rom.Ar[0, :])
        np.testing.assert_allclose(x_rec, self.rom.X[:,[0]])
        

    