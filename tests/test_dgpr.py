import pytest
import src.openmeasure.dgpr as dgpr
import numpy as np
import matplotlib.pyplot as plt

class TestdGPR:

    def setup_method(self, method):
        self.n_points = 20  # number of "grid points"
        self.n_features = 1 # number of features
        self.m = 100  # number of "snapshots"

        self.angle = np.linspace(0, 4*np.pi, self.m)
        self.time = np.linspace(0, 1, self.m)
        self.dt = self.time[1] - self.time[0]
        
        self.V = np.zeros((self.m, 2))

        self.V[:, 0] = np.sin(self.angle)  
        self.V[:, 0] /= np.linalg.norm(self.V[:,0]) 

        self.V[:, 1] = np.cos(self.angle)  
        self.V[:, 1] /= np.linalg.norm(self.V[:,1])

        Sigma = np.array([100., 50.]) 

        rng = np.random.default_rng()
        k = np.ones((self.n_points*self.n_features,))
        
        rng = np.random.default_rng(1233435232)
        A = np.empty((self.n_points*self.n_features, 2))
        A[:,0] = 100*rng.normal(size=(self.n_points*self.n_features))
        A[:,1] = 100*rng.normal(size=(self.n_points*self.n_features))

        self.U, _ = np.linalg.qr(A)

        X0 = np.linalg.multi_dot([self.U, np.diag(Sigma), self.V.T])  
        X_cnt = 20 + np.zeros((X0.shape[0], 1))
        X = X0 + X_cnt
        X_dot = np.gradient(X, self.dt, axis=1)

        self.time_train = self.time[:self.m//2]
        self.time_test = self.time[self.m//2:]

        self.V_train = self.V[:self.m//2, :]
        self.V_test = self.V[self.m//2:, :]

        self.X_train = X[:, :self.m//2]
        self.X_test = X[:, self.m//2:]

        self.X_dot_train = X_dot[:, :self.m//2]
        self.X_dot_test = X_dot[:, self.m//2:]

        xyz = np.zeros((self.n_points, 3))
        self.dgpr = dgpr.dGPR(self.X_train, self.X_dot_train, self.dt, self.n_features, xyz)
        self.dgpr_vi = dgpr.dGPR_vi(self.X_train, self.X_dot_train, self.dt, self.n_features, xyz)
        

    def teardown_method(self, method):
        pass

    def test_fit(self):
        self.dgpr.fit(scale_type='none', select_modes='number', n_modes=2)
        np.testing.assert_allclose(np.abs(self.U), np.abs(self.dgpr.Ur), atol=1e-3)
        
    def test_predict(self):
        self.dgpr.fit(scale_type='none')
        self.dgpr.train()
        
        X0_test = (self.X_test - self.dgpr.X_cnt)/self.dgpr.X_scl
        X0_dot_test = self.X_dot_test/self.dgpr.X_scl

        self.V_test = np.transpose(self.dgpr.Psi @ X0_test)
        V_dot_test = np.transpose(self.dgpr.Psi @ (X0_test + self.dt*X0_dot_test) - self.dgpr.Psi @ X0_test)/self.dt
        
        V_dot_pred, _ = self.dgpr.predict(self.V_test)

        np.testing.assert_allclose(V_dot_test, V_dot_pred, rtol=1e-3, atol=5e-1)

    def test_predict_vi(self):
        self.dgpr_vi.fit(scale_type='none')
        self.dgpr_vi.train(n_batch=10, verbose=True)
        
        X0_test = (self.X_test - self.dgpr_vi.X_cnt)/self.dgpr_vi.X_scl
        X0_dot_test = self.X_dot_test/self.dgpr_vi.X_scl

        self.V_test = np.transpose(self.dgpr_vi.Psi @ X0_test)
        V_dot_test = np.transpose(self.dgpr_vi.Psi @ (X0_test + self.dt*X0_dot_test) - self.dgpr_vi.Psi @ X0_test)/self.dt
        
        V_dot_pred, _ = self.dgpr_vi.predict(self.V_test)

        np.testing.assert_allclose(V_dot_test, V_dot_pred, rtol=1e-2, atol=5e-1)

    def test_forecast(self):
        self.dgpr.fit(scale_type='none')
        self.dgpr.train()
        
        V_fc = self.dgpr.forecast(self.V_test[0, :], self.time_test.size, 10, self.dt)
        V_fc_mean = np.mean(V_fc, axis=1)

        np.testing.assert_allclose(self.V_test, V_fc_mean, rtol=1e-3, atol=5e-1)

    def test_forecast_vi(self):
        self.dgpr_vi.fit(scale_type='none')
        self.dgpr_vi.train()
        
        V_fc = self.dgpr_vi.forecast(self.V_test[0, :], self.time_test.size, 10, self.dt)
        V_fc_mean = np.mean(V_fc, axis=1)

        np.testing.assert_allclose(self.V_test, V_fc_mean, rtol=1e-3, atol=5e-1)        
