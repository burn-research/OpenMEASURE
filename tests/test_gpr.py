import pytest
import src.openmeasure.gpr as gpr
import numpy as np
import matplotlib.pyplot as plt

class TestGPR:

    def setup_method(self, method):
        self.n_points = 20  # number of "grid points"
        self.n_features = 1 # number of features
        self.m = 10  # number of "snapshots"
        
        self.P = np.linspace(0, 1, 2*self.m)[:, np.newaxis]  # input parameters
        self.V = 10*np.sin(2*np.pi*self.P)   # target variable
        self.V = self.V/np.linalg.norm(self.V) #  normalisation

        Sigma = 100 

        self.U = np.zeros((self.n_points*self.n_features, 1))  # create the U mapping
        self.U[:, 0] = np.arange(1, self.U.shape[0]+1)
        self.U = self.U/np.linalg.norm(self.U)
        
        X0 = self.U @ (Sigma * self.V.T)  
        X_cnt = 20 + np.zeros((X0.shape[0], 1))
        X = X0 + X_cnt
        
        self.P_train = self.P[::2, :]
        self.P_test = self.P[1::2, :]

        self.V_train = self.V[::2, :]
        self.V_test = self.V[1::2, :]

        self.X_train = X[:, ::2]
        self.X_test = X[:, 1::2]

        xyz = np.zeros((self.n_points, 3))
        self.gpr = gpr.GPR(self.X_train, self.n_features, xyz, self.P_train)
        
    def teardown_method(self, method):
        pass


    def test_centering_and_scaling_parameters(self):
        P0 = self.gpr.scale_GPR_data(self.P_train, 'std')
        
        P_cnt = np.zeros_like(self.P_train)
        P_scl = np.zeros_like(self.P_train)
        for i in range(self.P_train.shape[1]):
            P_cnt[:,i] = np.mean(self.P_train[:,i])
            P_scl[:,i] = np.std(self.P_train[:,i])

        P0_check = (self.P_train - P_cnt)/P_scl 

        np.testing.assert_array_equal(P_cnt, self.gpr.P_cnt)
        np.testing.assert_array_equal(P_scl, self.gpr.P_scl)
        np.testing.assert_array_equal(P0_check, P0)

    def test_fit(self):
        self.gpr.fit(scaleX_type='none')
        np.testing.assert_allclose(np.abs(self.U), np.abs(self.gpr.Ur), atol=1e-5)
        
    def test_predict(self):
        self.gpr.fit(scaleX_type='none')
        self.gpr.train()
        A_pred, _ = self.gpr.predict(self.P_test)
        X_pred = self.gpr.reconstruct(A_pred)

        np.testing.assert_allclose(self.X_test, X_pred, rtol=1e-10, atol=5e-1)
        
        

        
        

    