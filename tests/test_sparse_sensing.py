'''
MODULE: sparse_sensing.py
@Authors:
    A. Procacci [1]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
@Contacts:
    alberto.procacci@ulb.be
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: alberto.procacci@ulb.be
'''

import unittest
import numpy as np
import sparse_sensing as sps

class testSPR(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1.,2.],
                           [3.,5.],
                           [-2.,4.],
                           [4.,6.]])
        self.n_features = 2
        self.mean_matrix = np.array([[2., 3.5],
                                     [1., 5.]])
        self.std_matrix = np.array([[1., 1.5],
                                    [3., 1.]])
        self.X0 = np.array([[-1., -1],
                            [ 1.,  1.],
                            [-1., -1],
                            [ 1.,  1.]])
        self.y = np.array([[2.5, 0],
                           [1, 1]])
        self.y0 = np.array([-0.2, -1.])
        self.x = np.array([2.12, 3.38, -1., 5.])
        self.x0 = np.array([-0.5, 0.5, -2., 1.])
        # self.x0_rec = np.array([-0.2, -4.6, 1., -1.])
        self.x_rec = np.array([2.5, -3, 5., 1.])
        
        self.U = np.array([[-0.5, 0.5],
                            [ 0.5,  0.83],
                            [-0.5, 0.17],
                            [ 0.5,  -0.17]])
        self.C = np.array([[1., 0., 0., 0.],
                           [0., 0., 0., 1.]])
        self.C_opt = np.array([[0., 1., 0., 0.],
                               [1., 0., 0., 0.]])
        
        
    def tearDown(self):
        pass
    
    def test_init(self):
        with self.assertRaises(TypeError):
            sps.SPR('X', 5)
    
    def test_scale_data(self):
        spr = sps.SPR(self.X, self.n_features)
        X0 = spr.scale_data()

        np.testing.assert_array_equal(np.round(spr.mean_matrix,2), self.mean_matrix)
        np.testing.assert_array_equal(np.round(spr.std_matrix,2), self.std_matrix)
        np.testing.assert_array_equal(np.round(X0,2), self.X0)
        
        np.testing.assert_array_equal(np.round(np.mean(X0, axis=0),2), np.array([0,0]))
        np.testing.assert_array_equal(np.round(np.var(X0, axis=0),2), np.array([1,1]))

    def test_scale_vector(self):
        spr = sps.SPR(self.X, self.n_features)
        spr.scale_data()
        y0 = spr.scale_vector(self.y, 'standard')
        
        np.testing.assert_array_equal(np.round(y0,2), self.y0)
        with self.assertRaises(NotImplementedError):
            spr.scale_vector(self.y, 'test')      

    def test_unscale_data(self):
        spr = sps.SPR(self.X, self.n_features)
        spr.scale_data()
        x = spr.unscale_data(self.x0, 'standard')

        np.testing.assert_array_equal(np.round(x,2), self.x)
        with self.assertRaises(NotImplementedError):
            spr.unscale_data(self.x0, 'test')

    def test_decomposition(self):
        spr = sps.SPR(self.X, self.n_features)
        X0 = spr.scale_data()
        U, exp_variance = spr.decomposition(X0, 'POD')
    
        np.testing.assert_array_equal(np.round(U,2), self.U)
        with self.assertRaises(NotImplementedError):
            U = spr.decomposition(X0, 'test')
        
    def test_reduction(self):
        spr = sps.SPR(self.X, self.n_features)
        X0 = spr.scale_data()
        U, exp_variance = spr.decomposition(X0, 'POD')
        Ur = spr.reduction(U, exp_variance, 'number', 1)
        
        np.testing.assert_array_equal(Ur, U[:,0, np.newaxis])
        
        with self.assertRaises(ValueError):
            Ur = spr.reduction(U, exp_variance, 'variance', 200)
            Ur = spr.reduction(U, exp_variance, 'number', 5)
            Ur = spr.reduction(U, exp_variance, 'test', 1)
        
        with self.assertRaises(TypeError):
            Ur = spr.reduction(U, exp_variance, 'number', 3.5)
        
    def test_optimal_placement(self):
        spr = sps.SPR(self.X, self.n_features)
        C_opt = spr.optimal_placement('standard', 'number', 2)
        
        np.testing.assert_array_equal(C_opt, self.C_opt)
        
    def test_fit_predict(self):
        spr = sps.SPR(self.X, self.n_features)
        x_rec = spr.fit_predict(self.C, self.y, 'standard', 'number', 2)
        
        np.testing.assert_array_equal(np.round(x_rec,2), self.x_rec)
        with self.assertRaises(ValueError):
            x_rec = spr.fit_predict(self.C[0,:], self.y, 'standard', 'number', 2)
            x_rec = spr.fit_predict(self.C[:,:2], self.y, 'standard', 'number', 2)
            x_rec = spr.fit_predict(self.C, self.y[:,0], 'standard', 'number', 2)

    def test_predict(self):
        spr = sps.SPR(self.X, self.n_features)
        with self.assertRaises(AttributeError):
            spr.predict(self.y)
            
        spr.fit_predict(self.C, self.y, 'standard', 'number', 2)
        x_rec = spr.predict(self.y)
        
        np.testing.assert_array_equal(np.round(x_rec,2), self.x_rec)
 

if __name__ == '__main__':
    unittest.main()