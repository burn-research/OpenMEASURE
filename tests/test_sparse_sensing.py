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
import sparse_sensing as sps

class testSPR(unittest.TestCase):
    def setUp(self):
        pass
        
    def tearDown(self):
        pass
    
    def test_init(self):
        with self.assertRaises(TypeError):
            sps.SPR('X', 5)
    
    def test_scale_data(self):
        pass

    def test_scale_vector(self):
        spr = sps.SPR(self.X, self.n_features)
        spr.scale_data()
        
        with self.assertRaises(NotImplementedError):
            spr.scale_vector(self.y, 'test')      

    def test_unscale_data(self):
        spr = sps.SPR(self.X, self.n_features)
        spr.scale_data()

        with self.assertRaises(NotImplementedError):
            spr.unscale_data(self.x0, 'test')

    def test_decomposition(self):
        spr = sps.SPR(self.X, self.n_features)
        X0 = spr.scale_data()
    
        with self.assertRaises(NotImplementedError):
            U = spr.decomposition(X0, 'test')
        
    def test_reduction(self):
        spr = sps.SPR(self.X, self.n_features)
        X0 = spr.scale_data()
        U, exp_variance = spr.decomposition(X0, 'POD')
        
        with self.assertRaises(ValueError):
            Ur = spr.reduction(U, exp_variance, 'variance', 200)
            Ur = spr.reduction(U, exp_variance, 'number', 5)
            Ur = spr.reduction(U, exp_variance, 'test', 1)
        
        with self.assertRaises(TypeError):
            Ur = spr.reduction(U, exp_variance, 'number', 3.5)
        
    def test_optimal_placement(self):
        pass
        
    def test_fit_predict(self):
        spr = sps.SPR(self.X, self.n_features)
        
        with self.assertRaises(ValueError):
            x_rec = spr.fit_predict(self.C[0,:], self.y, 'standard', 'number', 2)
            x_rec = spr.fit_predict(self.C[:,:2], self.y, 'standard', 'number', 2)
            x_rec = spr.fit_predict(self.C, self.y[:,0], 'standard', 'number', 2)

    def test_predict(self):
        spr = sps.SPR(self.X, self.n_features)
        with self.assertRaises(AttributeError):
            spr.predict(self.y)
 

if __name__ == '__main__':
    unittest.main()