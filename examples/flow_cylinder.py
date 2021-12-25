import numpy as np
import matplotlib as mp
from sparse_sensing import SPR


X = np.load('../data/flow_cylinder/X.npy')
x_test = np.load('../data/flow_cylinder/x_test.npy')

n_points = X.shape[0] // 2
m = X.shape[1]

ndx = 199
ndy = 449

xdim = np.arange(0, ndx)
ydim = np.arange(0, ndy)

xgrid, ygrid = np.meshgrid(xdim, ydim)

f = 1
u = X[f*n_points:(f+1)*n_points, 6]
ugrid = u.reshape(ndy, ndx)

mp.pyplot.contourf(xgrid, ygrid, ugrid)
mp.pyplot.show()
