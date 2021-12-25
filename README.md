# OpenMEASURE

OpenMEASURE is an open source software for soft sensing applications.

## Installation

Run the following command to install:

'''
pip install OpenMEASURE
'''

## Usage

'''
from sparse_sensing import SPR
import numpy as np

X = np.random.rand(5,16)
n_features = 4

# create Sparse Placement for Reconstrucion instance
spr = SPR(X, n_features)

# calculate the optimal measurement matrix
C = spr.optimal_placement
'''


