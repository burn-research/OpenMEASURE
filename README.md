# OpenMEASURE

OpenMEASURE is an open source software for soft sensing applications.

## Installation

Run the following command to install:

```python
pip install openmeasure
```

The following packages will be installed:
* Numpy >= 1.19
* Scipy >= 1.5
* Gpytorch >= 1.4.2

## Techniques

* Dimensinality reduction (POD)

* Reduced Order Model via GPR

* Sparse sensing:
    * Optimal sensor placement (QR decomposition)
    * Sparse placement for reconstruction

## Usage

```python
import numpy as np
from gpr import GPR
from sparse_sensing import SPR
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Replace this with the path where you saved the data directory
path = './data/ROM/'

# This is a n x m matrix where n = 165258 is the number of cells times the number of features
# and m = 41 is the number of simulations.
X_train = np.load(path + 'X_2D_train.npy')

# This is a n x 4 matrix containing the 4 testing simulations
X_test = np.load(path + 'X_2D_test.npy')

features = ['T', 'CH4', 'O2', 'CO2', 'H2O', 'H2', 'OH', 'CO', 'NOx']
n_features = len(features)

# This is the file containing the x,z positions of the cells
xz = np.load(path + 'xz.npy')
n_cells = xz.shape[0]

# This reads the files containing the parameters (D, H2, phi) with which 
# the simulation were computed
P_train = np.genfromtxt(path + 'parameters_train.csv', delimiter=',', skip_header=1)
P_test = np.genfromtxt(path + 'parameters_test.csv', delimiter=',', skip_header=1)

# Load the outline the mesh (for plotting)
mesh_outline = np.genfromtxt(path + 'mesh_outline.csv', delimiter=',', skip_header=1)

#---------------------------------Plotting utilities--------------------------------------------------
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

    
    ax.set_xlabel('$x (\mathrm{m})$', fontsize=8)
    ax.set_ylabel('$z (\mathrm{m})$', fontsize=8)
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
        ax.set_xlabel('$x (\mathrm{m})$')
        if i == 0:
            ax.set_ylabel('$z (\mathrm{m})$')
    
    fig.subplots_adjust(bottom=0., top=1., left=0., right=0.85, wspace=0.02, hspace=0.02)
    start = axs[1].get_position().bounds[1]
    height = axs[1].get_position().bounds[3]
    
    cb_ax = fig.add_axes([0.9, start, 0.05, height])
    cmap = mpl.cm.get_cmap(cmap_name, n_levels)
    norm = mpl.colors.Normalize(vmin=z_min, vmax=z_max)
    
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, 
                orientation='vertical', label=cbar_label)
    
    plt.show()

#---------------------------------Sparse sensing--------------------------------------------------

spr = SPR(X_train, n_features) # Create the spr object

# Compute the optimal measurement matrix using qr decomposition
n_sensors = 14
C_qr = spr.optimal_placement(select_modes='number', n_modes=n_sensors)

# Get the sensors positions and features
xz_sensors = np.zeros((n_sensors, 4))
for i in range(n_sensors):
    index = np.argmax(C_qr[i,:])
    xz_sensors[i,:2] = xz[index % n_cells, :]
    xz_sensors[i,2] = index // n_cells

plot_sensors(xz_sensors, features, mesh_outline)

# Sample a test simulation using the optimal qr matrix
y_qr = np.zeros((n_sensors,2))
y_qr[:,0] = C_qr @ X_test[:,3]

for i in range(n_sensors):
    y_qr[i,1] = np.argmax(C_qr[i,:]) // n_cells

# Fit the model and predict the low-dim vector (ap) and the high-dim solution (xp)
ap, xp = spr.fit_predict(C_qr, y_qr)

# Select the feature to plot
str_ind = 'T'
ind = features.index(str_ind)

plot_contours_tri(xz[:,0], xz[:,1], [X_test[ind*n_cells:(ind+1)*n_cells, 3], 
                xp[ind*n_cells:(ind+1)*n_cells]], cbar_label=str_ind)

#------------------------------------GPR ROM--------------------------------------------------
# Create the gpr object
gpr = GPR(X_train, P_train, n_features)

# Calculates the POD coefficients ap and the uncertainty for the test simulations
Ap, Sigmap = gpr.fit_predict(P_test, verbose=True)

# Reconstruct the high-dimensional state from the POD coefficients
Xp = gpr.reconstruct(Ap)

# Select the feature to plot
str_ind = 'OH'
ind = features.index(str_ind)

x_test = X_test[ind*n_cells:(ind+1)*n_cells,3]
xp_test = Xp[ind*n_cells:(ind+1)*n_cells, 3]

plot_contours_tri(xz[:,0], xz[:,1], [x_test, xp_test], cbar_label='str_ind')

```


