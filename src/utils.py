'''
MODULE: utils.py
@Authors:
    A. Procacci [1]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
@Contacts:
    alberto.procacci@ulb.be
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: alberto.procacci@ulb.be
'''

import numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix

def resample_to_grid(mesh, X, dimensions, verbose=False):
    '''
    Samples the mesh on a structured mesh of size dimensions.
    
    Parameters
    ----------
    mesh : pyvista mesh
        Original mesh imported using pyvista. 
    
    X : numpy array
        Data matrix of dimensions (n, m) where n is the number of 
        features multiplied by the number of cells, while m is the
        number of snapshots included.
    
    dimensions : list
        If the voxels mesh spans the entire space of the original mesh,
        it contains 3 scalars for the number of cells in the
        x, y, z direction. If the voxels mesh is a subset of the original 
        mesh, it contains the 3 numpy array for the coordinates of the 
        points in the x, y and z direction. Each numpy array has dimensions
        (n_x, n_y, n_z). 

    verbose : bool, optional
        If True, it will output the progress of the interpolation
        algorithm. Default is False.
    
    Returns
    -------
    mesh_int : pyvista mesh
        Structured mesh.
    
    X_int : numpy array
        X interpolated on the structured mesh.
    
    xyz_int : numpy array
        Array of dimensions (n_cells, 3) containing the position of 
        the cell centers.

    '''
    n_cells = mesh.n_cells
    n_features = X.shape[0]//n_cells
    
    if type(dimensions[0]) is np.ndarray:
        grid = pv.StructuredGrid(dimensions[0], 
                                 dimensions[1], 
                                 dimensions[2])
    
    elif type(dimensions[0]) is int:
        grid = pv.create_grid(mesh, dimensions=dimensions)
    
    else:
        raise TypeError('The objects in the list must be either integers or numpy arrays')
    
    n_cells_grid = grid.n_cells
    
    # Store every fields in the mesh object
    for i in range(n_features):
        for j in range(X.shape[1]):
            mesh.cell_data.set_array(X[i*n_cells:(i+1)*n_cells, j], f'f{i}/t{j}')
            if verbose:
                print(f'Storing snapshot {j+1}/{X.shape[1]}', end='\r', flush=True)

    # Sample the original mesh with the grid
    mesh_int = grid.sample(mesh, pass_cell_data=True, progress_bar=verbose)
    mesh_int = mesh_int.point_data_to_cell_data()
    n_cells_int = mesh_int.n_cells
    
    # Retrieve all the fields
    X_int = np.zeros((n_features*n_cells_int, X.shape[1]))
    for i in range(n_features):
        for j in range(X.shape[1]):
            tmp = mesh_int.cell_data[f'f{i}/t{j}']
            X_int[i*n_cells_int:(i+1)*n_cells_int, j] = tmp
            if verbose:
                print(f'Retrieving snapshot {j+1}/{X.shape[1]}', end='\r', flush=True)

    # Clear the data in the mesh objects
    mesh.clear_data()
    mesh_int.clear_data()

    xyz_int = mesh_int.cell_centers().points

    return mesh_int, X_int, xyz_int

class camera():
    '''
    Class to compute the 2D projections of a volumetric object.
    Points and vectors are represented by 4D numpy arrays. The first
    3 scalars are the x, y, z coordinates while the last scalar is
    0 for vectors and 1 for points.

    Attributes
    ----------
    p_cam : numpy array
        Position of the camera, dimension (4,).
    
    theta : numpy array
        Camera angles in radiants, dimension (3,).

    f_length : float
        Focal length in meters. This is a lens' property,
        and not the distance between the lens and the sensor. 

    n_aper : float
        Aperture of the lens.

    d_sensor : float
        Distance between the sensor plane and the lens.
        
    sensor_size_px : numpy array
        Size of the sensor in the x and y direction, in pixels.

    px_size : float
        Size of a single pixel in m. It is assumed that the pixels
        are square.
        
    n_pixels : int
        Number of pixels on the sensor.

    sensor_size_m : numpy array
        Size of the sensor in the x and y direction, in meters.
    
    d : float
        Distance between the camera and the the global center [0,0,0,1].
        
    d_object : float
        Distance between the lens and the object plane. Computed using
        the thin lens approximation 1/f = 1/d_sensor + 1/d_object
        
    m : float
        Magnifing factor m = d_sensor/d_object

    Methods
    ----------
    generate_camera()
        Creates a pyvista object that can be plotted to understand the 
        position of the camera relative to the object.
    
    project(obj_mesh, type_rec='parallel', N_rand=10, verbose=False)
        Computes the matrix C in the projection p = C f. 
    
    '''


    def __init__(self, p_cam, theta, f_length, n_aper, d_sensor, sensor_size_px, px_size):
        
        '''    
        Parameters
        ----------
        p_cam : numpy array
            Position of the camera, dimension (4,).
    
        theta : numpy array
            Camera angles in radiants, dimension (3,).

        f_length : float
            Focal length in meters. This is a lens' property,
            and not the distance between the lens and the sensor. 

        n_aper : float
            Aperture of the lens.

        d_sensor : float
            Distance between the sensor plane and the lens.
            
        sensor_size_px : numpy array
            Size of the sensor in the x and y direction, in pixels.

        px_size : float
            Size of a single pixel in m. It is assumed that the pixels
            are square.

        Returns
        -------
        None.

        '''

        self.p_cam = p_cam
        self.theta = theta
        self.f_length = f_length 
        self.n_aper = n_aper 
        self.d_sensor = d_sensor 
        self.sensor_size_px = sensor_size_px
        self.px_size = px_size 
    
        self.n_pixels = int(sensor_size_px[0]*sensor_size_px[1])
        self.sensor_size_m = px_size * sensor_size_px 
        self.d = np.linalg.norm(p_cam - np.array([0,0,0,1])) 
        
        m = d_sensor/f_length - 1
        if m > 1e-2:
            self.m = m    
            self.d_object = f_length/(1-f_length/d_sensor)
        else:
            self.m = 0
            self.d_object = -1
        
    def _extr_matrix(self):
        '''
        Computes the extrinsic camera matrix.
        '''

        E = np.zeros((4,4))
        
        R_x = np.array([[1, 0, 0, 0],
                        [0, np.cos(self.theta[0]), -np.sin(self.theta[0]), 0],
                        [0, np.sin(self.theta[0]), np.cos(self.theta[0]), 0],
                        [0, 0, 0, 1]])
        
        R_y = np.array([[np.cos(self.theta[1]), 0, np.sin(self.theta[1]), 0],
                        [0, 1, 0, 0],
                        [-np.sin(self.theta[1]), 0, np.cos(self.theta[1]), 0],
                        [0, 0, 0, 1]])
        
        R_z = np.array([[np.cos(self.theta[2]), -np.sin(self.theta[2]), 0, 0],
                        [np.sin(self.theta[2]), np.cos(self.theta[2]), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        
        E = np.linalg.multi_dot([R_x, R_y, R_z])
        T = -E @ self.p_cam
        E[:-1, -1] = T[:-1]
        
        return E
    
    def _sensor_coordinates(self):
        '''
        Computes the local pixel coordinates.
        '''

        xs = np.linspace(-self.sensor_size_m[0]/2 + self.px_size/2, 
                          self.sensor_size_m[0]/2 - self.px_size/2,
                          self.sensor_size_px[0])
        
        ys = np.linspace(self.sensor_size_m[1]/2 - self.px_size/2, 
                         -self.sensor_size_m[1]/2 + self.px_size/2,
                          self.sensor_size_px[1])
        
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        
        xyz_sl = np.zeros((xs_grid.size, 4))
        xyz_sl[:, 0] = xs_grid.flatten()
        xyz_sl[:, 1] = ys_grid.flatten()
        xyz_sl[:, 3] = 1.
        
        return xyz_sl

    def _random_lens(self, N_rand):
        '''
        Computes random points on the lens for the raytracing 
        algorithm. The points are in local coordinates.
        '''

        # Radius of the lens
        R = self.f_length/(self.n_aper*2) 
        
        rng = np.random.default_rng()
        r = R * np.sqrt(rng.random(size=N_rand))
        theta = rng.random(size=N_rand) * 2 * np.pi
        
        xyz_ll = np.zeros((N_rand, 4))
        xyz_ll[:, 0] = r * np.cos(theta)
        xyz_ll[:, 1] = r * np.sin(theta)
        xyz_ll[:, 2] = -self.d_sensor
        xyz_ll[:, 3] = 1.

        return xyz_ll
    
    def generate_camera(self):
        '''
        Return the camera object for plotting. 

        Returns
        -------
        groupg : pyvista object
            Camera object in the global coordinates.

        '''

        box_length = 2*self.f_length
        boxl = pv.Box([-box_length/2, box_length/2, 
                      -box_length/2, box_length/2, 
                      -box_length/2, box_length/2])
        
        conel = pv.Cone(center=(0, 0, -3*box_length/4), 
                        direction=(0, 0, 1),
                        height=box_length/2,
                        radius=box_length/4,
                        resolution=20)

        linel = pv.Line((0,0,0), (0,0,-2*self.d), resolution=2)
        groupl = boxl.merge([conel, linel])

        E = self._extr_matrix()  
        E_inv = np.linalg.inv(E)
        groupg = groupl.transform(E_inv)
        
        return groupg

    def project(self, obj_mesh, type_rec='parallel', N_rand=10, verbose=False):
        '''
        Return the projection of the volumetric object. 
        
        Parameters
        ----------
        obj_mesh : pyvista mesh
            Mesh of the volumetric object. Preferably a voxel mesh.

        type_rec : str, optional
            Algorithm used for the projection. If 'parallel', it will 
            compute the projection as a set of parallel rays. If 'pinhole',
            it will use the pinhole model. If 'thin_lens', it will use
            the thin lens model. Default is 'thin lens'.

        N_rand : int, optional.
            Number of random rays computed for each pixel. Not used for
            parallel projection. Default is 10.

        verbose : bool, optional
        If True, it will output the progress of the projection
            algorithm. Default is False.

        Returns
        -------
        C : scipy sparse matrix
            Sparse matrix that can be used for the projection.

        '''

        E = self._extr_matrix()  
        E_inv = np.linalg.inv(E)
        
        xyz_sl = self._sensor_coordinates()
        n_cells = obj_mesh.n_cells
        
        # Init rows and columns for C
        rows = [] 
        columns = []

        if type_rec == 'parallel':
            
            # Compute the mirror of the sensor for parallel rays
            xyz_sl_mirror = np.zeros_like(xyz_sl)
            xyz_sl_mirror[:,[0,1,3]] = xyz_sl[:,[0,1,3]]
            xyz_sl_mirror[:,2] = -2*self.d
            
            for i in range(self.n_pixels):
                p1l = xyz_sl[i, :]
                p2l = xyz_sl_mirror[i, :]

                p1g = E_inv @ p1l
                p2g = E_inv @ p2l

                l = obj_mesh.find_cells_intersecting_line(p1g[:-1], p2g[:-1])
            
                rows.extend([i for k in range(len(l))])
                columns.extend(l)

                if verbose:
                    print(f'Pixel {i+1}/{self.n_pixels}', end='\r',
                          flush=True)

        elif type_rec == 'pinhole':
            # Compute the center of the lens
            pll = np.array([0,0,-self.d_sensor,1])

            counts = [[] for i in range(self.n_pixels)]
            for i in range(self.n_pixels):
                # Compute random deviations in x and y
                rng = np.random.default_rng()
                dx_rand = self.px_size * (rng.random(size=N_rand) - 0.5)
                dy_rand = self.px_size * (rng.random(size=N_rand) - 0.5)
                
                for j in range(N_rand):
                    # Compute the point on sensor
                    psl = np.zeros((4,))
                    psl[0] = xyz_sl[i,0] + dx_rand[j]
                    psl[1] = xyz_sl[i,1] + dy_rand[j]
                    psl[3] = 1.

                    # Compute the final point
                    vfl = (pll - psl)/np.linalg.norm(pll - psl)
                    pfl = psl + 2*self.d*vfl
                    
                    psg = E_inv @ psl
                    pfg = E_inv @ pfl
                    
                    l = obj_mesh.find_cells_intersecting_line(psg[:-1], pfg[:-1])
                    
                    # Check if the voxel was already included
                    add = list(set(l) - set(counts[i]))
                    counts[i].extend(add)
                
                rows.extend([i for k in range(len(counts[i]))])
                columns.extend(counts[i])   

                if verbose:
                    print(f'Pixel {i+1}/{self.n_pixels}', end='\r',
                          flush=True)             

        elif type_rec == 'thin_lens':
            # Check if thin_lens is adeguate
            if self.m == 0:
                raise ValueError('For focus at infinity use a different model')
            
            # Compute the random points on the lens
            xyz_ll = self._random_lens(xyz_sl.shape[0] * N_rand)
            
            counts = [[] for i in range(xyz_sl.shape[0])]
            for i in range(self.n_pixels):
                rng = np.random.default_rng()
                dx_rand = self.px_size * (rng.random(size=N_rand) - 0.5)
                dy_rand = self.px_size * (rng.random(size=N_rand) - 0.5)
            
                for j in range(N_rand):
                    psl = np.zeros((4,))
                    psl[0] = xyz_sl[i,0] + dx_rand[j]
                    psl[1] = xyz_sl[i,1] + dy_rand[j]
                    psl[3] = 1.

                    pll = xyz_ll[i, :]
                
                    # Compute the point on the object plane
                    pol = np.zeros_like(pll)
                    pol[0] = -psl[0]/self.m
                    pol[1] = -psl[1]/self.m
                    pol[2] = -(self.d_object + self.d_sensor)
                    pol[3] = 1.
                
                    vfl = (pol - pll)/np.linalg.norm(pol - pll)
                    pfl = pll + 2*self.d*vfl
                
                    plg = E_inv @ pll
                    pfg = E_inv @ pfl

                    l = obj_mesh.find_cells_intersecting_line(plg[:-1], pfg[:-1])
                    add = list(set(l) - set(counts[i]))
                    counts[i].extend(add)
                
                rows.extend([i for k in range(len(counts[i]))])
                columns.extend(counts[i])

                if verbose:
                    print(f'Pixel {i+1}/{self.n_pixels}', end='\r',
                          flush=True)
                
        counts = np.array([1 for k in range(len(rows))])
        C = csr_matrix((counts, (rows, columns)), 
                        shape=(xyz_sl.shape[0], n_cells))

        return C