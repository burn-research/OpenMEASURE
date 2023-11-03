from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
      name='OpenMEASURE',
      version='0.3.1',
      description='Python package for soft sensing applications',
      py_modules=['sparse_sensing', 'gpr', 'cokriging', 'utils'],
      package_dir={'':'src'},
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=['numpy>=1.24.2', 'scipy>=1.10.1', 'gpytorch>=1.9.1', 
                        'cvxpy>=1.3.1', 'openmdao>=3.15.0', 'pyvista>=0.41.1'],
      url='https://github.com/albertoprocacci/OpenMEASURE',
      author='Alberto Procacci',
      author_email='alberto.procacci@gmail.com',
      )
