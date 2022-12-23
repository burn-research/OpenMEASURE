from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
      name='OpenMEASURE',
      version='0.1.12',
      description='Python package for soft sensing applications',
      py_modules=['sparse_sensing', 'gpr'],
      package_dir={'':'src'},
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=['numpy>=1.19', 'scipy>=1.7.1', 'gpytorch>=1.5.1', 
                        'cvxpy>=1.1.3'],
      url='https://github.com/albertoprocacci/OpenMEASURE',
      author='Alberto Procacci',
      author_email='alberto.procacci@gmail.com',
      )
