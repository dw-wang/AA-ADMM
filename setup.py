from setuptools import setup, find_packages

setup(name='aa_admm',
      version='0.1',
      description='A Python package for type-II Anderson accelerated alternating direction method of multipliers (AA-ADMM).',
      author='Dawei Wang',
      author_email='dawei.wang@uwaterloo.ca',
      license='Apache License, Version 2.0',
      packages=find_packages(),
      install_requires=['matplotlib',
                        'cvxpy >= 1.0.25',
                        'numpy >= 1.14',
                        'scipy >= 1.2.1'])
