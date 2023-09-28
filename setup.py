from setuptools import setup, find_packages

setup(
    name='linear-solver',
    version='0.0.1',
    description='Linear Programming solver wrapper for R.',
    author='Siddhi47',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'rpy2',
        'pandas',
        ],
    )

