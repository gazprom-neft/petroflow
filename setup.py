"""PetroFlow is a library that allows processing well data (logs, core images
etc.) and conveniently train machine learning models.
"""

import re
from setuptools import setup, find_packages


with open('petroflow/__init__.py', 'r') as f:
    VERSION = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)


setup(
    name='petroflow',
    packages=find_packages(exclude=['examples']),
    version=VERSION,
    url='https://github.com/gazprom-neft/petroflow',
    license='CC BY-NC-SA 4.0',
    author='Gazprom Neft DS team',
    author_email='rhudor@gmail.com',
    description='A framework for well data processing',
    zip_safe=False,
    platforms='any',
    install_requires=[
        'numpy>=1.17.2',
        'scipy>=1.3.1',
        'pandas>=0.24.2',
        'numba>=0.46.0',
        'scikit-image>=0.15.0',
        'scikit-learn>=0.21.3',
        'opencv-python>=4.1.1',
        'matplotlib>=3.0.1',
        'seaborn>=0.9.0',
        'plotly>=4.1.1',
        'pint>=0.11',
        'pillow>=6.2.0',
        'blosc==1.8.1',
        'feather_format>=0.4.0',
        'lasio>=0.23',
        'dill>=0.3.0',
        'multiprocess>=0.70.9',
        'tqdm>=4.36.1',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.12.0'],
        'tensorflow-gpu': ['tensorflow-gpu>=1.12.0'],
        'torch': ['torch>=1.3.0']
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: CC BY-NC-SA 4.0',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering'
    ],
)
