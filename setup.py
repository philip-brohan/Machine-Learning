"""Setup configuration for the ML-Experiments.

"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
import glob

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ML_Experiments',
    version='0.0.1',
    description="Philip's Machine Learning Experiments",

    # From README - see above
    long_description=long_description,
    #long_description_content_type='text/x-rst',

    url='https://brohan.org/ML_Experiments/',

    author='Philip Brohan',
    author_email='philip.brohan@metofice.gov.uk',

    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3.6',
    ],

    # Keywords for your project. What does your project relate to?
    keywords='weather tensorflow machine-learning climate',

    # Automatically find the software to be included
    package_dir = {'': 'experiments'},
    packages=find_packages(where='experiments'),


    # Other packages that your project depends on.
    install_requires=[
        'scitools-iris>=2.2',
        'cartopy>=0.16',
        'IRData>0.0',
        'Meteorographica>0.0',
        'numpy>=1.15.2',
        'scipy>=1.1.0',
        'pandas>=0.23.4',
        'tensorflow>=1.13',
        'matplotlib>=2.2.3',
    ],


)
