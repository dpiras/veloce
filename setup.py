from setuptools import setup, find_packages
import shutil

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

PACKAGENAME = 'velocemu'

setup(
    name='velocemu',
    version="0.1.0",
    author='Davide Piras',
    author_email='dr.davide.piras@gmail.com',
    description='The velocity covariance emulator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dpiras/veloce',
    license='GNU General Public License v3.0 (GPLv3)',
    packages=find_packages(),
    package_data= {'velocemu': ['trained_models/*', 'mock_data/*']},    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=['scipy',
                      'matplotlib',
                      'numpy',
                      'jax==0.4.35',
                      'cosmopower-jax',
                      'tensorflow==2.13',
                      'tqdm',
                      'corner']
                      )

