from setuptools import setup
import io

from colossus.utils import settings

with io.open('README.rst', encoding = 'utf-8') as f:
	long_description = f.read()
	
setup(name = 'colossus',
	version = settings.__version__,
	description = 'Cosmology, halo, and large-scale structure tools',
	long_description = long_description,
	url = 'https://bitbucket.org/bdiemer/colossus',
	author = 'Benedikt Diemer',
	author_email = 'benedikt.diemer@cfa.harvard.edu',
	license = 'MIT',
	requires = ['numpy', 'scipy', 'six'],
	packages = ['colossus', 
				'colossus.cosmology', 
				'colossus.lss', 
				'colossus.halo', 
				'colossus.tests', 
				'colossus.utils'],
	classifiers = [
				'Environment :: Console',
				'Intended Audience :: Developers',
				'Intended Audience :: Education',
				'Intended Audience :: Science/Research',
				'License :: OSI Approved :: MIT License',
				'Operating System :: OS Independent',
				'Programming Language :: Python :: 2',
				'Programming Language :: Python :: 2.7',
				'Programming Language :: Python :: 3',
				'Programming Language :: Python :: 3.3',
				'Programming Language :: Python :: 3.4',
				'Programming Language :: Python :: 3.5',
				'Programming Language :: Python :: 3.6',
				'Programming Language :: Python :: 3.7',
				'Topic :: Scientific/Engineering :: Astronomy',
				'Topic :: Scientific/Engineering :: Physics'
				],
	zip_safe = False)
