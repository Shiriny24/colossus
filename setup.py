from setuptools import setup

setup(name = 'colossus',
	version = '1.0.6',
	description = ' Cosmology, halo and large-scale structure tools',
	url = 'https://bitbucket.org/bdiemer/colossus',
	author = 'Benedikt Diemer',
	author_email = 'benedikt.diemer@cfa.harvard.edu',
	license = 'MIT',
	requires=['numpy', 'scipy'],
	packages = ['colossus', 
				'colossus.cosmology', 
				'colossus.demos', 
				'colossus.halo', 
				'colossus.tests', 
				'colossus.utils'],
	zip_safe = False)
