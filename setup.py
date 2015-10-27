from setuptools import setup

setup(name = 'colossus',
	version = '0.9.5',
	description = ' Cosmology, halo and large-scale structure tools',
	url = 'https://bitbucket.org/bdiemer/colossus',
	author = 'Benedikt Diemer',
	author_email = 'benedikt.diemer@cfa.harvard.edu',
	license = 'MIT',
	requires=['numpy', 'scipy'],
	packages = ['colossus', 'colossus.demos'],
	zip_safe = False)
