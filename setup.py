from setuptools import setup

setup(name = 'colossus',
	version = '0.9.1',
	description = ' Cosmology, halo and large-scale structure tools',
	url = 'https://bitbucket.org/bdiemer/colossus',
	author = 'Benedikt Diemer',
	author_email = 'bdiemer@oddjob.uchicago.edu',
	license = 'MIT',
	requires=['numpy', 'scipy'],
	packages = ['colossus', 'colossus.demos'],
	zip_safe = False)
