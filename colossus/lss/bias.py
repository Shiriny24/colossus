###################################################################################################
#
# bias.py                   (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module implements the following functions related to halo bias:

.. autosummary:: 
    haloBiasFromNu
    haloBias
	twoHaloTerm
   
Currently, only the bias model of Tinker et al. 2010 is implemented.
   
---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np

from colossus.utils import constants
from colossus import defaults
from colossus.cosmology import cosmology
from colossus.lss import lss
from colossus.halo import mass_so

###################################################################################################

models = ['sc', 'sheth01', 'tinker10']
"""A list of all implemented bias models."""

###################################################################################################
# HALO BIAS
###################################################################################################

def haloBiasFromNu(nu, z, mdef, model = defaults.HALO_BIAS_MODEL):
	"""
	The halo bias at a given peak height. 

	The halo bias. The mass definition, mdef, must correspond to the mass that was used to 
	evaluate the peak height.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	nu: array_like
		Peak height; can be a number or a numpy array.
	z: array_like
		Redshift; can be a number or a numpy array.
	mdef: str
		The mass definition corresponding to the mass that was used to evaluate the peak height.
	model: str
		The bias model used.
	
	Returns
	-----------------------------------------------------------------------------------------------
	bias: array_like
		Halo bias; has the same dimensions as nu or z.

	See also
	-----------------------------------------------------------------------------------------------
	haloBias: The halo bias at a given mass. 
	"""
	
	if model == 'sc':
		bias = modelSC(nu, z)
	elif model == 'tinker10':
		bias = modelTinker10(nu, z, mdef)
	elif model == 'sheth01':
		bias = modelSheth01(nu, z)
	else:
		msg = 'Unkown model, %s.' % (model)
		raise Exception(msg)

	return bias

###################################################################################################

def haloBias(M, z, mdef, model = defaults.HALO_BIAS_MODEL):
	"""
	The halo bias at a given mass. 

	This function is a wrapper around haloBiasFromNu.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: array_like
		Redshift; can be a number or a numpy array.
	mdef: str
		The mass definition in which M is given.
	model: str
		The bias model used.

	Returns
	-----------------------------------------------------------------------------------------------
	bias: array_like
		Halo bias; has the same dimensions as M or z.

	See also
	-----------------------------------------------------------------------------------------------
	haloBiasFromNu: The halo bias at a given peak height. 
	"""
		
	nu = lss.peakHeight(M, z)
	bias = haloBiasFromNu(nu, z, mdef, model = model)
	
	return bias

###################################################################################################

def twoHaloTerm(r, M, z, mdef, model = defaults.HALO_BIAS_MODEL):
	"""
	The 2-halo term as a function of radius and halo mass. 

	The 2-halo term in the halo-matter correlation function describes the excess density around 
	halos due to the proximity of other halos. This contribution can be approximated as the matter-
	matter correlation function times a linear bias which depends on the peak height of the halo.
	
	Sometimes this term includes an additional factor of the mean density which is omitted here. 
	
	Parameters
	-----------------------------------------------------------------------------------------------
	r: array_like
		Halocentric radius in physical kpc/h; can be a number or a numpy array.
	M: float
		Halo mass in :math:`M_{\odot}/h`
	z: float
		Redshift
	mdef: str
		The mass definition in which M is given.
	model: str
		The bias model used.

	Returns
	-----------------------------------------------------------------------------------------------
	rho_2h: array_like
		The density due to the 2-halo term in physical :math:`M_{\odot}h^2/kpc^3`; has the same 
		dimensions as r.
	"""	
	
	cosmo = cosmology.getCurrent()
	bias = haloBias(M, z, mdef, model = model)
	r_comoving_Mpc = r / 1000.0 * (1.0 + z)
	xi = cosmo.correlationFunction(r_comoving_Mpc, z)
	rho_2h = cosmo.rho_m(z) * bias * xi
	
	return rho_2h

###################################################################################################
# SPECIFIC MODELS
###################################################################################################

def modelSC(nu, z):
	"""

	Parameters
	-----------------------------------------------------------------------------------------------
	nu: array_like
		Peak height; can be a number or a numpy array.
	z: array_like
		Redshift; can be a number or a numpy array.
		
	Returns
	-----------------------------------------------------------------------------------------------
	bias: array_like
		Halo bias; has the same dimensions as nu or z.
	"""
	
	delta_c = lss.collapseOverdensity()

	bias = 1.0 + (nu**2 - 1.0) / delta_c
	
	return bias

###################################################################################################

def modelSheth01(nu, z):
	"""
	The halo bias at a given peak height, according to Sheth et al. 2001. 
	
	Parameters
	-----------------------------------------------------------------------------------------------
	nu: array_like
		Peak height; can be a number or a numpy array.
	z: array_like
		Redshift; can be a number or a numpy array.
		
	Returns
	-----------------------------------------------------------------------------------------------
	bias: array_like
		Halo bias; has the same dimensions as nu or z.
	"""
	
	a = 0.707
	b = 0.5
	c = 0.6
	roota = np.sqrt(a)
	anu2 = a * nu**2
	anu2c = anu2**c
	t1 = b * (1.0 - c) * (1.0 - 0.5 * c)
	
	# TODO: is this correct? growth factor multiplied in?
	delta_sc = lss.collapseOverdensity()

	bias = 1.0 +  1.0 / (roota * delta_sc) * (roota * anu2 + roota * b * anu2**(1.0 - c) - anu2c / (anu2c + t1))

	return bias

###################################################################################################

def modelTinker10(nu, z, mdef):
	"""
	The halo bias at a given peak height according to Tinker et al. 2010. 

	The halo bias, using the approximation of Tinker et al. 2010, ApJ 724, 878. The mass definition,
	mdef, must correspond to the mass that was used to evaluate the peak height. Note that the 
	Tinker bias function is universal in redshift at fixed peak height, but only for mass 
	definitions defined wrt the mean density of the universe. For other definitions, :math:`\\Delta_m`
	evolves with redshift, leading to an evolving bias at fixed peak height. 
	
	Parameters
	-----------------------------------------------------------------------------------------------
	nu: array_like
		Peak height; can be a number or a numpy array.
	z: array_like
		Redshift; can be a number or a numpy array.
	mdef: str
		The mass definition
		
	Returns
	-----------------------------------------------------------------------------------------------
	bias: array_like
		Halo bias; has the same dimensions as nu or z.
	"""
	
	cosmo = cosmology.getCurrent()
	Delta = mass_so.densityThreshold(z, mdef) / cosmo.rho_m(z)
	y = np.log10(Delta)

	A = 1.0 + 0.24 * y * np.exp(-1.0 * (4.0 / y)**4)
	a = 0.44 * y - 0.88
	B = 0.183
	b = 1.5
	C = 0.019 + 0.107 * y + 0.19 * np.exp(-1.0 * (4.0 / y)**4)
	c = 2.4
	
	bias = 1.0 - A * nu**a / (nu**a + constants.DELTA_COLLAPSE**a) + B * nu**b + C * nu**c
	
	return bias

###################################################################################################
