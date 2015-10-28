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
    
---------------------------------------------------------------------------------------------------
Module Reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy

from colossus.cosmology import cosmology
from colossus.halo import basics

###################################################################################################
# HALO BIAS
###################################################################################################

def haloBiasFromNu(nu, z, mdef):
	"""
	The halo bias at a given peak height. 

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

	See also
	-----------------------------------------------------------------------------------------------
	haloBias: The halo bias at a given mass. 
	"""
	
	cosmo = cosmology.getCurrent()
	Delta = basics.densityThreshold(z, mdef) / cosmo.rho_m(z)
	y = numpy.log10(Delta)

	A = 1.0 + 0.24 * y * numpy.exp(-1.0 * (4.0 / y)**4)
	a = 0.44 * y - 0.88
	B = 0.183
	b = 1.5
	C = 0.019 + 0.107 * y + 0.19 * numpy.exp(-1.0 * (4.0 / y)**4)
	c = 2.4

	bias = 1.0 - A * nu**a / (nu**a + cosmology.AST_delta_collapse**a) + B * nu**b + C * nu**c

	return bias

###################################################################################################

def haloBias(M, z, mdef):
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
		The mass definition

	Returns
	-----------------------------------------------------------------------------------------------
	bias: array_like
		Halo bias; has the same dimensions as M or z.

	See also
	-----------------------------------------------------------------------------------------------
	haloBiasFromNu: The halo bias at a given peak height. 
	"""
		
	cosmo = cosmology.getCurrent()
	nu = cosmo.peakHeight(M, z)
	b = haloBiasFromNu(nu, z, mdef)
	
	return b

###################################################################################################

def twoHaloTerm(r, M, z, mdef):
	"""
	The 2-halo term as a function of radius and halo mass. 

	The 2-halo term in the halo-matter correlation function describes the excess density around 
	halos due to the proximity of other halos. This contribution can be approximated as the matter-
	matter correlation function times a linear bias which depends on the peak height of the halo.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	r: array_like
		Halocentric radius in physical kpc/h; can be a number or a numpy array.
	M: float
		Halo mass in :math:`M_{\odot}/h`
	z: float
		Redshift
	mdef: str
		The mass definition

	Returns
	-----------------------------------------------------------------------------------------------
	rho_2h: array_like
		The density due to the 2-halo term in physical :math:`M_{\odot}h^2/kpc^3`; has the same 
		dimensions as r.
	"""	
	
	cosmo = cosmology.getCurrent()
	b = haloBias(M, z, mdef)
	r_comoving_Mpc = r / 1000.0 * (1.0 + z)
	xi = cosmo.correlationFunction(r_comoving_Mpc)
	rho_2h = cosmo.rho_m(z) * (1.0 + b * xi)
	
	return rho_2h

###################################################################################################
