###################################################################################################
#
# Halo.py		 		(c) Benedikt Diemer
#						University of Chicago
#     				    bdiemer@oddjob.uchicago.edu
#
###################################################################################################

"""
This module implements basic aspects of dark matter halos, such as spherical overdensity masses
and halo bias. For functions that rely on a particular form of the halo density profile, please 
see the :mod:`HaloDensityProfile` module.

---------------------------------------------------------------------------------------------------
Spherical overdensity masses
---------------------------------------------------------------------------------------------------

Halo masses and radii are most commonly defined using spherical overdensity mass definitions. A
spherical overdensity radius is the radius within which the halo has an overdensity :math:`\Delta`
with respect to some reference density, usually the mean or critical density of the universe. The
following functions compute the overdensity threshold :math:`\Delta` and convert from radius to 
mass and vice versa:

.. autosummary:: 
	densityThreshold
	deltaVir
    M_to_R
    R_to_M

Throughout all Colossus modules that use spherical overdensity masses, the definition is passed
through a parameter called ``mdef`` which is a string and can take on the following values:

========== ========== ==================== ================================================================
Type       mdef       Examples             Explanation
========== ========== ==================== ================================================================
Matter     '<int>m'   178m, 200m           An integer number times the mean matter density of the universe
Critical   '<int>c'   200c, 500c, 2500c    An integer number times the critical density of the universe
Virial     'vir'      vir                  An overdensity that varies with redshift (Bryan & Norman 1998)
========== ========== ==================== ================================================================

---------------------------------------------------------------------------------------------------
Halo bias
---------------------------------------------------------------------------------------------------

.. autosummary:: 
    haloBiasFromNu
    haloBias
    
---------------------------------------------------------------------------------------------------
Detailed Documentation
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import math
import numpy

import Cosmology

###################################################################################################
# FUNCTIONS RELATED TO SPHERICAL OVERDENSITY MASSES
###################################################################################################

def densityThreshold(z, mdef):
	"""
	The threshold density for a given spherical overdensity mass definition.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	z: array_like
		Redshift; can be a number or a numpy array.
	mdef: str
		The mass definition
		
	Returns
	-----------------------------------------------------------------------------------------------
	rho: array_like
		The threshold density in physical :math:`M_{\odot}h^2/kpc^3`; has the same dimensions as z.

	See also
	-----------------------------------------------------------------------------------------------
	deltaVir: The virial overdensity in units of the critical density.
	"""
	
	cosmo = Cosmology.getCurrent()
	rho_crit = Cosmology.AST_rho_crit_0_kpc3 * cosmo.Ez(z)**2

	if mdef[len(mdef) - 1] == 'c':
		delta = int(mdef[:-1])
		rho_treshold = rho_crit * delta

	elif mdef[len(mdef) - 1] == 'm':
		delta = int(mdef[:-1])
		rho_m = Cosmology.AST_rho_crit_0_kpc3 * cosmo.Om0 * (1.0 + z)**3
		rho_treshold = delta * rho_m

	elif mdef == 'vir':
		delta = deltaVir(z)
		rho_treshold = rho_crit * delta

	else:
		msg = 'Invalid mass definition, %s.' % mdef
		raise Exception(msg)

	return rho_treshold

###################################################################################################

def deltaVir(z):
	"""
	The virial overdensity in units of the critical density.
	
	This function uses the fitting formula of Bryan & Norman 1998 to determine the virial 
	overdensity. While the universe is dominated by matter, this overdensity is about 178. Once 
	dark energy starts to matter, it decreases. 
	
	Parameters
	-----------------------------------------------------------------------------------------------
	z: array_like
		Redshift; can be a number or a numpy array.
		
	Returns
	-----------------------------------------------------------------------------------------------
	Delta: array_like
		The virial overdensity; has the same dimensions as z.

	See also
	-----------------------------------------------------------------------------------------------
	densityThreshold: The threshold density for a given mass definition.
	"""
	
	cosmo = Cosmology.getCurrent()
	x = cosmo.Om(z) - 1.0
	Delta = 18 * math.pi**2 + 82.0 * x - 39.0 * x**2

	return Delta

###################################################################################################

def M_to_R(M, z, mdef):
	"""
	Spherical overdensity mass from radius.
	
	This function returns a spherical overdensity halo radius for a halo mass M. Note that this 
	function is independent of the form of the density profile.

	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition
		
	Returns
	-----------------------------------------------------------------------------------------------
	R: array_like
		Halo radius in physical kpc/h; has the same dimensions as M.

	See also
	-----------------------------------------------------------------------------------------------
	R_to_M: Spherical overdensity radius from mass.
	"""
	
	rho = densityThreshold(z, mdef)
	R = (M * 3.0 / 4.0 / math.pi / rho)**(1.0 / 3.0)

	return R

###################################################################################################

def R_to_M(R, z, mdef):
	"""
	Spherical overdensity radius from mass.
	
	This function returns a spherical overdensity halo mass for a halo radius R. Note that this 
	function is independent of the form of the density profile.

	Parameters
	-----------------------------------------------------------------------------------------------
	R: array_like
		Halo radius in physical kpc/h; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition
		
	Returns
	-----------------------------------------------------------------------------------------------
	M: array_like
		Mass in :math:`M_{\odot}/h`; has the same dimensions as R.

	See also
	-----------------------------------------------------------------------------------------------
	M_to_R: Spherical overdensity mass from radius.
	"""
	
	rho = densityThreshold(z, mdef)
	M = 4.0 / 3.0 * math.pi * rho * R**3

	return M

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
	
	cosmo = Cosmology.getCurrent()
	Delta = densityThreshold(z, mdef) / cosmo.matterDensity(z)
	y = numpy.log10(Delta)

	A = 1.0 + 0.24 * y * numpy.exp(-1.0 * (4.0 / y)**4)
	a = 0.44 * y - 0.88
	B = 0.183
	b = 1.5
	C = 0.019 + 0.107 * y + 0.19 * numpy.exp(-1.0 * (4.0 / y)**4)
	c = 2.4

	bias = 1.0 - A * nu**a / (nu**a + Cosmology.AST_delta_collapse**a) + B * nu**b + C * nu**c

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
		
	cosmo = Cosmology.getCurrent()
	nu = cosmo.peakHeight(M, z)
	b = haloBiasFromNu(nu, z, mdef)
	
	return b
