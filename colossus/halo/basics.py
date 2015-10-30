###################################################################################################
#
# basics.py                 (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module implements basic aspects of dark matter halos, such as spherical overdensity masses. 
For functions that rely on a particular form of the halo density profile, please see the 
:mod:`halo.profile` module.

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
Module Reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np

from colossus.utils import constants
from colossus.cosmology import cosmology

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
	
	cosmo = cosmology.getCurrent()
	rho_crit = constants.RHO_CRIT_0_KPC3 * cosmo.Ez(z)**2

	if mdef[-1] == 'c':
		delta = int(mdef[:-1])
		rho_treshold = rho_crit * delta

	elif mdef[-1] == 'm':
		delta = int(mdef[:-1])
		rho_m = constants.RHO_CRIT_0_KPC3 * cosmo.Om0 * (1.0 + z)**3
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
	
	cosmo = cosmology.getCurrent()
	x = cosmo.Om(z) - 1.0
	Delta = 18 * np.pi**2 + 82.0 * x - 39.0 * x**2

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
	R = (M * 3.0 / 4.0 / np.pi / rho)**(1.0 / 3.0)

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
	M = 4.0 / 3.0 * np.pi * rho * R**3

	return M

###################################################################################################
