###################################################################################################
#
# mass_so.py                (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module implements basic aspects of spherical overdensity mass definitions for dark matter 
halos. The functions in this module are independent of the form of the density profile. For 
functions that rely on a particular form of the halo density profile, please see the 
:doc:`halo_mass_defs` and :doc:`halo_mass_adv` sections.

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

Furthermore, spherical overdensity radii are commonly denoted by R<mdef> (e.g., R200m) and masses
by M<mdef>, e.g. M500c.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np

from colossus.cosmology import cosmology

###################################################################################################
# FUNCTIONS RELATED TO SPHERICAL OVERDENSITY MASSES
###################################################################################################

def parseMassDefinition(mdef):
	"""
	The type and overdensity of a given spherical overdensity mass definition.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	mdef: str
		The mass definition
		
	Returns
	-----------------------------------------------------------------------------------------------
	mdef_type: str
		Can either be based on the mean density (``mdef_type=='m'``), the critical density 
		(``mdef_type=='m'``) or the virial overdensity (``mdef_type=='vir'``).
	mdef_delta: int
		The overdensity; if ``mdef_type=='vir'``, the overdensity depends on redshift, and this
		parameter is None.
	"""
	
	if mdef[-1] == 'c':
		mdef_type = 'c'
		mdef_delta = int(mdef[:-1])

	elif mdef[-1] == 'm':
		mdef_type = 'm'
		mdef_delta = int(mdef[:-1])

	elif mdef == 'vir':
		mdef_type = 'vir'
		mdef_delta = None

	else:
		msg = 'Invalid mass definition, %s.' % mdef
		raise Exception(msg)
	
	return mdef_type, mdef_delta

###################################################################################################

def parseRadiusMassDefinition(rmdef):
	"""
	Parse a radius or mass identifier as well as the mass definition.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	rmdef: str
		The radius or mass identifier
		
	Returns
	-----------------------------------------------------------------------------------------------
	radius_mass: str
		Can be 'R' for radius or 'M' for mass.
	mdef: str
		The mdef the mass or radius are based on.
	mdef_type: str
		Can either be based on the mean density (``mdef_type=='m'``), the critical density 
		(``mdef_type=='m'``) or the virial overdensity (``mdef_type=='vir'``).
	mdef_delta: int
		The overdensity; if ``mdef_type=='vir'``, the overdensity depends on redshift, and this
		parameter is None.
	"""
		
	if rmdef[0] in ['r', 'R']:
		radius_mass = 'R'
	elif rmdef[0] in ['m', 'M']:
		radius_mass = 'M'
	else:
		msg = 'Invalid identifier, %s. Must be either R for radius or M for mass.' % rmdef[0]
		raise Exception(msg)
	
	mdef = rmdef[1:]
	mdef_type, mdef_delta = parseMassDefinition(mdef)
	
	return radius_mass, mdef, mdef_type, mdef_delta

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
	mdef_type, mdef_delta = parseMassDefinition(mdef)
	
	if mdef_type == 'c':
		rho_treshold = mdef_delta * cosmo.rho_c(z)
	elif mdef_type == 'm':
		rho_treshold = mdef_delta * cosmo.rho_m(z)
	elif mdef_type == 'vir':
		rho_treshold = deltaVir(z) * cosmo.rho_c(z)
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
