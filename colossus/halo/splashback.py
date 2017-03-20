###################################################################################################
#
# splashback.py             (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module represents a collection of routines related to the splashback radius.

---------------------------------------------------------------------------------------------------
Background
---------------------------------------------------------------------------------------------------

:math:`M_{sp}`, the splashback mass that is contained within :math:`R_{sp}`, the splashback 
  radius. The radius corresponds to the apocenter of particles on their first orbit after infall,
  and thus physically separates matter that is orbiting in the halo potential and matter that has
  not fallen in yet. Operationally, :math:`R_{sp}` is defined to be the radius where the logarithmic
  slope of the 3D density profile is most negative.

.. autosummary::	
	RspOverR200m
	MspOverM200m
	Rsp
	Msp

For more information, please see Diemer & Kravtsov 2014, Adhikari et al. 2014, and More, Diemer & 
Kravtsov 2015.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np

from colossus import defaults
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.halo import mass_defs
from colossus.halo import mass_adv

###################################################################################################

def RspOverR200m(nu200m = None, z = None, Gamma = None, statistic = 'median'):
	"""
	The ratio :math:`R_{sp} / R_{200m}` from either the accretion rate, :math:`\\Gamma`, or
	the peak height, :math:`\\nu`.
	
	This function implements the relations calibrated in More, Diemer & Kravtsov 2015. Either
	the accretion rate :math:`\\Gamma` and redshift, or the peak height :math:`\\nu`, must not 
	be ``None``. 

	Parameters
	-----------------------------------------------------------------------------------------------
	nu200m: array_like
		The peak height as computed from :math:`M_{200m}`; can be a number or a numpy array.
	z: array_like
		Redshift; can be a number or a numpy array.
	Gamma: array_like
		The mass accretion rate, as defined in Diemer & Kravtsov 2014; can be a number or a 
		numpy array.
	statistic: str
		Can be ``mean`` or ``median``, determining whether the function returns the best fit to the 
		mean or median profile of a halo sample.
	
	Returns
	-----------------------------------------------------------------------------------------------
	ratio: array_like
		:math:`R_{sp} / R_{200m}`; has the same dimensions as z, Gamma, or nu, depending
		on which of those parameters is an array.
		
	See also
	-----------------------------------------------------------------------------------------------
	MspOverM200m: The ratio :math:`M_{sp} / M_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
	Rsp: :math:`R_{sp}` as a function of spherical overdensity radius.
	Msp: :math:`M_{sp}` as a function of spherical overdensity mass.
	"""

	if (Gamma is not None) and (z is not None):
		cosmo = cosmology.getCurrent()
		if statistic == 'median':
			ratio = 0.54 * (1.0 + 0.53 * cosmo.Om(z)) * (1 + 1.36 * np.exp(-Gamma / 3.04))
		elif statistic == 'mean':
			ratio = 0.58 * (1.0 + 0.63 * cosmo.Om(z)) * (1 + 1.08 * np.exp(-Gamma / 2.26))
		else:
			msg = 'Unknown statistic, %s.' % statistic
			raise Exception(msg)
	elif nu200m is not None:
		if statistic == 'median':
			ratio = 0.81 * (1.0 + 0.97 * np.exp(-nu200m / 2.44))
		elif statistic == 'mean':
			ratio = 0.88 * (1.0 + 0.77 * np.exp(-nu200m / 1.95))
		else:
			msg = 'Unknown statistic, %s.' % statistic
			raise Exception(msg)
	else:
		msg = 'Need either Gamma and z, or nu.'
		raise Exception(msg)

	return ratio

###################################################################################################

def MspOverM200m(nu200m = None, z = None, Gamma = None, statistic = 'median'):
	"""
	The ratio :math:`M_{sp} / M_{200m}` from either the accretion rate, :math:`\\Gamma`, or
	the peak height, :math:`\\nu`.
	
	This function implements the relations calibrated in More, Diemer & Kravtsov 2015. Either
	the accretion rate :math:`\\Gamma` and redshift, or the peak height :math:`\\nu`, must not 
	be ``None``. 

	Parameters
	-----------------------------------------------------------------------------------------------
	nu_vir: array_like
		The peak height as computed from :math:`M_{200m}`; can be a number or a numpy array.
	z: array_like
		Redshift; can be a number or a numpy array.
	Gamma: array_like
		The mass accretion rate, as defined in Diemer & Kravtsov 2014; can be a number or a 
		numpy array.
	statistic: str
		Can be ``mean`` or ``median``, determining whether the function returns the best fit to the 
		mean or median profile of a halo sample.
	
	Returns
	-----------------------------------------------------------------------------------------------
	ratio: array_like
		:math:`M_{sp} / M_{200m}`; has the same dimensions as z, Gamma, or nu, depending
		on which of those parameters is an array.
		
	See also
	-----------------------------------------------------------------------------------------------
	RspOverR200m: The ratio :math:`R_{sp} / R_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
	Rsp: :math:`R_{sp}` as a function of spherical overdensity radius.
	Msp: :math:`M_{sp}` as a function of spherical overdensity mass.
	"""
	
	if (Gamma is not None) and (z is not None):
		cosmo = cosmology.getCurrent()
		if statistic == 'median':
			ratio = 0.59 * (1.0 + 0.35 * cosmo.Om(z)) * (1 + 0.92 * np.exp(-Gamma / 4.54))
		elif statistic == 'mean':
			ratio = 0.70 * (1.0 + 0.37 * cosmo.Om(z)) * (1 + 0.62 * np.exp(-Gamma / 2.69))
		else:
			msg = 'Unknown statistic, %s.' % statistic
			raise Exception(msg)
	elif nu200m is not None:
		if statistic == 'median':
			ratio = 0.82 * (1.0 + 0.63 * np.exp(-nu200m / 3.52))
		elif statistic == 'mean':
			ratio = 0.92 * (1.0 + 0.45 * np.exp(-nu200m / 2.26))
		else:
			msg = 'Unknown statistic, %s.' % statistic
			raise Exception(msg)
	else:
		msg = 'Need either Gamma and z, or nu.'
		raise Exception(msg)
	
	return ratio

###################################################################################################

def Rsp(R, z, mdef, c = None, profile = defaults.HALO_MASS_CONVERSION_PROFILE):
	"""
	:math:`R_{sp}` as a function of spherical overdensity radius.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	R: array_like
		Spherical overdensity radius in physical :math:`kpc/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		Mass definition in which R and c are given.
	c: array_like
		Halo concentration; must have the same dimensions as R, or be ``None`` in which case the 
		concentration is computed automatically.
	profile: str
		The functional form of the profile assumed in the conversion between mass definitions; 
		can be ``nfw`` or ``dk14``.

	Returns
	-----------------------------------------------------------------------------------------------
	Rsp: array_like
		:math:`R_{sp}` in physical :math:`kpc/h`; has the same dimensions as R.
		
	See also
	-----------------------------------------------------------------------------------------------
	RspOverR200m: The ratio :math:`R_{sp} / R_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
	MspOverM200m: The ratio :math:`M_{sp} / M_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
	Msp: :math:`M_{sp}` as a function of spherical overdensity mass.
	"""
	
	if mdef == '200m':
		R200m = R
		M200m = mass_so.R_to_M(R200m, z, '200m')
	else:
		M = mass_so.R_to_M(R, z, mdef)
		if c is None:
			M200m, R200m, _ = mass_adv.changeMassDefinitionCModel(M, z, mdef, '200m', profile = profile)
		else:
			M200m, R200m, _ = mass_defs.changeMassDefinition(M, c, z, mdef, '200m', profile = profile)
			
	cosmo = cosmology.getCurrent()
	nu200m = cosmo.peakHeight(M200m, z)
	Rsp = R200m * RspOverR200m(nu200m = nu200m)
	
	return Rsp

###################################################################################################

def Msp(M, z, mdef, c = None, profile = defaults.HALO_MASS_CONVERSION_PROFILE):
	"""
	:math:`M_{sp}` as a function of spherical overdensity mass.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Spherical overdensity mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		Mass definition in which M and c are given.
	c: array_like
		Halo concentration; must have the same dimensions as M, or be ``None`` in which case the 
		concentration is computed automatically.
	profile: str
		The functional form of the profile assumed in the conversion between mass definitions; 
		can be ``nfw`` or ``dk14``.

	Returns
	-----------------------------------------------------------------------------------------------
	Msp: array_like
		:math:`M_{sp}` in :math:`M_{\odot}/h`; has the same dimensions as M.
		
	See also
	-----------------------------------------------------------------------------------------------
	RspOverR200m: The ratio :math:`R_{sp} / R_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
	MspOverM200m: The ratio :math:`M_{sp} / M_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
	Rsp: :math:`R_{sp}` as a function of spherical overdensity radius.
	"""
	
	if mdef == '200m':
		M200m = M
	else:
		if c is None:
			M200m, _, _ = mass_adv.changeMassDefinitionCModel(M, z, mdef, '200m', profile = profile)
		else:
			M200m, _, _ = mass_defs.changeMassDefinition(M, c, z, mdef, '200m', profile = profile)
	
	cosmo = cosmology.getCurrent()
	nu200m = cosmo.peakHeight(M200m, z)
	Msp = M200m * MspOverM200m(nu200m = nu200m)
	
	return Msp

###################################################################################################
