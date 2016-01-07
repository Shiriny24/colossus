###################################################################################################
#
# mass_adv.py               (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module represents a collection of advanced utilities related to halo mass definitions. 

---------------------------------------------------------------------------------------------------
Changing mass definitions assuming a concentration
---------------------------------------------------------------------------------------------------

The :func:`halo.mass_defs.changeMassDefinition()` function needs to know the concentration of a 
halo. For convenience, the following function uses a concentration model to estimate the 
concentration::

	M200m, R200m, c200m = changeMassDefinitionCModel(1E12, 1.0, 'vir', '200m')
	
By default, the function uses the ``diemer_15`` concentration model (see the documentation of the
:mod:`halo.concentration` module). This function is not included in the :mod:`halo.mass_defs`
module in order to avoid circular dependencies.

---------------------------------------------------------------------------------------------------
Alternative mass definitions
---------------------------------------------------------------------------------------------------

Two alternative mass definitions (as in, not spherical overdensity masses) are implemented in this 
module:

* :math:`M_{sp}`, the splashback mass that is contained within :math:`R_{sp}`, the splashback 
  radius. The radius corresponds to the apocenter of particles on their first orbit after infall,
  and thus physically separates matter that is orbiting in the halo potential and matter that has
  not fallen in yet. Operationally, :math:`R_{sp}` is defined to be the radius where the logarithmic
  slope of the 3D density profile is most negative.
* :math:`M_{<4r_s}`, the mass within 4 scale radii. This mass definition quantifies the mass in
  the inner part of the halo. During the fast accretion regime, this mass definition tracks
  :math:`M_{vir}`, but when the halo stops accreting it approaches a constant. 

:math:`M_{<4r_s}`: can be computed from both NFW and DK14 profiles, while :math:`R_{sp}` and 
:math:`M_{sp}` can only be computed from DK14 profiles. For both mass definitions there are
converter functions:

.. autosummary::	
	RspOverR200m
	MspOverM200m
	Rsp
	Msp
	M4rs

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
from colossus.halo import profile_nfw
from colossus.halo import concentration

###################################################################################################

def changeMassDefinitionCModel(M, z, mdef_in, mdef_out, 
							profile = defaults.HALO_MASS_CONVERSION_PROFILE, 
							c_model = defaults.HALO_CONCENTRATION_MODEL):
	"""
	Change the spherical overdensity mass definition, using a model for the concentration.
	
	This function is a wrapper for the :func:`halo.mass_defs.changeMassDefinition()` function. Instead of forcing 
	the user to provide concentrations, they are computed from a model indicated by the ``c_model``
	parameter.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	M_i: array_like
		The initial halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z_i: float
		The initial redshift.
	mdef_i: str
		The initial mass definition.
	mdef_f: str
		The final mass definition (can be the same as mdef_i, or different).
	profile: str
		The functional form of the profile assumed in the computation; can be ``nfw`` or ``dk14``.
	c_model: str
		The identifier of a concentration model (see :mod:`halo.concentration` for valid inputs).

	Returns
	-----------------------------------------------------------------------------------------------
	Mnew: array_like
		The new halo mass in :math:`M_{\odot}/h`; has the same dimensions as M_i.
	Rnew: array_like
		The new halo radius in physical kpc/h; has the same dimensions as M_i.
	cnew: array_like
		The new concentration (now referring to the new mass definition); has the same dimensions 
		as M_i.
		
	See also
	-----------------------------------------------------------------------------------------------
	halo.mass_defs.pseudoEvolve: Evolve the spherical overdensity radius for a fixed profile.
	halo.mass_defs.changeMassDefinition: Change the spherical overdensity mass definition.
	"""
	
	c = concentration.concentration(M, mdef_in, z, model = c_model)
	
	return mass_defs.pseudoEvolve(M, c, z, mdef_in, z, mdef_out, profile = profile)

###################################################################################################

def M4rs(M, z, mdef, c = None):
	"""
	Convert a spherical overdensity mass to :math:`M_{<4rs}`.
	
	See the section on mass definitions for the definition of :math:`M_{<4rs}`.

	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Spherical overdensity halo mass in :math:`M_{\odot} / h`; can be a number or a numpy
		array.
	z: float
		Redshift
	mdef: str
		The spherical overdensity mass definition in which M (and optionally c) are given.
	c: array_like
		Concentration. If this parameter is not passed, concentration is automatically 
		computed. Must have the same dimensions as M.
		
	Returns
	-----------------------------------------------------------------------------------------------
	M4rs: array_like
		The mass within 4 scale radii, :math:`M_{<4rs}`, in :math:`M_{\odot} / h`; has the 
		same dimensions as M.
	"""

	if c is None:
		c = concentration.concentration(M, mdef, z)
	
	Mfrs = M * profile_nfw.NFWProfile.mu(4.0) / profile_nfw.NFWProfile.mu(c)
	
	return Mfrs

###################################################################################################

def RspOverR200m(nu200m = None, z = None, Gamma = None):
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
		ratio =  0.54 * (1 + 0.53 * cosmo.Om(z)) * (1 + 1.36 * np.exp(-Gamma / 3.04))
	elif nu200m is not None:
		ratio = 0.81 * (1.0 + 0.97 * np.exp(-nu200m / 2.44))
	else:
		msg = 'Need either Gamma and z, or nu.'
		raise Exception(msg)

	return ratio

###################################################################################################

def MspOverM200m(nu200m = None, z = None, Gamma = None):
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
		ratio =  0.59 * (1 + 0.35 * cosmo.Om(z)) * (1 + 0.92 * np.exp(-Gamma / 4.54))
	elif nu200m is not None:
		ratio = 0.82 * (1.0 + 0.63 * np.exp(-nu200m / 3.52))
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
			M200m, R200m, _ = changeMassDefinitionCModel(M, z, mdef, '200m', profile = profile)
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
			M200m, _, _ = changeMassDefinitionCModel(M, z, mdef, '200m', profile = profile)
		else:
			M200m, _, _ = mass_defs.changeMassDefinition(M, c, z, mdef, '200m', profile = profile)
	
	cosmo = cosmology.getCurrent()
	nu200m = cosmo.peakHeight(M200m, z)
	Msp = M200m * MspOverM200m(nu200m = nu200m)
	
	return Msp

###################################################################################################
