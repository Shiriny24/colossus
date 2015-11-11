###################################################################################################
#
# mass_defs.py              (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module implements functions related to halo mass definitions that rely on particular forms of 
the halo density profiles. For the most basic aspects of spherical overdensity mass definitions, 
see the :doc:`halo_mass_so` section.

---------------------------------------------------------------------------------------------------
Basic usage
---------------------------------------------------------------------------------------------------

Some functions make use of density profiles, but are not necessarily tied to a particular 
functional form:

.. autosummary::
	pseudoEvolve
	changeMassDefinition

Pseudo-evolution is the evolution of a spherical overdensity halo radius, mass, and concentration 
due to an evolving reference density (see Diemer, More & Kravtsov 2013 for more information). 
The :func:`pseudoEvolve` function is a very general implementation of this effect. The function 
assumes a profile that is fixed in physical units, and computes how the radius, mass and 
concentration evolve due to changes in mass definition and/or redshift. In the following 
example we compute the pseudo-evolution of a halo with virial mass :math:`M_{vir}=10^{12} M_{\odot}/h` 
from z=1 to z=0::

	M, R, c = pseudoEvolve(1E12, 10.0, 1.0, 'vir', 0.0, 'vir')
	
Here we have assumed that the halo has a concentration :math:`c_{vir} = 10` at z=1. Another 
useful application of this function is to convert one spherical overdensity mass definitions 
to another::

	M200m, R200m, c200m = changeMassDefinition(1E12, 10.0, 1.0, 'vir', '200m')
	
Here we again assumed a halo with :math:`M_{vir}=10^{12} M_{\odot}/h` and :math:`c_{vir} = 10` 
at z=1, and converted it to the 200m mass definition. 

Often, we do not know the concentration of a halo and wish to estimate it using a concentration-
mass model. This function is performed by a convenient wrapper for the 
:func:`changeMassDefinition` function, see :func:`halo.mass_adv.changeMassDefinitionCModel`.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
import inspect

from colossus.utils import utilities
from colossus.halo import mass_so
from colossus.halo import profile_nfw

###################################################################################################
# FUNCTIONS THAT CAN REFER TO DIFFERENT FORMS OF THE DENSITY PROFILE
###################################################################################################

def pseudoEvolve(M_i, c_i, z_i, mdef_i, z_f, mdef_f, profile = 'nfw'):
	"""
	Evolve the spherical overdensity radius for a fixed profile.
	
	This function computes the evolution of spherical overdensity mass and radius due to a changing 
	reference density, an effect called 'pseudo-evolution' (Diemer, et al. 2013 ApJ 766, 25). The 
	user passes the mass and concentration of the density profile, together with a redshift and 
	mass definition to which M and c refer. Given this profile, we evaluate the spherical overdensity
	radius at a different redshift and/or mass definition.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	M_i: array_like
		The initial halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	c_i: float
		The initial halo concentration; must have the same dimensions as M_i.
	z_i: float
		The initial redshift.
	mdef_i: str
		The initial mass definition.
	z_f: float
		The final redshift (can be smaller, equal to, or larger than z_i).
	mdef_f: str
		The final mass definition (can be the same as mdef_i, or different).
	profile: str or HaloDensityProfile
		The functional form of the profile assumed in the computation; can be ``nfw`` or an 
		instance of HaloDensityProfile which must have the parameter rs. In the latter case, only 
		one value of M_i (rather than an array) can be converted.

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
	changeMassDefinition: Change the spherical overdensity mass definition.
	"""
	
	M_i, is_array = utilities.getArray(M_i)
	c_i, _ = utilities.getArray(c_i)
	N = len(M_i)
	Rnew = np.zeros_like(M_i)
	cnew = np.zeros_like(M_i)

	if profile == 'nfw':
		
		# We do not instantiate NFW profile objects, but instead use the faster static functions
		rhos, rs = profile_nfw.NFWProfile.fundamentalParameters(M_i, c_i, z_i, mdef_i)
		density_threshold = mass_so.densityThreshold(z_f, mdef_f)
		for i in range(N):
			cnew[i] = profile_nfw.NFWProfile.xDelta(rhos[i], rs[i], density_threshold, x_guess = c_i[i])
		Rnew = rs * cnew

	elif inspect.isclass(profile):
		
		for i in range(N):
			prof = profile(M = M_i[i], mdef = mdef_i, z = z_i, c = c_i[i])
			Rnew[i] = prof.RDelta(z_f, mdef_f)
			cnew[i] = Rnew[i] / prof.rs

	else:
		msg = 'This function is not defined for profile %s.' % (profile)
		raise Exception(msg)

	if not is_array:
		Rnew = Rnew[0]
		cnew = cnew[0]

	Mnew = mass_so.R_to_M(Rnew, z_f, mdef_f)
	
	return Mnew, Rnew, cnew

###################################################################################################

def changeMassDefinition(M, c, z, mdef_in, mdef_out, profile = 'nfw'):
	"""
	Change the spherical overdensity mass definition.
	
	This function is a special case of the more general pseudoEvolve function; here, the redshift
	is fixed, but we change the mass definition. This leads to a different spherical overdensity
	radius, mass, and concentration.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	M_i: array_like
		The initial halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	c_i: array_like
		The initial halo concentration; must have the same dimensions as M_i.
	z_i: float
		The initial redshift.
	mdef_i: str
		The initial mass definition.
	mdef_f: str
		The final mass definition (can be the same as mdef_i, or different).
	profile: str
		The functional form of the profile assumed in the computation; can be ``nfw`` or ``dk14``.

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
	pseudoEvolve: Evolve the spherical overdensity radius for a fixed profile.
	halo.mass_adv.changeMassDefinitionCModel: Change the spherical overdensity mass definition, using a model for the concentration.
	"""
	
	return pseudoEvolve(M, c, z, mdef_in, z, mdef_out, profile = profile)

###################################################################################################
