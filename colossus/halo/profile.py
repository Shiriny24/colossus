###################################################################################################
#
# profile.py                (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module implements radial density profiles of dark matter halos, and functions that rely on 
them. It contains a generic base class for halo density profiles, as well as derived classes
for particular functional forms of the profile.

---------------------------------------------------------------------------------------------------
Basic usage
---------------------------------------------------------------------------------------------------

Some functions make use of density profiles, but are not necessarily tied to a particular 
functional form:

.. autosummary::
	pseudoEvolve
	changeMassDefinition
	radiusFromPdf

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
:func:`changeMassDefinition` function, see :func:`halo.mass_definitions.changeMassDefinitionCModel`.

---------------------------------------------------------------------------------------------------
Module Reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
import scipy.optimize
import scipy.interpolate
import inspect

from colossus.utils import utilities
from colossus.halo import basics
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
		density_threshold = basics.densityThreshold(z_f, mdef_f)
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

	Mnew = basics.R_to_M(Rnew, z_f, mdef_f)
	
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
	halo.mass_definitions.changeMassDefinitionCModel: Change the spherical overdensity mass definition, using a model for the concentration.
	"""
	
	return pseudoEvolve(M, c, z, mdef_in, z, mdef_out, profile = profile)

###################################################################################################

def radiusFromPdf(M, c, z, mdef, cumulativePdf,
					interpolate = True, min_interpolate_pdf = 0.01):
	"""
	Get the radius where the cumulative density distribution of a halo has a certain value, 
	assuming an NFW profile. 
	
	This function can be useful when assigning radii to satellite galaxies in mock halos, for 
	example. The function is optimized for speed when M is a large array. The density 
	distribution is cut off at the virial radius corresponding to the given mass 
	definition. For example, if ``mdef == vir``, the NFW profile is cut off at :math:`R_{vir}`. 
	The accuracy achieved is about 0.2%, unless min_interpolate_pdf is changed to a lower value; 
	below 0.01, the accuracy of the interpolation drops.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Halo mass in units of :math:`M_{\odot}/h`; can be a number or a numpy array.
	c: array_like
		Halo concentration, in the same definition as M; must have the same dimensions as M.
	z: float
		Redshift
	mdef: str
		The mass definition in which the halo mass M is given. 
	cumulativePdf: array_like
		The cumulative pdf that we are seeking. If an array, this array needs to have the same 
		dimensions as the M array.
	c_model: str
		The model used to evaluate concentration if ``c is None``.
	interpolate: bool
		If ``interpolate == True``, an interpolation table is built before computing the radii. This 
		is much faster if M is a large array. 
	min_interpolate_pdf: float
		For values of the cumulativePdf that fall below this value, the radius is computed exactly,
		even if ``interpolation == True``. The reason is that the interpolation becomes unreliable
		for these very low pdfs. 
		
	Returns
	-----------------------------------------------------------------------------------------------
	r: array_like
		The radii where the cumulative pdf(s) is/are achieved, in units of physical kpc/h; has the 
		same dimensions as M.

	Warnings
	-----------------------------------------------------------------------------------------------
		If many pdf values fall below ``min_interpolate_pdf``, this will slow the function
		down significantly.
	"""

	def equ(c, target):
		return profile_nfw.NFWProfile.mu(c) - target
	
	def getX(c, p):
		
		target = profile_nfw.NFWProfile.mu(c) * p
		x = scipy.optimize.brentq(equ, 0.0, c, args = target)
		
		return x
	
	M_array, is_array = utilities.getArray(M)
	R = basics.M_to_R(M, z, mdef)
	N = len(M_array)
	x = 0.0 * M_array
	c_array, _ = utilities.getArray(c)
	p_array, _ = utilities.getArray(cumulativePdf)
	
	if interpolate:

		# Create an interpolator on a regular grid in c-p space.
		bin_width_c = 0.1
		c_min = np.min(c_array) * 0.99
		c_max = np.max(c_array) * 1.01
		c_bins = np.arange(c_min, c_max + bin_width_c, bin_width_c)
		
		p_bins0 = np.arange(0.0, 0.01, 0.001)
		p_bins1 = np.arange(0.01, 0.1, 0.01)
		p_bins2 = np.arange(0.1, 1.1, 0.1)
		p_bins = np.concatenate((p_bins0, p_bins1, p_bins2))
		
		N_c = len(c_bins)
		N_p = len(p_bins)

		x_ = np.zeros((N_c, N_p), dtype = float)
		for i in range(N_c):			
			for j in range(N_p):
				p = p_bins[j]
				target = profile_nfw.NFWProfile.mu(c_bins[i]) * p
				x_[i, j] = scipy.optimize.brentq(equ, 0.0, c_bins[i], args = target) / c_bins[i]
		
		spl = scipy.interpolate.RectBivariateSpline(c_bins, p_bins, x_)

		# For very small values, overwrite the interpolated values with the exact value.
		for i in range(N):
			if p_array[i] < min_interpolate_pdf:
				x[i] = getX(c_array[i], cumulativePdf[i]) / c_array[i]
			else:
				x[i] = spl(c_array[i], p_array[i])

		r = R * x
	
	else:

		# A simple root-finding algorithm. 
		for i in range(N):
			x[i] = getX(c_array[i], cumulativePdf[i])
		r = R / c_array * x
			
	if not is_array:
		r = r[0]
	
	return r

###################################################################################################
