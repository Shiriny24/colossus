###################################################################################################
#
# concentration.py          (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module implements a range of models for halo concentration as a function of mass, redshift, 
and cosmology. The main function in this module, :func:`concentration`, is a wrapper for all 
models::
	
	setCosmology('WMAP9')
	cvir = concentration(1E12, 'vir', 0.0, model = 'diemer15')

Alternatively, the user can also call the individual model functions directly. However, there are
two aspects which the concentration() function handles automatically. 

First, many concentration models are only valid over a certain range of masses, redshifts, and 
cosmologies. If the user requests a mass or redshift outside these ranges, the function returns a 
*soft fail*: the concentration value is computed, but a warning is displayed and/or a False flag
is returned in a boolean array. If a concentration model cannot be computed, this leads to a *hard
fail* and a returned value of INVALID_CONCENTRATION (-1).

Second, each model was only calibrated for one of a few particular mass definitions, such as 
:math:`c_{200c}`, :math:`c_{vir}`, or :math:`c_{200m}`. The concentration() function 
automatically converts these definitions to the definition chosen by the user (see :doc:`halo_mass`
for more information on spherical overdensity masses). For the conversion, we necessarily have to
assume a particular form of the density profile (see the documentation of the 
:func:`halo.mass_defs.changeMassDefinition` function). 

.. warning::
	The conversion to other mass definitions can degrade the accuracy of the predicted 
	concentration.
	
The conversion to other mass definitions can degrade the accuracy by up to ~15-20% for certain 
mass definitions, masses, and redshifts. Using the DK14 profile (see :mod:`halo.profile_dk14`) 
for the mass conversion gives slightly improved results, but the conversion is slower. Please 
see Appendix C in Diemer & Kravtsov 2015 for details.

.. warning::
	The user must ensure that the cosmology is set consistently.
	
Many concentration models were calibrated only for one particular cosmology (though the default
concentration model, 'diemer15', is valid for all masses, redshifts, and cosmologies). Neither the 
concentration() function nor the invidual model functions issue warnings if the set cosmology does
not match the concentration model (with the exception of the modelKlypin15fromM() model). For 
example, it is possible to set a WMAP9 cosmology, and evaluate the Duffy et al. 2008 model which 
is only valid for a WMAP5 cosmology. When using such models, it is the user's responsibility to 
ensure consistency with other calculations.

---------------------------------------------------------------------------------------------------
Concentration models
---------------------------------------------------------------------------------------------------

The following models are supported in this module, and their ID can be passed as the ``model`` 
parameter to the :func:`concentration` function:

============== ================ ================== =========== ========== ============================================================================
ID             Native mdefs     M range (z=0)      z range     Cosmology  Paper
============== ================ ================== =========== ========== ============================================================================
bullock01	   200c             Almost any         Any         Any        `Bullock et al. 2001 <http://adsabs.harvard.edu/abs/2001MNRAS.321..559B>`_
duffy08        200c, vir, 200m  1E11 < M < 1E15    0 < z < 2   WMAP5      `Duffy et al. 2008 <http://adsabs.harvard.edu/abs/2008MNRAS.390L..64D>`_
klypin11       vir              3E10 < M < 5E14    0           WMAP7      `Klypin et al. 2011 <http://adsabs.harvard.edu/abs/2011ApJ...740..102K>`_
prada12        200c             Any                Any         Any        `Prada et al. 2012 <http://adsabs.harvard.edu/abs/2012MNRAS.423.3018P>`_
bhattacharya13 200c, vir, 200m  2E12 < M < 2E15    0 < z < 2   WMAP7      `Bhattacharya et al. 2013 <http://adsabs.harvard.edu/abs/2013ApJ...766...32B>`_
dutton14       200c, vir        M > 1E10           0 < z < 5   Pl1        `Dutton & Maccio 2014 <http://adsabs.harvard.edu/abs/2014MNRAS.441.3359D>`_
diemer15       200c             Any                Any         Any        `Diemer & Kravtsov 2015 <http://adsabs.harvard.edu/abs/2015ApJ...799..108D>`_
klypin16_m     200c, vir        M > 1E10           0 < z < 5   Pl1/WMAP7  `Klypin et al. 2016 <http://adsabs.harvard.edu/abs/2016MNRAS.457.4340K>`_
klypin16_nu    200c, vir        M > 1E10           0 < z < 5   Pl1        `Klypin et al. 2016 <http://adsabs.harvard.edu/abs/2016MNRAS.457.4340K>`_
============== ================ ================== =========== ========== ============================================================================

---------------------------------------------------------------------------------------------------
Module reference
--------------------------------------------------------------------------------------------------- 
"""

###################################################################################################

import numpy as np
import scipy.interpolate
import scipy.optimize
import warnings
from collections import OrderedDict

from colossus.utils import utilities
from colossus.utils import constants
from colossus import defaults
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.halo import mass_so
from colossus.halo import mass_defs

###################################################################################################

class ConcentrationModel():
	"""
	This object contains certain characteristics of a concentration model, most importantly the 
	mass definitions for which concentration can be output (note that the concentration() function
	can automatically convert mass definitions). The listing also contains flags for models that
	are supposed to be universally valid (i.e., at all redshifts, masses, and cosmologies), and 
	models that implement multiple statistics (mean, median etc).
	
	The ``models`` variable is a dictionary of :class:`ConcentrationModel` objects containing all 
	available models.
	"""
	def __init__(self):
		
		self.mdefs = []
		self.func = None
		self.universal = False
		self.depends_on_statistic = False
		
		return

###################################################################################################

models = OrderedDict()

models['bullock01'] = ConcentrationModel()
models['bullock01'].mdefs = ['200c']

models['duffy08'] = ConcentrationModel()
models['duffy08'].mdefs = ['200c', 'vir', '200m']

models['klypin11'] = ConcentrationModel()
models['klypin11'].mdefs = ['vir']

models['prada12'] = ConcentrationModel()
models['prada12'].mdefs = ['200c']
models['prada12'].universal = True

models['bhattacharya13'] = ConcentrationModel()
models['bhattacharya13'].mdefs = ['200c', 'vir', '200m']

models['dutton14'] = ConcentrationModel()
models['dutton14'].mdefs = ['200c', 'vir']

models['diemer15'] = ConcentrationModel()
models['diemer15'].mdefs = ['200c']
models['diemer15'].universal = True
models['diemer15'].depends_on_statistic = True

models['klypin16_m'] = ConcentrationModel()
models['klypin16_m'].mdefs = ['200c', 'vir']

models['klypin16_nu'] = ConcentrationModel()
models['klypin16_nu'].mdefs = ['200c', 'vir']

###################################################################################################

INVALID_CONCENTRATION = -1.0
"""The concentration value returned if the model routine fails to compute."""

###################################################################################################

def concentration(M, mdef, z,
				model = defaults.HALO_CONCENTRATION_MODEL, statistic = defaults.HALO_CONCENTRATION_STATISTIC,
				conversion_profile = defaults.HALO_MASS_CONVERSION_PROFILE, 
				range_return = False, range_warning = True):
	"""
	Concentration as a function of halo mass and redshift.
	
	This function encapsulates all the concentration models implemented in this module. It
	automatically converts between mass definitions if necessary. For some models, a cosmology 
	must be set (see the documentation of the :mod:`cosmology.cosmology` module).
	
	In some cases, the function cannot return concentrations for the masses, redshift, or cosmology
	requested due to limitations on a particular concentration model. If so, the mask return 
	parameter contains a boolean list indicating which elements are valid. It is highly recommended
	that you switch this functionality on by setting ``range_return = True`` if you are not sure
	about the concentration model used.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	mdef: str
		The mass definition in which the halo mass M is given, and in which c is returned. 
	z: float
		Redshift
	model: str
		The model of the c-M relation used; see list above.
	statistic: str
		Some models distinguish between the ``mean`` and ``median`` concentration. Note that most 
		models do not, in which case this parameter is ignored.
	conversion_profile: str
		The profile form used to convert from one mass definition to another. See explanation above.
	range_return: bool
		If ``True``, the function returns a boolean mask indicating the validty of the returned 
		concentrations.
	range_warning: bool
		If ``True``, a warning is thrown if the user requested a mass or redshift where the model is 
		not calibrated. This warning is suppressed if range_return is True, since it is assumed 
		that the user will evaluate the returned mask array to check the validity of the returned
		concentrations.
		
	Returns
	-----------------------------------------------------------------------------------------------
	c: array_like
		Halo concentration(s) in the mass definition mdef; has the same dimensions as M.
	mask: array_like
		If ``range_return == True``, the function returns True/False values, where 
		False indicates that the model was not calibrated at the chosen mass or redshift; has the
		same dimensions as M.
	"""
	
	guess_factors = [2.0, 5.0, 10.0, 100.0, 10000.0]
	n_guess_factors = len(guess_factors)

	# ---------------------------------------------------------------------------------------------
	# Evaluate the concentration model

	def evaluateC(func, M, universal, args):
		if not universal:
			c, mask = func(M, *args)
		else:
			mask = None
			c = func(M, *args)
		return c, mask
	
	# ---------------------------------------------------------------------------------------------
	# This equation is zero for a mass MDelta (in the mass definition of the c-M model) when the
	# corresponding mass in the user's mass definition is M_desired.
	def eq(MDelta, M_desired, mdef_model, func, universal, args):
		
		cDelta, _ = evaluateC(func, MDelta, universal, args)
		if cDelta < 0.0:
			return np.nan
		Mnew, _, _ = mass_defs.changeMassDefinition(MDelta, cDelta, z, mdef_model, mdef)
		
		return Mnew - M_desired

	# ---------------------------------------------------------------------------------------------
	# Distinguish between models
		
	if not model in models.keys():
		msg = 'Unknown model, %s.' % (model)
		raise Exception(msg)
	
	mdefs_model = models[model].mdefs
	universal = models[model].universal
	func = models[model].func
	args = (z,)
	if models[model].depends_on_statistic:
		args = args + (statistic,)
	
	# Now check whether the definition the user has requested is the native definition of the model.
	# If yes, we just return the model concentration. If not, the problem is much harder. Without 
	# knowing the concentration, we do not know what mass in the model definition corresponds to 
	# the input mass M. Thus, we need to   find both M and c iteratively.
	if mdef in mdefs_model:
		
		if len(mdefs_model) > 1:
			args = args + (mdef,)
		c, mask = evaluateC(func, M, universal, args)
		
		# Generate a mask if the model doesn't return one
		if universal and range_return:
			if utilities.isArray(c):
				mask = np.ones((len(c)), dtype = bool)
			else:
				mask = True
			
	else:
		
		# Convert to array
		M_array, is_array = utilities.getArray(M)
		N = len(M_array)
		mask = np.ones((N), dtype = bool)

		mdef_model = mdefs_model[0]
		if len(mdefs_model) > 1:
			args = args + (mdef_model,)

		# To a good approximation, the relation M2 / M1 = Delta1 / Delta2. We use this mass
		# as a guess around which to look for the solution.
		Delta_ratio = mass_so.densityThreshold(z, mdef) / mass_so.densityThreshold(z, mdef_model)
		M_guess = M_array * Delta_ratio
		c = np.zeros_like(M_array)
		
		for i in range(N):
			
			# Iteratively enlarge the search range, if necessary
			args_solver = M_array[i], mdef_model, func, universal, args
			j = 0
			MDelta = None
			while MDelta is None and j < n_guess_factors:
				M_min = M_guess[i] / guess_factors[j]
				M_max = M_guess[i] * guess_factors[j]
				
				# If we catch an exception at this point, or the model function can't compute c,
				# it's not gonna get better by going to more extreme masses, we might as well give
				# up.
				try:
					eq_min = eq(M_min, *args_solver)
					eq_max = eq(M_max, *args_solver)
					if np.isnan(eq_min) or np.isnan(eq_max):
						break						
					if eq_min * eq_max < 0.0:
						MDelta = scipy.optimize.brentq(eq, M_min, M_max, args = args_solver)
					else:
						j += 1
				except Exception:
					break

			if MDelta is None or MDelta < 0.1:
				if range_warning:
					msg = 'Could not find concentration for model %s, mass %.2e, mdef %s.' \
						% (model, M_array[i], mdef)
					warnings.warn(msg)
				c[i] = INVALID_CONCENTRATION
				mask[i] = False
			
			else:
				cDelta, mask_element = evaluateC(func, MDelta, universal, args)
				_, _, c[i] = mass_defs.changeMassDefinition(MDelta, cDelta, z, mdef_model,
										mdef, profile = conversion_profile)
				if not universal:
					mask[i] = mask_element
	
		# If necessary, convert back to scalars
		if not is_array:
			c = c[0]
			mask = mask[0]

	# Spit out warning if the range was violated
	if range_warning and not range_return and not universal:
		mask_array, _ = utilities.getArray(mask)
		if False in mask_array:
			warnings.warn('Some masses or redshifts are outside the validity of the concentration model.')
	
	if range_return:
		return c, mask
	else:
		return c

###################################################################################################
# BULLOCK ET AL 2001 / MACCIO ET AL 2008 MODEL
###################################################################################################

def modelBullock01(M200c, z):
	"""
	The model of Bullock et al. 2001, MNRAS 321, 559, in the improved version of Maccio et al. 2008,
	MNRAS 391, 1940.
	
	This model is universal, but limited by the finite growth factor in a given cosmology which 
	means that the model cannot be evaluated for arbitrarily large masses (halos that will never 
	collapse).
	  
	Parameters
	-----------------------------------------------------------------------------------------------
	M200c: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
		
	Returns
	-----------------------------------------------------------------------------------------------
	c200c: array_like
		Halo concentration; has the same dimensions as M.
	mask: array_like
		Boolean, has the same dimensions as M. Where ``False``, one or more input parameters were
		outside the range where the model was calibrated, and the returned concentration may not 
		be reliable.
	"""
	
	K = 3.85
	F = 0.01

	# Get an inverse interpolator to determine D+ from z. This is an advanced use of the internal
	# table system of the cosmology class.
	cosmo = cosmology.getCurrent()
	interp = cosmo._zInterpolator('growthfactor', cosmo._growthFactorExact, inverse = True, future = True)
	Dmin = interp.get_knots()[0]
	Dmax = interp.get_knots()[-1]

	# The math works out such that we are looking for the redshift where the growth factor is
	# equal to the peak height of a halo with mass F * M.
	M_array, is_array = utilities.getArray(M200c)
	D_target = peaks.peakHeight(F * M_array, 0.0)
	mask = (D_target > Dmin) & (D_target < Dmax)
	N = len(M_array)
	c200c = np.zeros((N), dtype = float)
	H0 = cosmo.Hz(z)
	for i in range(N):
		if mask[i]:
			zc = interp(D_target[i])
			Hc = cosmo.Hz(zc)
			c200c[i] = K * (Hc / H0)**0.6666
		else:
			c200c[i] = INVALID_CONCENTRATION
	
	if not is_array:
		c200c = c200c[0]
		mask = mask[0]
		
	return c200c, mask

###################################################################################################
# DUFFY ET AL 2008 MODEL
###################################################################################################

def modelDuffy08(M, z, mdef):
	"""
	The power-law fits of Duffy et al. 2008, MNRAS 390, L64. This model was calibrated for a WMAP5
	cosmology.
	  
	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition in which the mass is given, and in which concentration is returned.
		Can be ``200c``, ``vir``, or ``200m`` for this function.
		
	Returns
	-----------------------------------------------------------------------------------------------
	c: array_like
		Halo concentration; has the same dimensions as M.
	mask: array_like
		Boolean, has the same dimensions as M. Where ``False``, one or more input parameters were
		outside the range where the model was calibrated, and the returned concentration may not 
		be reliable.
	"""
	
	if mdef == '200c':
		A = 5.71
		B = -0.084
		C = -0.47
	elif mdef == 'vir':
		A = 7.85
		B = -0.081
		C = -0.71
	elif mdef == '200m':
		A = 10.14
		B = -0.081
		C = -1.01
	else:
		msg = 'Invalid mass definition for Duffy et al. 2008 model, %s.' % mdef
		raise Exception(msg)

	c = A * (M / 2E12)**B * (1.0 + z)**C
	mask = (M >= 1E11) & (M <= 1E15) & (z <= 2.0)
	
	return c, mask

###################################################################################################
# KLYPIN ET AL 2011 MODEL
###################################################################################################

def modelKlypin11(Mvir, z):
	"""
	The power-law fit of Klypin et al. 2011, ApJ 740, 102.
	
	This model was calibrated for the WMAP7 cosmology of the Bolshoi simulation. Note that this 
	model relies on concentrations that were measured approximately from circular velocities, rather 
	than from a fit to the actual density profiles. Klypin et al. 2011 also give fits at particular 
	redshifts other than zero. However, there is no clear procedure to interpolate between redshifts, 
	particularly since the z = 0 relation has a different functional form than the high-z 
	relations. Thus, we only implement the z = 0 relation here.
	  
	Parameters
	-----------------------------------------------------------------------------------------------
	Mvir: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
		
	Returns
	-----------------------------------------------------------------------------------------------
	cvir: array_like
		Halo concentration; has the same dimensions as Mvir.
	mask: array_like
		Boolean, has the same dimensions as Mvir. Where ``False``, one or more input parameters were
		outside the range where the model was calibrated, and the returned concentration may not 
		be reliable.
	"""

	cvir = 9.6 * (Mvir / 1E12)**-0.075
	mask = (Mvir > 3E10) & (Mvir < 5E14) & (z < 0.01)

	return cvir, mask

###################################################################################################
# PRADA ET AL 2012 MODEL
###################################################################################################

def modelPrada12(M200c, z):
	"""
	The model of Prada et al. 2012, MNRAS 423, 3018. 
	
	Like the Diemer & Kravtsov 2014 model, this model predicts :math:`c_{200c}` and is based on 
	the :math:`c-\\nu` relation. The model was calibrated on the Bolshoi and Multidark simulations, 
	but is in principle applicable to any cosmology. The implementation follows equations 12 to 22 in 
	Prada et al. 2012. This function uses the exact values for sigma rather than their approximation.

	Parameters
	-----------------------------------------------------------------------------------------------
	M200c: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
		
	Returns
	-----------------------------------------------------------------------------------------------
	c200c: array_like
		Halo concentration; has the same dimensions as M200c.
	"""

	def cmin(x):
		return 3.681 + (5.033 - 3.681) * (1.0 / np.pi * np.arctan(6.948 * (x - 0.424)) + 0.5)
	def smin(x):
		return 1.047 + (1.646 - 1.047) * (1.0 / np.pi * np.arctan(7.386 * (x - 0.526)) + 0.5)

	cosmo = cosmology.getCurrent()
	nu = peaks.peakHeight(M200c, z)

	a = 1.0 / (1.0 + z)
	x = (cosmo.Ode0 / cosmo.Om0) ** (1.0 / 3.0) * a
	B0 = cmin(x) / cmin(1.393)
	B1 = smin(x) / smin(1.393)
	temp_sig = 1.686 / nu
	temp_sigp = temp_sig * B1
	temp_C = 2.881 * ((temp_sigp / 1.257) ** 1.022 + 1) * np.exp(0.06 / temp_sigp ** 2)
	c200c = B0 * temp_C

	return c200c

###################################################################################################
# BHATTACHARYA ET AL 2013 MODEL
###################################################################################################

def modelBhattacharya13(M, z, mdef):
	"""
	The fits of Bhattacharya et al. 2013, ApJ 766, 32. This model was calibrated for a WMAP7 
	cosmology.

	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition in which the mass is given, and in which concentration is returned.
		Can be ``200c``, ``vir``, or ``200m``.
		
	Returns
	-----------------------------------------------------------------------------------------------
	c: array_like
		Halo concentration; has the same dimensions as M.
	mask: array_like
		Boolean, has the same dimensions as M. Where ``False``, one or more input parameters were
		outside the range where the model was calibrated, and the returned concentration may not 
		be reliable.
	"""

	cosmo = cosmology.getCurrent()
	D = cosmo.growthFactor(z)
	
	# Note that peak height in the B13 paper is defined wrt. the mass definition in question, so 
	# we can just use M to evaluate nu. 
	nu = peaks.peakHeight(M, z)

	if mdef == '200c':
		c_fit = 5.9 * D**0.54 * nu**-0.35
	elif mdef == 'vir':
		c_fit = 7.7 * D**0.90 * nu**-0.29
	elif mdef == '200m':
		c_fit = 9.0 * D**1.15 * nu**-0.29
	else:
		msg = 'Invalid mass definition for Bhattacharya et al. 2013 model, %s.' % mdef
		raise Exception(msg)
				
	M_min = 2E12
	M_max = 2E15
	if z > 0.5:
		M_max = 2E14
	if z > 1.5:
		M_max = 1E14
	mask = (M >= M_min) & (M <= M_max) & (z <= 2.0)
	
	return c_fit, mask

###################################################################################################
# DUTTON & MACCIO 2014 MODEL
###################################################################################################

def modelDutton14(M, z, mdef):
	"""
	The power-law fits of Dutton & Maccio 2014, MNRAS 441, 3359. This model was calibrated for the 
	``planck13`` cosmology.

	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition in which the mass is given, and in which concentration is returned.
		Can be ``200c`` or ``vir``.
		
	Returns
	-----------------------------------------------------------------------------------------------
	c: array_like
		Halo concentration; has the same dimensions as M.
	mask: array_like
		Boolean, has the same dimensions as M. Where ``False``, one or more input parameters were
		outside the range where the model was calibrated, and the returned concentration may not 
		be reliable.
	"""

	if mdef == '200c':
		a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
		b = -0.101 + 0.026 * z
	elif mdef == 'vir':
		a = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * z**1.08)
		b = -0.097 + 0.024 * z
	else:
		msg = 'Invalid mass definition for Dutton & Maccio 2014 model, %s.' % mdef
		raise Exception(msg)
	
	logc = a + b * np.log10(M / 1E12)
	c = 10**logc

	mask = (M > 1E10) & (z <= 5.0)

	return c, mask

###################################################################################################
# DIEMER & KRAVTSOV 2015 MODEL
###################################################################################################

DIEMER15_KAPPA = 0.69

DIEMER15_MEDIAN_PHI_0 = 6.58
DIEMER15_MEDIAN_PHI_1 = 1.37
DIEMER15_MEDIAN_ETA_0 = 6.82
DIEMER15_MEDIAN_ETA_1 = 1.42
DIEMER15_MEDIAN_ALPHA = 1.12
DIEMER15_MEDIAN_BETA = 1.69

DIEMER15_MEAN_PHI_0 = 7.14
DIEMER15_MEAN_PHI_1 = 1.60
DIEMER15_MEAN_ETA_0 = 4.10
DIEMER15_MEAN_ETA_1 = 0.75
DIEMER15_MEAN_ALPHA = 1.40
DIEMER15_MEAN_BETA = 0.67

###################################################################################################

def modelDiemer15fromM(M200c, z, statistic = 'median'):
	"""
	The Diemer & Kravtsov 2014 model for concentration, as a function of mass :math:`M_{200c}` and 
	redhsift. A cosmology must be set before executing this function (see the documentation of the 
	cosmology module).

	Parameters
	-----------------------------------------------------------------------------------------------
	M200c: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	statistic: str
		Can be ``mean`` or ``median``.
		
	Returns
	-----------------------------------------------------------------------------------------------
	c200c: array_like
		Halo concentration; has the same dimensions as M200c.
	
	See also
	-----------------------------------------------------------------------------------------------
	modelDiemer15fromNu: The same function, but with peak height as input.
	"""
	
	cosmo = cosmology.getCurrent()
	
	if cosmo.power_law:
		n = cosmo.power_law_n * M200c / M200c
	else:
		n = _diemer15_n_fromM(M200c)
	
	nu = peaks.peakHeight(M200c, z)
	c200c = _diemer15(nu, n, statistic)

	return c200c

###################################################################################################

def modelDiemer15fromNu(nu200c, z, statistic = 'median'):
	"""
	The Diemer & Kravtsov 2014 model for concentration, as a function of peak height 
	:math:`\\nu_{200c}` and redhsift. A cosmology must be set before executing this function (see 
	the documentation of the cosmology module).

	Parameters
	-----------------------------------------------------------------------------------------------
	nu200c: array_like
		Halo peak heights; can be a number or a numpy array. The peak heights must correspond to 
		:math:`M_{200c}` and a top-hat filter.
	z: float
		Redshift
	statistic: str
		Can be ``mean`` or ``median``.
		
	Returns
	-----------------------------------------------------------------------------------------------
	c200c: array_like
		Halo concentration; has the same dimensions as nu200c.
	
	See also
	-----------------------------------------------------------------------------------------------
	modelDiemer15fromM: The same function, but with mass as input.
	"""

	cosmo = cosmology.getCurrent()
	
	if cosmo.power_law:
		n = cosmo.power_law_n * nu200c / nu200c
	else:
		n = _diemer15_n_fromnu(nu200c, z)
	
	ret = _diemer15(nu200c, n, statistic)

	return ret

###################################################################################################

# The universal prediction of the Diemer & Kravtsov 2014 model for a given peak height, power 
# spectrum slope, and statistic.

def _diemer15(nu, n, statistic = 'median'):

	if statistic == 'median':
		floor = DIEMER15_MEDIAN_PHI_0 + n * DIEMER15_MEDIAN_PHI_1
		nu0 = DIEMER15_MEDIAN_ETA_0 + n * DIEMER15_MEDIAN_ETA_1
		alpha = DIEMER15_MEDIAN_ALPHA
		beta = DIEMER15_MEDIAN_BETA
	elif statistic == 'mean':
		floor = DIEMER15_MEAN_PHI_0 + n * DIEMER15_MEAN_PHI_1
		nu0 = DIEMER15_MEAN_ETA_0 + n * DIEMER15_MEAN_ETA_1
		alpha = DIEMER15_MEAN_ALPHA
		beta = DIEMER15_MEAN_BETA
	else:
		raise Exception("Unknown statistic.")
	
	c = 0.5 * floor * ((nu0 / nu)**alpha + (nu / nu0)**beta)
	
	return c

###################################################################################################

# Compute the characteristic wavenumber for a particular halo mass.

def _diemer15_k_R(M):

	cosmo = cosmology.getCurrent()
	rho0 = cosmo.rho_m(0.0)
	R = (3.0 * M / 4.0 / np.pi / rho0)**(1.0 / 3.0) / 1000.0
	k_R = 2.0 * np.pi / R * DIEMER15_KAPPA

	return k_R

###################################################################################################

# Get the slope n = d log(P) / d log(k) at a scale k_R and a redshift z. The slope is computed from
# the Eisenstein & Hu 1998 approximation to the power spectrum (without BAO).

def _diemer15_n(k_R):

	if np.min(k_R) < 0:
		raise Exception("k_R < 0.")

	cosmo = cosmology.getCurrent()
	
	# The way we compute the slope depends on the settings in the cosmology module. If interpolation
	# tables are used, we can compute the slope directly from the spline interpolation which is
	# very fast. If not, we need to compute the slope manually.
	if cosmo.interpolation:
		n = cosmo.matterPowerSpectrum(k_R, model = 'eisenstein98_zb', derivative = True)
		
	else:
		# We need coverage to compute the local slope at kR, which can be an array. Thus, central
		# difference derivatives don't make much sense here, and we use a spline instead.
		k_min = np.min(k_R) * 0.9
		k_max = np.max(k_R) * 1.1
		logk = np.arange(np.log10(k_min), np.log10(k_max), 0.01)
		Pk = cosmo.matterPowerSpectrum(10**logk, model = 'eisenstein98_zb')
		interp = scipy.interpolate.InterpolatedUnivariateSpline(logk, np.log10(Pk))
		n = interp(np.log10(k_R), nu = 1)
	
	return n

###################################################################################################

# Wrapper for the function above which accepts M instead of k.

def _diemer15_n_fromM(M):

	k_R = _diemer15_k_R(M)
	n = _diemer15_n(k_R)
	
	return n

###################################################################################################

# Wrapper for the function above which accepts nu instead of M.

def _diemer15_n_fromnu(nu, z):

	M = peaks.massFromPeakHeight(nu, z)
	n = _diemer15_n_fromM(M)
	
	return n

###################################################################################################
# KLYPIN ET AL 2016 MODELS
###################################################################################################

def modelKlypin16fromM(M, z, mdef):
	"""
	The mass-based fits of Klypin et al. 2016.
	
	Klypin et al. 2016 suggest both peak height-based and mass-based fitting functions for 
	concentration; this function implements the mass-based version. For this version, the 
	fits are only given for the ``planck13`` and ``bolshoi`` cosmologies. Thus, the user must set 
	one of those cosmologies before evaluating this model. The best-fit parameters refer to the 
	mass-selected samples of all halos (as opposed to :math:`v_{max}`-selected samples, or relaxed 
	halos).

	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition in which the mass(es) are given, and in which concentration is returned.
		Can be ``200c`` or ``vir``.
		
	Returns
	-----------------------------------------------------------------------------------------------
	c: array_like
		Halo concentration; has the same dimensions as M.
	mask: array_like
		Boolean, has the same dimensions as M. Where ``False``, one or more input parameters were
		outside the range where the model was calibrated, and the returned concentration may not 
		be reliable.
	
	See also
	-----------------------------------------------------------------------------------------------
	modelKlypin16fromNu: An alternative fitting function suggested in the same paper.
	"""
	if not mdef in ['200c', 'vir']:
		msg = 'Invalid mass definition for Klypin et al 2016 m-based model, %s.' % mdef
		raise Exception(msg)

	cosmo = cosmology.getCurrent()

	if cosmo.name == 'planck13':
		z_bins = [0.0, 0.35, 0.5, 1.0, 1.44, 2.15, 2.5, 2.9, 4.1, 5.4]
		if mdef == '200c':
			C0_bins = [7.4, 6.25, 5.65, 4.3, 3.53, 2.7, 2.42, 2.2, 1.92, 1.65]
			gamma_bins = [0.120, 0.117, 0.115, 0.110, 0.095, 0.085, 0.08, 0.08, 0.08, 0.08]
			M0_bins = [5.5E5, 1E5, 2E4, 900.0, 300.0, 42.0, 17.0, 8.5, 2.0, 0.3]
		elif mdef == 'vir':
			C0_bins = [9.75, 7.25, 6.5, 4.75, 3.8, 3.0, 2.65, 2.42, 2.1, 1.86]
			gamma_bins = [0.110, 0.107, 0.105, 0.1, 0.095, 0.085, 0.08, 0.08, 0.08, 0.08]
			M0_bins = [5E5, 2.2E4, 1E4, 1000.0, 210.0, 43.0, 18.0, 9.0, 1.9, 0.42]
			
	elif cosmo.name == 'bolshoi':
		z_bins = [0.0, 0.5, 1.0, 1.44, 2.15, 2.5, 2.9, 4.1]
		if mdef == '200c':
			C0_bins = [6.6, 5.25, 3.85, 3.0, 2.1, 1.8, 1.6, 1.4]
			gamma_bins = [0.110, 0.105, 0.103, 0.097, 0.095, 0.095, 0.095, 0.095]
			M0_bins = [2E6, 6E4, 800.0, 110.0, 13.0, 6.0, 3.0, 1.0]
		elif mdef == 'vir':
			C0_bins = [9.0, 6.0, 4.3, 3.3, 2.3, 2.1, 1.85, 1.7]
			gamma_bins = [0.1, 0.1, 0.1, 0.1, 0.095, 0.095, 0.095, 0.095]
			M0_bins = [2E6, 7E3, 550.0, 90.0, 11.0, 6.0, 2.5, 1.0]
		
	else:
		msg = 'Invalid cosmology for Klypin et al 2016 m-based model, %s.' % cosmo.name
		raise Exception(msg)

	C0 = np.interp(z, z_bins, C0_bins)
	gamma = np.interp(z, z_bins, gamma_bins)
	M0 = np.interp(z, z_bins, M0_bins)
	M0 *= 1E12

	c = C0 * (M / 1E12)**-gamma * (1.0 + (M / M0)**0.4)
	
	mask = (M > 1E10) & (z <= z_bins[-1])

	return c, mask

###################################################################################################

def modelKlypin16fromNu(M, z, mdef):
	"""
	The peak height-based fits of Klypin et al. 2016.
	
	Klypin et al. 2016 suggest both peak height-based and mass-based fitting functions for 
	concentration; this function implements the peak height-based version. For this version, the 
	fits are only given for the ``planck13`` cosmology. Thus, the user must set this cosmology
	before evaluating this model. The best-fit parameters refer to the mass-selected samples of 
	all halos (as opposed to :math:`v_{max}`-selected samples, or relaxed halos).

	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition in which the mass is given, and in which concentration is returned.
		Can be ``200c`` or ``vir``.
		
	Returns
	-----------------------------------------------------------------------------------------------
	c: array_like
		Halo concentration; has the same dimensions as M.
	mask: array_like
		Boolean, has the same dimensions as M. Where ``False``, one or more input parameters were
		outside the range where the model was calibrated, and the returned concentration may not 
		be reliable.
	
	See also
	-----------------------------------------------------------------------------------------------
	modelKlypin16fromM: An alternative fitting function suggested in the same paper.
	"""

	if mdef == '200c':
		z_bins = [0.0, 0.38, 0.5, 1.0, 1.44, 2.5, 2.89, 5.41]
		a0_bins = [0.4, 0.65, 0.82, 1.08, 1.23, 1.6, 1.68, 1.7]
		b0_bins = [0.278, 0.375, 0.411, 0.436, 0.426, 0.375, 0.360, 0.351]
	elif mdef == 'vir':
		z_bins = [0.0, 0.38, 0.5, 1.0, 1.44, 2.5, 5.5]
		a0_bins = [0.75, 0.9, 0.97, 1.12, 1.28, 1.52, 1.62]
		b0_bins = [0.567, 0.541, 0.529, 0.496, 0.474, 0.421, 0.393]
	else:
		msg = 'Invalid mass definition for Klypin et al 2016 peak height-based model, %s.' % mdef
		raise Exception(msg)

	nu = peaks.peakHeight(M, z)
	sigma = constants.DELTA_COLLAPSE / nu
	a0 = np.interp(z, z_bins, a0_bins)
	b0 = np.interp(z, z_bins, b0_bins)

	sigma_a0 = sigma / a0
	c = b0 * (1.0 + 7.37 * sigma_a0**0.75) * (1.0 + 0.14 * sigma_a0**-2.0)
	
	mask = (M > 1E10) & (z <= z_bins[-1])

	return c, mask

###################################################################################################
# Pointers to model functions
###################################################################################################

models['bullock01'].func = modelBullock01
models['duffy08'].func = modelDuffy08
models['klypin11'].func = modelKlypin11
models['prada12'].func = modelPrada12
models['bhattacharya13'].func = modelBhattacharya13
models['dutton14'].func = modelDutton14
models['diemer15'].func = modelDiemer15fromM
models['klypin16_m'].func = modelKlypin16fromM
models['klypin16_nu'].func = modelKlypin16fromNu

###################################################################################################
