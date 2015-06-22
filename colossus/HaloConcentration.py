###################################################################################################
#
# HaloConcentration.py 	(c) Benedikt Diemer
#						University of Chicago
#     				    bdiemer@oddjob.uchicago.edu
#
###################################################################################################

"""
This module implements a range of models for halo concentration as a function of mass, redshift, 
and cosmology. 

---------------------------------------------------------------------------------------------------
Basic usage
---------------------------------------------------------------------------------------------------

The main function in this module, :func:`concentration`, is a wrapper for all models::
	
	setCosmology('WMAP9')
	cvir = concentration(1E12, 'vir', 0.0, model = 'diemer15')

Alternatively, the user can also call the individual model functions directly. Note, however, that 
most models are only valid over a certain range of masses, redshifts, and cosmologies.

Furthermore, each model was only calibrated for one of a few particular mass definitions, such as 
:math:`c_{200c}`, :math:`c_{vir}`, or :math:`c_{200m}`. The :func:`concentration` function 
automatically converts these definitions to the definition chosen by the user. For documentation 
on spherical overdensity mass definitions, please see the documentation of the :mod:`Halo` module.

***************************************************************************************************
Concentration models
***************************************************************************************************

The following models are supported in this module, and their ID can be passed as the ``model`` 
parameter to the :func:`concentration` function. Alternatively, the user 
can call the models directly using the <ID>_c() functions documented below. Please note that those 
functions do not convert mass definitions, check on the validity of the results etc.

============== ================ ================== =========== =============== ========================== =================
ID             Native mdefs     M range (z=0)      z range     Cosmology       Paper                      Reference
============== ================ ================== =========== =============== ========================== =================
diemer15       200c             Any                Any         Any             Diemer & Kravtsov 2015     ApJ 799, 108
klypin15_nu    200c, vir        M > 1E10           0 < z < 5   Planck1         Klypin et al. 2014
klypin15_m     200c, vir        M > 1E10           0 < z < 5   Planck1/WMAP7   Klypin et al. 2014
dutton14       200c, vir        M > 1E10           0 < z < 5   Planck1         Dutton & Maccio 2014       MNRAS 441, 3359
bhattacharya13 200c, vir, 200m  2E12 < M < 2E15    0 < z < 2   WMAP7           Bhattacharya et al. 2013   ApJ 766, 32
prada12        200c             Any                Any         Any             Prada et al. 2012          MNRAS 423, 3018
klypin11       vir              3E10 < M < 5E14    0           WMAP7           Klypin et al. 2011         ApJ 740, 102
duffy08        200c, vir, 200m  1E11 < M < 1E15    0 < z < 2   WMAP5           Duffy et al. 2008          MNRAS 390, L64
bullock01	   200c             Almost any         Any         Any             Bullock et al. 2001        MNRAS 321, 559
============== ================ ================== =========== =============== ========================== =================

***************************************************************************************************
Conversion between mass definitions
***************************************************************************************************
	
If the user requests a mass definition that is not one of the native definitions of the c-M model,
the mass and concentration are converted, necessarily assuming a particular form of the density
profile. For this purpose, the user can choose between ``nfw`` and ``dk14`` profiles.
	
.. warning:: The conversion to other mass definitions can degrade the accuracy of the predicted 
	concentration. Diemer & Kravtsov 2014 we have evaluated this added inaccuracy,
	and found that it can degrade the prediction by up to ~15-20% for certain mass definitions, 
	masses, and redshifts. The density profile of Diemer & Kravtsov 2014a gives slightly improved
	results, but the conversion is slower. Please see Appendix C in Diemer & Kravtsov 2014 for
	details.
	              
---------------------------------------------------------------------------------------------------
Performance optimization
---------------------------------------------------------------------------------------------------

Some models, including the diemer15 model, use certain cosmological quantities, such as 
:math:`\sigma(R)`, that can be computationally intensive. If you wish to compute concentration for 
many different cosmologies (for example, in an MCMC chain), please consult the documentation of the 
``interpolation`` switch in the Cosmology module.

---------------------------------------------------------------------------------------------------
Detailed Documentation
--------------------------------------------------------------------------------------------------- 
"""

###################################################################################################

import math
import numpy
import scipy.interpolate
import scipy.optimize
import warnings

import Utilities
import Cosmology
import Halo
import HaloDensityProfile

###################################################################################################

def concentration(M, mdef, z, \
				model = 'diemer15', statistic = 'median', conversion_profile = 'nfw', \
				range_return = False, range_warning = True):
	"""
	Concentration as a function of halo mass and redshift, for different concentration models, 
	statistics, and conversion profiles. For some models, a cosmology must be set (see the 
	documentation of the Cosmology module).
	
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
		not calibrated.
		
	Returns
	-----------------------------------------------------------------------------------------------
	c: array_like
		Halo concentration(s) in the mass definition mdef; has the same dimensions as M.
	mask: array_like
		If ``range_return == True``, the function returns True/False values, where 
		False indicates that the model was not calibrated at the chosen mass or redshift; has the
		same dimensions as M.
		
	Warnings
	-----------------------------------------------------------------------------------------------
	Many concentration models were calibrated on a particular cosmology. Note that this function 
	does not always issue a warning if the user has not set the same cosmology! For example, it is
	possible to set a WMAP9 cosmology, and evaluate the Duffy et al. 2008 model which is only valid
	for a WMAP5 cosmology. When using such models, it is the user's responsibility to ensure 
	consistency with other calculations.
	"""
	
	guess_factors = [5.0, 10.0, 100.0, 10000.0]
	n_guess_factors = len(guess_factors)

	# Evaluate the concentration model
	def evaluateC(func, M, limited, args):
		if limited:
			c, mask = func(M, *args)
		else:
			mask = None
			c = func(M, *args)
		return c, mask
	
	# This equation is zero for a mass MDelta (in the mass definition of the c-M model) when the
	# corresponding mass in the user's mass definition is M_desired.
	def eq(MDelta, M_desired, mdef_model, func, limited, args):
		cDelta, _ = evaluateC(func, MDelta, limited, args)
		Mnew, _, _ = HaloDensityProfile.changeMassDefinition(MDelta, cDelta, z, mdef_model, mdef,\
												profile = 'nfw')
		return Mnew - M_desired

	# Distinguish between models
	if model == 'diemer15':
		mdefs_model = ['200c']
		func = diemer15_c200c_M
		args = (z, statistic)
		limited = False
		
	elif model == 'klypin15_nu':
		mdefs_model = ['200c', 'vir']
		func = klypin15_nu_c
		args = (z,)
		limited = True

	elif model == 'klypin15_m':
		mdefs_model = ['200c', 'vir']
		func = klypin15_m_c
		args = (z,)
		limited = True

	elif model == 'dutton14':
		mdefs_model = ['200c', 'vir']
		func = dutton14_c
		args = (z,)
		limited = True

	elif model == 'bhattacharya13':
		mdefs_model = ['200c', 'vir', '200m']
		func = bhattacharya13_c
		args = (z,)
		limited = True

	elif model == 'prada12':
		mdefs_model = ['200c']
		func = prada12_c200c
		args = (z,)
		limited = False

	elif model == 'klypin11':
		mdefs_model = ['vir']
		func = klypin11_cvir
		args = (z,)
		limited = True
		
	elif model == 'duffy08':
		mdefs_model = ['200c', 'vir', '200m']
		func = duffy08_c
		args = (z,)
		limited = True
	
	elif model == 'bullock01':
		mdefs_model = ['200c']
		func = bullock01_c200c
		args = (z,)
		limited = True
	
	else:
		msg = 'Unknown model, %s.' % (model)
		raise Exception(msg)
	
	# Now check whether the definition the user has requested is the native definition of the model.
	# If yes, we just return the model concentration. If not, the problem is much harder. Without 
	# knowing the concentration, we do not know what mass in the model definition corresponds to 
	# the input mass M. Thus, we need to find both M and c iteratively.
	if mdef in mdefs_model:
		
		if len(mdefs_model) > 1:
			args = args + (mdef,)
		c, mask = evaluateC(func, M, limited, args)
		
		# Generate a mask if the model doesn't return one
		if not limited and range_return:
			if Utilities.isArray(c):
				mask = numpy.ones((len(c)), dtype = bool)
			else:
				mask = True
			
	else:
		
		# Convert to array
		M_array, is_array = Utilities.getArray(M)
		N = len(M_array)
		mask = numpy.ones((N), dtype = bool)

		mdef_model = mdefs_model[0]
		if len(mdefs_model) > 1:
			args = args + (mdef_model,)

		# To a good approximation, the relation M2 / M1 = Delta1 / Delta2. We use this mass
		# as a guess around which to look for the solution.
		Delta_ratio = Halo.densityThreshold(z, mdef) / Halo.densityThreshold(z, mdef_model)
		M_guess = M_array * Delta_ratio
		c = numpy.zeros_like(M_array)
		
		for i in range(N):
			
			# Iteratively enlarge the search range, if necessary
			args_solver = M_array[i], mdef_model, func, limited, args
			j = 0
			MDelta = None
			while MDelta is None and j < n_guess_factors:
				try:
					M_min = M_guess[i] / guess_factors[j]
					M_max = M_guess[i] * guess_factors[j]
					MDelta = scipy.optimize.brentq(eq, M_min, M_max, args = args_solver)
				except Exception:
					j += 1

			if MDelta is None:
				msg = 'Could not find concentration for mass %.2e, mdef %s. The mask array indicates invalid concentrations.' % (MDelta, mdef)
				warnings.warn(msg)
				c[i] = 0.0
				mask[i] = False
				
			cDelta, mask_element = evaluateC(func, MDelta, limited, args)
			_, _, c[i] = HaloDensityProfile.changeMassDefinition(MDelta, cDelta, z, mdef_model, \
									mdef, profile = conversion_profile)
			if limited:
				mask[i] = mask_element
	
		# If necessary, convert back to scalars
		if not is_array:
			c = c[0]
			mask = mask[0]

	# Spit out warning if the range was violated
	if range_warning and not range_return and limited:
		mask_array, _ = Utilities.getArray(mask)
		if False in mask_array:
			warnings.warn('Some masses or redshifts are outside the validity of the concentration model.')
	
	if range_return:
		return c, mask
	else:
		return c

###################################################################################################
# DIEMER & KRAVTSOV 2014 MODEL
###################################################################################################

diemer15_kappa = 0.69

diemer15_median_phi_0 = 6.58
diemer15_median_phi_1 = 1.37
diemer15_median_eta_0 = 6.82
diemer15_median_eta_1 = 1.42
diemer15_median_alpha = 1.12
diemer15_median_beta = 1.69

diemer15_mean_phi_0 = 7.14
diemer15_mean_phi_1 = 1.60
diemer15_mean_eta_0 = 4.10
diemer15_mean_eta_1 = 0.75
diemer15_mean_alpha = 1.40
diemer15_mean_beta = 0.67

###################################################################################################

def diemer15_c200c_M(M200c, z, statistic = 'median'):
	"""
	The Diemer & Kravtsov 2014 model for concentration, as a function of mass :math:`M_{200c}` and 
	redhsift. A cosmology must be set before executing this function (see the documentation of the 
	Cosmology module).

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
	diemer15_c200c_nu: The same function, but with peak height as input.
	"""
	
	cosmo = Cosmology.getCurrent()
	
	if cosmo.power_law:
		n = cosmo.power_law_n * M200c / M200c
	else:
		n = diemer15_compute_n_M(M200c)
	
	nu = cosmo.peakHeight(M200c, z)
	c200c = diemer15_c200c_n(nu, n, statistic)

	return c200c

###################################################################################################

def diemer15_c200c_nu(nu200c, z, statistic = 'median'):
	"""
	The Diemer & Kravtsov 2014 model for concentration, as a function of peak height 
	:math:`\\nu_{200c}` and redhsift. A cosmology must be set before executing this function (see 
	the documentation of the Cosmology module).

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
	diemer15_c200c_M: The same function, but with mass as input.
	"""

	cosmo = Cosmology.getCurrent()
	
	if cosmo.power_law:
		n = cosmo.power_law_n * nu200c / nu200c
	else:
		n = diemer15_compute_n_nu(nu200c, z)
	
	ret = diemer15_c200c_n(nu200c, n, statistic)

	return ret

###################################################################################################

# The universal prediction of the Diemer & Kravtsov 2014 model for a given peak height, power 
# spectrum slope, and statistic.

def diemer15_c200c_n(nu, n, statistic = 'median'):

	if statistic == 'median':
		floor = diemer15_median_phi_0 + n * diemer15_median_phi_1
		nu0 = diemer15_median_eta_0 + n * diemer15_median_eta_1
		alpha = diemer15_median_alpha
		beta = diemer15_median_beta
	elif statistic == 'mean':
		floor = diemer15_mean_phi_0 + n * diemer15_mean_phi_1
		nu0 = diemer15_mean_eta_0 + n * diemer15_mean_eta_1
		alpha = diemer15_mean_alpha
		beta = diemer15_mean_beta
	else:
		raise Exception("Unknown statistic.")
	
	c = 0.5 * floor * ((nu0 / nu)**alpha + (nu / nu0)**beta)
	
	return c

###################################################################################################

# Compute the characteristic wavenumber for a particular halo mass.

def diemer15_wavenumber_k_R(M):

	cosmo = Cosmology.getCurrent()
	rho0 = cosmo.rho_m(0.0)
	R = (3.0 * M / 4.0 / math.pi / rho0) ** (1.0 / 3.0) / 1000.0
	k_R = 2.0 * math.pi / R * diemer15_kappa

	return k_R

###################################################################################################

# Get the slope n = d log(P) / d log(k) at a scale k_R and a redshift z. The slope is computed from
# the Eisenstein & Hu 1998 approximation to the power spectrum (without BAO).

def diemer15_compute_n(k_R):

	if numpy.min(k_R) < 0:
		raise Exception("k_R < 0.")

	cosmo = Cosmology.getCurrent()
	
	# The way we compute the slope depends on the settings in the Cosmology module. If interpolation
	# tables are used, we can compute the slope directly from the spline interpolation which is
	# very fast. If not, we need to compute the slope manually.
	if cosmo.interpolation:
		n = cosmo.matterPowerSpectrum(k_R, Pk_source = 'eh98smooth', derivative = True)
		
	else:
		# We need coverage to compute the local slope at kR, which can be an array. Thus, central
		# difference derivatives don't make much sense here, and we use a spline instead.
		k_min = numpy.min(k_R) * 0.9
		k_max = numpy.max(k_R) * 1.1
		logk = numpy.arange(numpy.log10(k_min), numpy.log10(k_max), 0.01)
		Pk = cosmo.matterPowerSpectrum(10**logk, Pk_source = 'eh98smooth')
		interp = scipy.interpolate.InterpolatedUnivariateSpline(logk, numpy.log10(Pk))
		n = interp(numpy.log10(k_R), nu = 1)
	
	return n

###################################################################################################

# Wrapper for the function above which accepts M instead of k.

def diemer15_compute_n_M(M):

	k_R = diemer15_wavenumber_k_R(M)
	n = diemer15_compute_n(k_R)
	
	return n

###################################################################################################

# Wrapper for the function above which accepts nu instead of M.

def diemer15_compute_n_nu(nu, z):

	cosmo = Cosmology.getCurrent()
	M = cosmo.massFromPeakHeight(nu, z)
	n = diemer15_compute_n_M(M)
	
	return n

###################################################################################################
# KLYPIN ET AL 2015 MODELS
###################################################################################################

def klypin15_nu_c(M, z, mdef):
	"""
	The peak height-based fits of Klypin et al. 2015.
	
	Klypin et al. 2015 suggest both peak height-based and mass-based fitting functions for 
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
	klypin15_m_c: An alternative fitting function suggested in the same paper.
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
		msg = 'Invalid mass definition for Klypin et al 2015 peak height-based model, %s.' % mdef
		raise Exception(msg)

	cosmo = Cosmology.getCurrent()
	nu = cosmo.peakHeight(M, z)
	sigma = Cosmology.AST_delta_collapse / nu
	a0 = numpy.interp(z, z_bins, a0_bins)
	b0 = numpy.interp(z, z_bins, b0_bins)

	sigma_a0 = sigma / a0
	c = b0 * (1.0 + 7.37 * sigma_a0**0.75) * (1.0 + 0.14 * sigma_a0**-2.0)
	
	mask = (M > 1E10) & (z <= z_bins[-1])

	return c, mask

###################################################################################################

def klypin15_m_c(M, z, mdef):
	"""
	The mass-based fits of Klypin et al. 2015.
	
	Klypin et al. 2015 suggest both peak height-based and mass-based fitting functions for 
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
	klypin15_nu_c: An alternative fitting function suggested in the same paper.
	"""
	if not mdef in ['200c', 'vir']:
		msg = 'Invalid mass definition for Klypin et al 2015 m-based model, %s.' % mdef
		raise Exception(msg)

	cosmo = Cosmology.getCurrent()

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
		msg = 'Invalid cosmology for Klypin et al 2015 m-based model, %s.' % cosmo.name
		raise Exception(msg)

	C0 = numpy.interp(z, z_bins, C0_bins)
	gamma = numpy.interp(z, z_bins, gamma_bins)
	M0 = numpy.interp(z, z_bins, M0_bins)
	M0 *= 1E12

	c = C0 * (M / 1E12)**-gamma * (1.0 + (M / M0)**0.4)
	
	mask = (M > 1E10) & (z <= z_bins[-1])

	return c, mask

###################################################################################################
# DUTTON & MACCIO 2014 MODEL
###################################################################################################

def dutton14_c(M, z, mdef):
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
		a = 0.520 + (0.905 - 0.520) * numpy.exp(-0.617 * z**1.21)
		b = -0.101 + 0.026 * z
	elif mdef == 'vir':
		a = 0.537 + (1.025 - 0.537) * numpy.exp(-0.718 * z**1.08)
		b = -0.097 + 0.024 * z
	else:
		msg = 'Invalid mass definition for Dutton & Maccio 2014 model, %s.' % mdef
		raise Exception(msg)
	
	logc = a + b * numpy.log10(M / 1E12)
	c = 10**logc

	mask = (M > 1E10) & (z <= 5.0)

	return c, mask

###################################################################################################
# BHATTACHARYA ET AL 2013 MODEL
###################################################################################################

def bhattacharya13_c(M, z, mdef):
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

	cosmo = Cosmology.getCurrent()
	D = cosmo.growthFactor(z)
	
	# Note that peak height in the B13 paper is defined wrt. the mass definition in question, so 
	# we can just use M to evaluate nu. 
	nu = cosmo.peakHeight(M, z)

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
# PRADA ET AL 2012 MODEL
###################################################################################################

def prada12_c200c(M200c, z):
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
		return 3.681 + (5.033 - 3.681) * (1.0 / math.pi * math.atan(6.948 * (x - 0.424)) + 0.5)
	def smin(x):
		return 1.047 + (1.646 - 1.047) * (1.0 / math.pi * math.atan(7.386 * (x - 0.526)) + 0.5)

	cosmo = Cosmology.getCurrent()
	nu = cosmo.peakHeight(M200c, z)

	a = 1.0 / (1.0 + z)
	x = (cosmo.OL0 / cosmo.Om0) ** (1.0 / 3.0) * a
	B0 = cmin(x) / cmin(1.393)
	B1 = smin(x) / smin(1.393)
	temp_sig = 1.686 / nu
	temp_sigp = temp_sig * B1
	temp_C = 2.881 * ((temp_sigp / 1.257) ** 1.022 + 1) * numpy.exp(0.06 / temp_sigp ** 2)
	c200c = B0 * temp_C

	return c200c

###################################################################################################
# KLYPIN ET AL 2011 MODEL
###################################################################################################

def klypin11_cvir(Mvir, z):
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
# DUFFY ET AL 2008 MODEL
###################################################################################################

def duffy08_c(M, z, mdef):
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
# BULLOCK ET AL 2001 / MACCIO ET AL 2008 MODEL
###################################################################################################

def bullock01_c200c(M200c, z):
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
	cosmo = Cosmology.getCurrent()
	interp = cosmo._zInterpolator('growthfactor', cosmo._growthFactorExact, inverse = True, future = True)
	Dmin = interp.get_knots()[0]
	Dmax = interp.get_knots()[-1]

	# The math works out such that we are looking for the redshift where the growth factor is
	# equal to the peak height of a halo with mass F * M.
	M_array, is_array = Utilities.getArray(M200c)
	D_target = cosmo.peakHeight(F * M_array, 0.0)
	mask = (D_target > Dmin) & (D_target < Dmax)

	N = len(M_array)
	c200c = numpy.zeros((N), dtype = float)
	H0 = cosmo.Hz(z)
	for i in range(N):
		if mask[i]:
			zc = interp(D_target[i])
			Hc = cosmo.Hz(zc)
			c200c[i] = K * (Hc / H0)**0.6666
		else:
			c200c[i] = 0.0
	
	if not is_array:
		c200c = c200c[0]
		mask = mask[0]
		
	return c200c, mask

###################################################################################################
