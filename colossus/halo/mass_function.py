###################################################################################################
#
# mass_function.py          (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module implements a range of models for the halo mass function as a function of mass, redshift, 
and cosmology. The main function in this module, :func:`massFunction`, is a wrapper for all 
models::
	
	setCosmology('WMAP9')
	mfunc = massFunction(1E12, 'vir', 0.0, model = 'tinker08')

Alternatively, the user can also call the individual model functions directly.

---------------------------------------------------------------------------------------------------
Mass function models
---------------------------------------------------------------------------------------------------

The following models are supported in this module, and their ID can be passed as the ``model`` 
parameter to the :func:`massFunction` function:

============== ================ ================== =========== ========== ======================================
ID             Native mdefs     M range (z=0)      z range     Cosmology  Paper
============== ================ ================== =========== ========== ======================================
tinker08       200c             Any                Any         Any        
============== ================ ================== =========== ========== ======================================

---------------------------------------------------------------------------------------------------
Module reference
--------------------------------------------------------------------------------------------------- 
"""

###################################################################################################

import numpy as np

from colossus.utils import constants
from colossus import defaults
from colossus.cosmology import cosmology
from colossus.halo import mass_so

###################################################################################################

MODELS = ['tinker08', 'watson13_fof']
"""A list of all implemented mass function models."""

###################################################################################################

def massFunction(M, mdef, z,
				model = defaults.HALO_MASS_FUNCTION_MODEL):
	"""
	The abundance of halos as a function of mass and redshift.
	
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
		
	Returns
	-----------------------------------------------------------------------------------------------
	mfunc: array_like
		The halo mass function
	"""

	# Compute peak height and sigma
	cosmo = cosmology.getCurrent()
	nu = cosmo.peakHeight(M, z)
	sigma = constants.DELTA_COLLAPSE / nu

	# Parse mass definition, convert to Delta_m equivalent
	mdef_type, mdef_delta = mass_so.parseMassDefinition(mdef)
	if mdef_type == 'c':
		Delta_m = mdef_delta / cosmo.Om(z)
	elif mdef_type == 'm':
		Delta_m = mdef_delta
	elif mdef_type == 'vir':
		Delta_m = mass_so.deltaVir(z) / cosmo.Om(z)
	else:
		msg = 'Invalid mass definition, %s.' % mdef
		raise Exception(msg)

	# Evaluate model
	if model == 'press74':	
		f = modelPress74(sigma)
		
	elif model == 'sheth99':	
		f = modelSheth99(sigma)

	elif model == 'jenkins01':	
		f = modelJenkins01(sigma)

	elif model == 'sheth02':	
		f = modelSheth02(sigma)

	elif model == 'reed03':	
		f = modelReed03(sigma)

	elif model == 'tinker08':
		f = modelTinker08(sigma, Delta_m, z)
	
	elif model == 'watson13_fof':
		f = modelWatson13_fof(sigma)
	
	elif model == 'watson13_so':
		f = modelWatson13_so(sigma, Delta_m, z)
	
	else:
		msg = 'Unknown model, %s.' % (model)
		raise Exception(msg)

	return f

###################################################################################################

def convertMassFunction(mfunc, M, z, q_in, q_out):

	if q_in == q_out:
		return mfunc

	cosmo = cosmology.getCurrent()
	R = cosmo.lagrangianR(M)
	d_ln_sigma_d_ln_R = cosmo.sigma(R, z, derivative = True)
	rho_Mpc = cosmo.rho_m(0.0) * 1E9
	
	if q_in == 'dndlnM':
		dn_dlnM = mfunc

	elif q_in == 'f':
		dn_dlnM = -(1.0 / 3.0) * mfunc * rho_Mpc / M * d_ln_sigma_d_ln_R
	
	elif q_in == 'M2dndM':
		dn_dlnM = mfunc / M * rho_Mpc
	
	else:
		raise Exception('Cannot handle input quantity %s.' % q_in)
	
	if q_out == 'dndlnM':
		mfunc_out = dn_dlnM
		
	elif q_out == 'M2dndM':
		mfunc_out = dn_dlnM * M / rho_Mpc
		
	elif q_out == 'f':
		mfunc_out = -3.0 * dn_dlnM * M / rho_Mpc / d_ln_sigma_d_ln_R
	
	else:
		raise Exception('Cannot handle output quantity %s.' % q_out)
	
	return mfunc_out

###################################################################################################
# FUNCTIONS FOR INDIVIDUAL MASS FUNCTION MODELS
###################################################################################################

def modelPress74(sigma):
	
	delta_c = constants.DELTA_COLLAPSE
	f = np.sqrt(2.0 / np.pi) * delta_c / sigma * np.exp(-0.5 * delta_c**2 / sigma**2)
	
	return f

###################################################################################################

# TODO
# Equation 10 in Sheth & Tormen 1999. The extra factor of two in front is due to the definition
# according to which the PS-mass function would correspond to A = 0.5. ????

def modelSheth99(sigma):
	
	delta_c = constants.DELTA_COLLAPSE
	A = 0.3222
	a = 0.707
	p = 0.3
	
	nu_p = a * delta_c**2 / sigma**2
	f = 2.0 * A * np.sqrt(nu_p / 2.0 / np.pi) * np.exp(-0.5 * nu_p) * (1.0 + nu_p**-p)
	
	return f

###################################################################################################

def modelJenkins01(sigma):
	
	f = 0.315 * np.exp(-np.abs(np.log(1.0 / sigma) + 0.61)**3.8)
	
	return f

###################################################################################################

def modelSheth02(sigma):
	
	delta_c = constants.DELTA_COLLAPSE
	#A = 0.3222
	a = 0.707
	#p = 0.3
	
	nu_p = a * delta_c**2 / sigma**2
	t1 = nu_p**-0.6
	f = 2.0 * (1.0 + 0.094 * t1) * np.sqrt(nu_p / 2.0 / np.pi) * np.exp(-nu_p * (1.0 + 0.5 * t1)**2 / 2.0)
	
	return f

###################################################################################################

def modelReed03(sigma):
	
	f_ST = modelSheth99(sigma)
	f = f_ST * np.exp(-0.7 / (sigma * np.cosh(2.0 * sigma)**5))
	
	return f

###################################################################################################

# The AHF fit

def modelWatson13_so(sigma, Delta_m, z):
	
	cosmo = cosmology.getCurrent()
	
	A = 0.194
	alpha = 1.805
	beta = 2.267
	gamma = 1.287
	
	f_178 = A * ((beta / sigma)**alpha + 1.0) * np.exp(-gamma / sigma**2)
	
	Delta_178 = Delta_m / 178.0
	C = np.exp(0.023 * (Delta_178 - 1.0))
	d = -0.456 * cosmo.Om(z) - 0.139
	Gamma = C * Delta_178**d * np.exp(0.072 * (1.0 - Delta_178) / sigma**2.130)
	f = f_178 * Gamma
	
	return f

###################################################################################################

def modelWatson13_fof(sigma):
	
	A = 0.282
	alpha = 2.163
	beta = 1.406
	gamma = 1.210
	
	f = A * ((beta / sigma)**alpha + 1.0) * np.exp(-gamma / sigma**2)
	
	return f

###################################################################################################

def modelTinker08(sigma, Delta_m, z):

	fit_Delta = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
	fit_A0 = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
	fit_a0 = np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
	fit_b0 = np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
	fit_c0 = np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])
		
	# Compute fit parameters and f-function
	if Delta_m < fit_Delta[0]:
		raise Exception('Delta_m %d is too small, minimum %d.' % (Delta_m, fit_Delta[0]))
	if Delta_m > fit_Delta[-1]:
		raise Exception('Delta_m %d is too large, maximum %d.' % (Delta_m, fit_Delta[-1]))
	
	A0 = np.interp(Delta_m, fit_Delta, fit_A0)
	a0 = np.interp(Delta_m, fit_Delta, fit_a0)
	b0 = np.interp(Delta_m, fit_Delta, fit_b0)
	c0 = np.interp(Delta_m, fit_Delta, fit_c0)
	
	alpha = 10**(-(0.75 / np.log10(Delta_m / 75.0))**1.2)
	A = A0 * (1.0 + z)**-0.14
	a = a0 * (1.0 + z)**-0.06
	b = b0 * (1.0 + z)**-alpha
	c = c0
	f = A * ((sigma / b)**-a + 1.0) * np.exp(-c / sigma**2)
	
	return f

###################################################################################################
