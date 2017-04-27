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
tinker08       200c             Any                Any         Any        Diemer & Kravtsov 2015 (ApJ 799, 108)
============== ================ ================== =========== ========== ======================================

---------------------------------------------------------------------------------------------------
Module reference
--------------------------------------------------------------------------------------------------- 
"""

###################################################################################################

import numpy as np
import scipy.interpolate
import scipy.optimize
import warnings

from colossus.utils import utilities
from colossus.utils import constants
from colossus import defaults
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.halo import mass_defs

###################################################################################################

MODELS = ['tinker08']
"""A list of all implemented concentration models."""

#INVALID_CONCENTRATION = -1.0
#"""The concentration value returned if the model routine fails to compute."""

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

	# ---------------------------------------------------------------------------------------------
	# Distinguish between models
	if model == 'tinker08':
		func = modelTinker08
		args = (z)
		limited = False
	
	else:
		msg = 'Unknown model, %s.' % (model)
		raise Exception(msg)

	return

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
# TINKER 08 MODEL
###################################################################################################

def modelTinker08(M, mdef, z):

	fit_Delta = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
	fit_A0 = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
	fit_a0 = np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
	fit_b0 = np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
	fit_c0 = np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])

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
