###################################################################################################
#
# mass_function.py          (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
The mass function quantifies how many halos of a given mass have formed at a given redshift and
cosmology.

---------------------------------------------------------------------------------------------------
Basic usage
---------------------------------------------------------------------------------------------------

This module implements a range of models for the halo mass function. The easiest and recommended
use of the module is through the :func:`massFunction` function, a wrapper for all individual
models::
	
	setCosmology('WMAP9')
	mfunc_so = massFunction(1E12, 0.0, mdef = 'vir', model = 'tinker08')
	mfunc_fof = massFunction(1E12, 0.0, mdef = 'fof', model = 'watson13')

Of course, the function accepts numpy arrays for the mass parameter. By default, the function 
returns f, but it can also return other units.

---------------------------------------------------------------------------------------------------
Mass function models
---------------------------------------------------------------------------------------------------

The following models are supported in this module, and their ID can be passed as the ``model`` 
parameter to the :func:`massFunction` function:

============== ========== =========== ======================================
ID             mdefs      z-dep.       Reference
============== ========== =========== ======================================
press74        fof        delta_c     `Press& Schechter 1974 <http://adsabs.harvard.edu/abs/1974ApJ...187..425P>`_
sheth99	       fof        delta_c     `Sheth & Tormen 1999 <http://adsabs.harvard.edu/abs/1999MNRAS.308..119S>`_
jenkins01      fof        No	      `Jenkins et al. 2001 <http://adsabs.harvard.edu/abs/2001MNRAS.321..372J>`_
reed03	       fof        delta_c     `Reed et al. 2003 <http://adsabs.harvard.edu/abs/2003MNRAS.346..565R>`_
warren06       fof        No	      `Warren et al. 2006 <http://adsabs.harvard.edu/abs/2006ApJ...646..881W>`_
tinker08       Any SO     Yes	      `Tinker et al. 2008 <http://adsabs.harvard.edu/abs/2008ApJ...688..709T>`_
crocce10       fof        No          `Crocce et al. 2010 <http://adsabs.harvard.edu/abs/2010MNRAS.403.1353C>`_
courtin11      fof        No	      `Courtin et al. 2011 <http://adsabs.harvard.edu/abs/2011MNRAS.410.1911C>`_
bhattacharya11 fof        Yes         `Bhattacharya et al. 2011 <http://adsabs.harvard.edu/abs/2011ApJ...732..122B>`_
watson13       fof        No (FOF)    `Watson et al. 2013 <http://adsabs.harvard.edu/abs/2013MNRAS.433.1230W>`_
============== ========== =========== ======================================

---------------------------------------------------------------------------------------------------
Module reference
--------------------------------------------------------------------------------------------------- 
"""

###################################################################################################

import numpy as np

from colossus import defaults
from colossus.cosmology import cosmology
from colossus.lss import lss
from colossus.halo import mass_so

###################################################################################################

models = ['press74', 'sheth99', 'jenkins01', 'reed03', 'warren06', 'tinker08', 'crocce10', 
		'courtin11', 'bhattacharya11', 'watson13']
"""A list of all implemented mass function models."""

###################################################################################################

def massFunction(M, z,
				mdef = 'fof', q_out = 'f', model = defaults.HALO_MASS_FUNCTION_MODEL,
				sigma_args = {}, **kwargs):
	"""
	The abundance of halos as a function of mass and redshift.
	
	This function is a wrapper for all individual models implemented in this module. It needs mass
	and redshift, as well as a mass definition which is set to 'fof' by default (see the model 
	table for valid redshifts). See the documentation of the :func:`convertMassFunction` function
	for the units in which the mass function can be returned, as controlled by the ``q_out`` 
	parameter.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition in which the halo mass M is given and which the returned mass function
		refers to. Please see the model table for the mass definitions for which each model is 
		valid.
	q_out: str
		The units in which the mass function is returned; see :func:`convertMassFunction`.
	model: str
		The model of the mass function used.
	sigma_args: dict
		Extra arguments to be passed to the :func:`cosmology.cosmology.Cosmology.sigma` function when mass
		is converted to sigma.
	kwargs: kwargs
		Extra parameters that are passed to the mass function; see the documentation of the 
		individual models for possible parameters.
		
	Returns
	-----------------------------------------------------------------------------------------------
	mfunc: array_like
		The halo mass function in the desired units.
	"""

	# Compute sigma
	cosmo = cosmology.getCurrent()
	R = lss.lagrangianR(M)
	sigma = cosmo.sigma(R, z, **sigma_args)

	# Evaluate model
	if model == 'press74':
		func = modelPress74
		args = (sigma, z)
		mdefs = ['fof']
		
	elif model == 'sheth99':	
		func = modelSheth99
		args = (sigma, z)
		mdefs = ['fof']

	elif model == 'jenkins01':	
		func = modelJenkins01
		args = (sigma,)
		mdefs = ['fof']

	elif model == 'reed03':	
		func = modelReed03
		args = (sigma, z)
		mdefs = ['fof']

	elif model == 'warren06':	
		func = modelWarren06
		args = (sigma,)
		mdefs = ['fof']

	elif model == 'tinker08':
		func = modelTinker08
		args = (sigma, z, mdef)
		mdefs = ['*']
	
	elif model == 'crocce10':
		func = modelCrocce10
		args = (sigma,)
		mdefs = ['fof']
	
	elif model == 'courtin11':	
		func = modelCourtin11
		args = (sigma,)
		mdefs = ['fof']

	elif model == 'bhattacharya11':	
		func = modelBhattacharya11
		args = (sigma, z)
		mdefs = ['fof']

	elif model == 'watson13':
		func = modelWatson13
		args = (sigma,)
		mdefs = ['fof']
	
	else:
		msg = 'Unknown model, %s.' % (model)
		raise Exception(msg)

	if not '*' in mdefs and not mdef in mdefs:
		raise Exception('The mass definition %s is not allowed for model %s. Allowed are: %s.' % \
					(mdef, model, str(mdefs)))

	f = func(*args)

	mfunc = f
	if q_out != 'f':
		mfunc = convertMassFunction(f, M, z, 'f', q_out)

	return mfunc

###################################################################################################

def convertMassFunction(mfunc, M, z, q_in, q_out):
	"""
	Convert different units of the mass function.
	
	The mass function is typically given in ...
	
	Parameters
	-----------------------------------------------------------------------------------------------
	mfunc: array_like
		The mass function in the input units.
	M: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	q_in: str
		The units in which the mass function is input; can be ``f``, ``dndlnM``, or ``M2dndM``. See
		table on top of this file for the meaning of these units.
	q_out: str
		The units in which the mass function is returned; see above.
		
	Returns
	-----------------------------------------------------------------------------------------------
	mfunc: array_like
		The halo mass function in the desired units.
	"""

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

# TODO z dependence

def modelPress74(sigma, z):
	"""
	The mass function model of Press & Schechter 1974.
	
	...
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		The variance corresponding to the desired halo mass.
	z: float
		Redshift
		
	Returns
	-----------------------------------------------------------------------------------------------
	f: array_like
		The halo mass function.
	"""
	
	delta_c = lss.collapseOverdensity(corrections = True, z = z)
	f = np.sqrt(2.0 / np.pi) * delta_c / sigma * np.exp(-0.5 * delta_c**2 / sigma**2)
	
	return f

###################################################################################################

# TODO
# Equation 10 in Sheth & Tormen 1999. The extra factor of two in front is due to the definition
# according to which the PS-mass function would correspond to A = 0.5. ????

def modelSheth99(sigma, z):
	"""
	The mass function model of Sheth & Tormen 1999.
	
	...
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		The variance corresponding to the desired halo mass.
	z: float
		Redshift
		
	Returns
	-----------------------------------------------------------------------------------------------
	f: array_like
		The halo mass function.
	"""
		
	delta_c = lss.collapseOverdensity(corrections = True, z = z)
	A = 0.3222
	a = 0.707
	p = 0.3
	
	nu_p = a * delta_c**2 / sigma**2
	f = 2.0 * A * np.sqrt(nu_p / 2.0 / np.pi) * np.exp(-0.5 * nu_p) * (1.0 + nu_p**-p)
	
	return f

###################################################################################################

def modelJenkins01(sigma):
	"""
	The mass function model of Jenkins et al. 2001.
	
	... equation 9
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		The variance corresponding to the desired halo mass.
		
	Returns
	-----------------------------------------------------------------------------------------------
	f: array_like
		The halo mass function.
	"""
		
	f = 0.315 * np.exp(-np.abs(np.log(1.0 / sigma) + 0.61)**3.8)
	
	return f

###################################################################################################

def modelReed03(sigma, z):
	"""
	The mass function model of Reed et al. 2003.
	
	... equ 9
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		The variance corresponding to the desired halo mass.
	z: float
		Redshift
		
	Returns
	-----------------------------------------------------------------------------------------------
	f: array_like
		The halo mass function.
	"""
		
	f_ST = modelSheth99(sigma, z)
	f = f_ST * np.exp(-0.7 / (sigma * np.cosh(2.0 * sigma)**5))
	
	return f

###################################################################################################

def modelWarren06(sigma):
	"""
	The mass function model of Warren et al. 2006.
	
	... equ 5
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		The variance corresponding to the desired halo mass.
		
	Returns
	-----------------------------------------------------------------------------------------------
	f: array_like
		The halo mass function.
	"""
		
	A = 0.7234
	a = 1.625
	b = 0.2538
	c = 1.1982
	
	f = A * (sigma**-a + b) * np.exp(-c / sigma**2)
	
	return f

###################################################################################################

def modelTinker08(sigma, z, mdef):
	"""
	The mass function model of Tinker et al. 2008.
	
	...
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		The variance corresponding to the desired halo mass.
	z: float
		Redshift
	mdef: str
		The mass definition to which sigma corresponds.
		
	Returns
	-----------------------------------------------------------------------------------------------
	f: array_like
		The halo mass function.
	"""
	
	if mdef == 'fof':
		raise Exception('Cannot use mass definition fof for Tinker 08 model, need an SO definition.')
	
	cosmo = cosmology.getCurrent()
	Delta_m = mass_so.densityThreshold(z, mdef) / cosmo.rho_m(z)

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

# Their model is given for z = 0 and 0.5; not clear how to apply this, so this is only z = 0

def modelCrocce10(sigma):
	"""
	The mass function model of Crocce et al. 2010.
	
	... equ 5, table 2
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		The variance corresponding to the desired halo mass.
		
	Returns
	-----------------------------------------------------------------------------------------------
	f: array_like
		The halo mass function.
	"""
		
	A = 0.58
	a = 1.37
	b = 0.30
	c = 1.036
	
	f = A * (sigma**-a + b) * np.exp(-c / sigma**2)
	
	return f

###################################################################################################

def modelCourtin11(sigma):
	"""
	The mass function model of Courtin et al. 2011.
	
	... equ 22
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		The variance corresponding to the desired halo mass.
		
	Returns
	-----------------------------------------------------------------------------------------------
	f: array_like
		The halo mass function.
	"""
		
	#delta_c = lss.collapseOverdensity(corrections = True, z = z)
	delta_c = 1.673
	A = 0.348
	a = 0.695
	p = 0.1
	
	f = A * np.sqrt(2 * a / np.pi) * delta_c / sigma * (1.0 + (delta_c / sigma / np.sqrt(a))**(-2 * p)) * np.exp(-delta_c**2 * a / (2 * sigma**2))

	return f

###################################################################################################

def modelBhattacharya11(sigma, z):
	"""
	The mass function model of Bhattacharya et al. 2011.
	
	... equ 12
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		The variance corresponding to the desired halo mass.
	z: float
		Redshift
		
	Returns
	-----------------------------------------------------------------------------------------------
	f: array_like
		The halo mass function.
	"""
		
	# TODO why no corrections?
	delta_c = lss.collapseOverdensity(corrections = False, z = z)
	
	#A0 = 0.333
	a0 = 0.788
	A = 0.333 / (1.0 + z)**0.11
	a = a0 / (1.0 + z)**0.01
	p0 = 0.807
	q0 = 1.795
	
	# TODO A or A0?
	f = A * np.sqrt(2  / np.pi) * np.exp(-a0 * delta_c**2 / (2 * sigma**2)) \
		* (1.0 + (sigma**2 / a0 / delta_c**2)**p0) * (delta_c * np.sqrt(a) / sigma)**q0
	
	return f

###################################################################################################

def modelWatson13(sigma):
	"""
	The mass function model of Watson et al. 2013.
	
	...
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		The variance corresponding to the desired halo mass.
		
	Returns
	-----------------------------------------------------------------------------------------------
	f: array_like
		The halo mass function.
	"""
		
	A = 0.282
	alpha = 2.163
	beta = 1.406
	gamma = 1.210
	
	f = A * ((beta / sigma)**alpha + 1.0) * np.exp(-gamma / sigma**2)
	
	return f

###################################################################################################
