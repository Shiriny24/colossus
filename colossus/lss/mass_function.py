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

Of course, the function accepts numpy arrays for the mass parameter. By default, the mass function 
is returned as :math:`f(\\sigma)`, the natural units in Press-Schechter theory, where

.. math::
	\\frac{dn}{d \\ln(M)} = f(\\sigma) \\frac{\\rho_0}{M} \\frac{d \\ln(\\sigma^{-1})}{d \\ln(M)} 

where :math:`\\sigma` is the variance on the lagrangian size scale of the halo mass in question
(see :mod:`cosmology.cosmology`). The function can also return the mass function in other units,
namely :math:`dn/d\\ln(M)` (indicated by q_out = ``dndlnM``) and :math:`M^2 dn/dM` (indicated by 
q_out = ``M2dndM``). These conversions can be performed separately using the 
:func:`convertMassFunction` function.

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

Note that the mass definition (set to ``fof`` by default) needs to match one of the allowed mass 
definitions of the chosen model. For most models, only ``fof`` is allowed, but some SO models are 
calibrated to various mass definitions. The ``tinker08`` model can handle any overdensity between 
200m and 3200m (though they can be expressed as critical and virial overdensities as well).

There are two different types of redshift dependence of :math:`f(\\sigma)` listed above: some 
models explicitly depend on redshift (e.g., ``bhattacharya11``), some models only change through 
the small variation of the collapse overdensity :math:`\\delta_{\\rm c}` 
(see :func:`lss.lss.collapseOverdensity`). The ``tinker08`` model depends on redshift only through 
the conversion of the overdensity threshold.

The mass functions predicted by the models are, in principle, supposed to be valid across redshifts
and cosmologies. However, it has been shown that this is true only approximately. The functions in
this module do not check whether a model is used outside the range where it was calibrated.

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

class HaloMassFunctionModel():
	"""
	This object contains certain characteristics of a mass function model, namely the mass 
	definitions for which it is valid, whether it explicitly depends on the redshift (in some cases
	this dependence arises because of the slight dependence of the collapse overdensity on z), and
	how the collapse overdensity is computed by default (if applicable).

	The ``models`` variable is a dictionary of :class:`SplashbackModel` objects containing all 
	available models. The user can overwrite the properties of these models at their own risk.
	"""
	def __init__(self):
		
		self.func = None
		self.z_dependence = False
		self.deltac_dependence = False
		self.mdef_dependence = False
		self.mdefs = []
		
		return

###################################################################################################

models = {}

models['press74'] = HaloMassFunctionModel()
models['press74'].mdefs = ['fof']
models['press74'].deltac_dependence = True

models['sheth99'] = HaloMassFunctionModel()
models['sheth99'].mdefs = ['fof']
models['sheth99'].deltac_dependence = True

models['jenkins01'] = HaloMassFunctionModel()
models['jenkins01'].mdefs = ['fof']

models['reed03'] = HaloMassFunctionModel()
models['reed03'].mdefs = ['fof']
models['reed03'].deltac_dependence = True

models['warren06'] = HaloMassFunctionModel()
models['warren06'].mdefs = ['fof']

models['tinker08'] = HaloMassFunctionModel()
models['tinker08'].mdefs = ['*']
models['tinker08'].z_dependence = True
models['tinker08'].mdef_dependence = True

models['crocce10'] = HaloMassFunctionModel()
models['crocce10'].mdefs = ['fof']
models['crocce10'].z_dependence = True

models['courtin11'] = HaloMassFunctionModel()
models['courtin11'].mdefs = ['fof']

models['bhattacharya11'] = HaloMassFunctionModel()
models['bhattacharya11'].mdefs = ['fof']
models['bhattacharya11'].deltac_dependence = True

models['watson13'] = HaloMassFunctionModel()
models['watson13'].mdefs = ['fof']

###################################################################################################

def massFunction(x, z, q_in = 'M', mdef = 'fof', q_out = 'f', 
				model = defaults.HALO_MASS_FUNCTION_MODEL,
				sigma_args = {}, deltac_args = {}):
	"""
	The abundance of halos as a function of mass (or sigma) and redshift.
	
	This function is a wrapper for all individual models implemented in this module. It accepts
	either mass or the variance sigma and redshift as input, as well as a mass definition which 
	is set to ``fof`` by default (see the model table for valid redshifts). The output units are 
	controlled by the ``q_out`` parameter, see the basic usage section for details.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	x: array_like
		Either halo mass in :math:`M_{\odot}/h` or the variance :math:`\sigma`, depending on the 
		value of the ``q_in`` parameter; can be a number or a numpy array.
	z: float
		Redshift
	q_in: str
		Either ``M`` or ``sigma``, indicating which is passed for the ``x`` parameter.
	mdef: str
		The mass definition in which the halo mass M is given and which the returned mass function
		refers to. Please see the model table for the mass definitions for which each model is 
		valid.
	q_out: str
		The units in which the mass function is returned.
	model: str
		The model of the mass function used.
	sigma_args: dict
		Extra arguments to be passed to the :func:`cosmology.cosmology.Cosmology.sigma` function 
		when mass is converted to sigma.
	deltac_args: dict
		Extra parameters that are passed to the :func:`lss.lss.collapseOverdensity` function; see 
		the documentation of the individual models for possible parameters. Note that not all 
		models of the mass function rely on the collapse overdensity.
		
	Returns
	-----------------------------------------------------------------------------------------------
	mfunc: array_like
		The halo mass function in the desired units.
	"""

	# Compute sigma
	cosmo = cosmology.getCurrent()
	
	M = None
	if q_in == 'M':
		M = x
		R = lss.lagrangianR(x)
		sigma = cosmo.sigma(R, z, **sigma_args)
	elif q_in == 'sigma':
		sigma = x
	else:
		raise Exception('Unknown input quantity, %s.' % (q_in))

	# Evaluate model
	if not model in models.keys():
		msg = 'Unknown model, %s.' % (model)
		raise Exception(msg)

	model_props = models[model]
	args = (sigma,)
	if model_props.z_dependence or model_props.deltac_dependence:
		args += (z,)
	if model_props.mdef_dependence:
		args += (mdef,)
	if model_props.deltac_dependence:
		args += (deltac_args,)

	if not '*' in model_props.mdefs and not mdef in model_props.mdefs:
		raise Exception('The mass definition %s is not allowed for model %s. Allowed are: %s.' % \
					(mdef, model, str(model_props.mdefs)))

	f = model_props.func(*args)

	if q_out == 'f':
		mfunc = f
	else:
		if M is None:
			R = cosmo.sigma(sigma, inverse = True, **sigma_args)
			M = lss.lagrangianM(R)
		mfunc = convertMassFunction(f, M, z, 'f', q_out)

	return mfunc

###################################################################################################

def convertMassFunction(mfunc, M, z, q_in, q_out):
	"""
	Convert different units of the mass function.
	
	Virtually all models parameterize the mass function in the natural Press-Schechter units, 
	:math:`f(\\sigma)`. This function convert any allowed units into any other units. See the 
	basic usage section for details.
	
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
	R = lss.lagrangianR(M)
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

def modelPress74(sigma, z, deltac_args = {'corrections': True}):
	"""
	The mass function model of Press & Schechter 1974.
	
	This model depends on redshift only through the collapse overdensity :math:`\\delta_{\\rm c}`.
	By default, the collapse overdensity is computed including corrections due to cosmology.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		The variance corresponding to the desired halo mass.
	z: float
		Redshift
	deltac_corrections: bool
		If True, include non-EdS corrections in the computation of the collapse overdensity.
	
	Returns
	-----------------------------------------------------------------------------------------------
	f: array_like
		The halo mass function.
	"""
	
	print(deltac_args)
	
	delta_c = lss.collapseOverdensity(z = z, **deltac_args)
	nu = delta_c / sigma
	f = np.sqrt(2.0 / np.pi) * nu * np.exp(-0.5 * nu**2)
	
	return f

###################################################################################################

def modelSheth99(sigma, z, deltac_args = {'corrections': True}):
	"""
	The mass function model of Sheth & Tormen 1999.
	
	This model was created to account for the differences between the classic Press-Schechter model
	and measurements of the halo abundance in numerical simulations. The model is given in Equation 
	10. Note that the collapse overdensity is computed including corrections due to dark energy. 
	
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
		
	delta_c = lss.collapseOverdensity(z = z, **deltac_args)
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
	
	The model is given in Equation 9. It does not explicitly rely on the collapse overdensity and
	thus has no redshift evolution.
	
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

def modelReed03(sigma, z, deltac_args = {'corrections': True}):
	"""
	The mass function model of Reed et al. 2003.
	
	This model corrects the Sheth & Tormen 1999 model at high masses, the functional form is given
	in Equation 9.
	
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
		
	f_ST = modelSheth99(sigma, z, deltac_args = deltac_args)
	f = f_ST * np.exp(-0.7 / (sigma * np.cosh(2.0 * sigma)**5))
	
	return f

###################################################################################################

def modelWarren06(sigma):
	"""
	The mass function model of Warren et al. 2006.
	
	This model does not explicitly rely on the collapse overdenisty and thus has no redshift 
	dependence. The functional form is given in Equation 5. 
	
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
	
	This model was the first calibrated for SO rather than FOF masses, and can predict the mass 
	function for a large range of overdensities (:math:`200 \\leq \\Delta_{\\rm m} \\leq 3200`).
	The authors found that the SO mass function is not universal with redshift and took this 
	dependence into account explicitly (Equations 3-8).
	
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
	Delta_m = round(mass_so.densityThreshold(z, mdef) / cosmo.rho_m(z))

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

def modelCrocce10(sigma, z):
	"""
	The mass function model of Crocce et al. 2010.
	
	This function was calibrated between z = 0 and 1, and is given in Equation 22.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		The variance corresponding to the desired halo mass.
		
	Returns
	-----------------------------------------------------------------------------------------------
	f: array_like
		The halo mass function.
	"""
		
	zp1 = 1.0 + z
	A = 0.58 * zp1**-0.13
	a = 1.37 * zp1**-0.15
	b = 0.30 * zp1**-0.084
	c = 1.036 * zp1**-0.024
	
	f = A * (sigma**-a + b) * np.exp(-c / sigma**2)
	
	return f

###################################################################################################

def modelCourtin11(sigma):
	"""
	The mass function model of Courtin et al. 2011.
	
	The model is specified in Equation 22. It uses a fixed collapse overdensity 
	:math:`\\delta_{\\rm c} = 1.673` and thus does not evolve with redshift.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		The variance corresponding to the desired halo mass.
		
	Returns
	-----------------------------------------------------------------------------------------------
	f: array_like
		The halo mass function.
	"""
		
	delta_c = 1.673
	A = 0.348
	a = 0.695
	p = 0.1
	
	f = A * np.sqrt(2 * a / np.pi) * delta_c / sigma * (1.0 + (delta_c / sigma / np.sqrt(a))**(-2 * p)) * np.exp(-delta_c**2 * a / (2 * sigma**2))

	return f

###################################################################################################

def modelBhattacharya11(sigma, z, deltac_args = {'corrections': False}):
	"""
	The mass function model of Bhattacharya et al. 2011.
	
	This model was calibrated between redshift 0 and 2. The authors found that varying 
	:math:`\\delta_{\\rm c}` does not account for the redshift dependence. Thus, they keep it 
	fixed and added an explicit redshift dependence into the model. The functional form is given 
	in Table 4.
	
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
		
	delta_c = lss.collapseOverdensity(z = z, **deltac_args)
	nu = delta_c / sigma
	nu2 = nu**2

	zp1 = 1.0 + z
	A = 0.333 * zp1**-0.11
	a = 0.788 * zp1**-0.01
	p = 0.807
	q = 1.795

	f = A * np.sqrt(2 / np.pi) * np.exp(-a * nu2 * 0.5) * (1.0 + (a * nu2)**-p) * (nu * np.sqrt(a))**q
	
	return f

###################################################################################################

def modelWatson13(sigma):
	"""
	The mass function model of Watson et al. 2013.
	
	This function currently only contains the model for the FOF mass function as given in 
	Equation 12.
	
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
# Pointers to model functions
###################################################################################################

models['press74'].func = modelPress74
models['sheth99'].func = modelSheth99
models['jenkins01'].func = modelJenkins01
models['reed03'].func = modelReed03
models['warren06'].func = modelWarren06
models['tinker08'].func = modelTinker08
models['crocce10'].func = modelCrocce10
models['courtin11'].func = modelCourtin11
models['bhattacharya11'].func = modelBhattacharya11
models['watson13'].func = modelWatson13

###################################################################################################
