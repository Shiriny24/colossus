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
	splashback
	Rsp
	Msp

For more information, please see Diemer & Kravtsov 2014, Adhikari et al. 2014, and More, Diemer & 
Kravtsov 2015, and Diemer et al. 2017.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np

from colossus import defaults
from colossus.utils import utilities
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.halo import mass_defs

###################################################################################################

# A model can take a number of quantities as input (qx) and output a number of quantities (qy). 
# The model may additionally depend on variabes depends_on. The label and style fields are empty,
# but can be useful ...

class SplashbackModel():
	
	def __init__(self):
		
		self.label = ''
		self.qx = []
		self.qy = []
		self.depends_on = []
		self.min_Gamma = -np.inf
		self.max_Gamma = np.inf
		self.min_nu200m = 0.0
		self.max_nu200m = np.inf
		self.style = {}
		
		return

###################################################################################################

models = {}

models['more15'] = SplashbackModel()
models['more15'].qx = ['Gamma', 'nu200m']
models['more15'].qy = ['RspR200m', 'MspM200m', 'Deltasp']
models['more15'].depends_on = ['Gamma', 'z']

models['shi16'] = SplashbackModel()
models['shi16'].qx = ['Gamma']
models['shi16'].qy = ['RspR200m', 'MspM200m', 'Deltasp']
models['shi16'].depends_on = ['Gamma', 'z']
models['shi16'].min_Gamma = 0.5
models['shi16'].max_Gamma = 5.0

models['diemer17'] = SplashbackModel()
models['diemer17'].qx = ['Gamma', 'nu200m']
models['diemer17'].qy = ['RspR200m', 'MspM200m', 'Deltasp', 'RspR200m-1s', 'MspM200m-1s', 'Deltasp-1s']
models['diemer17'].depends_on = ['Gamma', 'z', 'nu', 'rspdef']

###################################################################################################

def splashback(qx, qy, x, z = None, nu = None, rspdef = None, statistic = None, model = 'diemer17'):
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

	# Check that this model exists, and that it can do what is requested
	if not model in models:
		raise Exception('Unknown model, %s.' % model)
	m = models[model]
	if not qx in m.qx:
		raise Exception('Model %s cannot handle input quantity %s.' % (model, qx))
	if not qy in m.qy:
		raise Exception('Model %s cannot output quantity %s.' % (model, qy))

	# Create a mask indicating where the results are valid
	if qx == 'Gamma':
		mask = (x >= m.min_Gamma) & (x <= m.max_Gamma)
	elif qx == 'nu200m':
		mask = (x >= m.min_nu200m) & (x <= m.max_nu200m)
	else:
		raise Exception('Unknown input quantity, %s.' % qx)
		
	if utilities.isArray(x):
		x = x[mask]
		if np.count_nonzero(mask) == 0:
			print('WARNING: Found no input values within the limits of the model %s.' % model)
			return None, False
	else:
		if not mask:
			print('WARNING: Input value %.1f is not within the limits of the model %s.' % (x, model))
			return None, False

	if z is None:
		Om = None
	else:
		cosmo = cosmology.getCurrent()
		Om = cosmo.Om(z)
	ret = None
	
	if model == 'diemer17':
		
		# The model is only valid between the 50th and 87th percentile
		p = modelDiemer17PercentileValue(rspdef)
		if (p > 0.0 and p < 0.5) or p > 0.87:
			mask[:] = False
			ret = np.array([])
			return ret, mask
		
		# If only nu200m is given, we compute Gamma from nu and z.
		if qx == 'Gamma':
			Gamma = x
		if qx == 'nu200m':
			nu = x
			Gamma = modelDiemer17Gamma(nu, z)
		
		if qy == 'RspR200m':
			ret = modelDiemer17RspR200m('RspR200m', Gamma, nu, z, rspdef)
		
		elif qy == 'MspM200m':
			ret = modelDiemer17RspR200m('MspM200m', Gamma, nu, z, rspdef)
		
		elif qy == 'Deltasp':
			rsp200m = modelDiemer17RspR200m('RspR200m', Gamma, nu, z, rspdef)
			msp200m = modelDiemer17RspR200m('MspM200m', Gamma, nu, z, rspdef)
			ret = 200.0 * msp200m / rsp200m**3
		
		else:
			if qx == 'nu200m':

				if qy == 'RspR200m-1s':
					ret = np.ones((len(mask)), np.float) * 0.07
				
				elif qy == 'MspM200m-1s':
					ret = np.ones((len(mask)), np.float) * 0.07
				
				elif qy == 'Deltasp-1s':
					ret = np.ones((len(mask)), np.float) * 0.15
				
			else:
		
				if qy == 'RspR200m-1s':
					ret = modelDiemer17Scatter('RspR200m-1s', Gamma, nu, z, rspdef)
				
				elif qy == 'MspM200m-1s':
					ret = modelDiemer17Scatter('MspM200m-1s', Gamma, nu, z, rspdef)
				
				elif qy == 'Deltasp-1s':
					rsp_1s = modelDiemer17Scatter('RspR200m-1s', Gamma, nu, z, rspdef)
					msp_1s = modelDiemer17Scatter('MspM200m-1s', Gamma, nu, z, rspdef)
					ret = np.sqrt(msp_1s**2 + 3.0 * rsp_1s**2)

				min_scatter = 0.02
				if utilities.isArray(ret):
					ret[ret < min_scatter] = min_scatter
				else:
					ret = max(ret, min_scatter)
			
	elif model == 'more15':

		if qy == 'RspR200m':
			if qx == 'Gamma':		
				ret = modelMore15RspR200m(z = z, Gamma = x)
			elif qx == 'nu200m':
				ret = modelMore15RspR200m(z = z, Gamma = None, nu200m = x)
		
		elif qy == 'MspM200m':
			if qx == 'Gamma':		
				ret = modelMore15MspM200m(z = z, Gamma = x)
			elif qx == 'nu200m':
				ret = modelMore15MspM200m(z = z, Gamma = None, nu200m = x)
				
		elif qy == 'Deltasp':
			if qx == 'Gamma':		
				msp200m = modelMore15MspM200m(z = z, Gamma = x)
				rsp200m = modelMore15RspR200m(z = z, Gamma = x)
			elif qx == 'nu200m':
				msp200m = modelMore15MspM200m(z = z, Gamma = None, nu200m = x)
				rsp200m = modelMore15RspR200m(z = z, Gamma = None, nu200m = x)
			ret = 200.0 * msp200m / rsp200m**3

	elif model == 'shi16':
		
		if qy == 'RspR200m':
			ret = modelShi16RspR200m(x, Om)
		elif qy == 'MspM200m':
			delta = modelShi16Delta(x, Om)
			rspr200m = modelShi16RspR200m(x, Om)
			ret = delta / 200.0 * rspr200m**3
		elif qy == 'Deltasp':
			ret = modelShi16Delta(x, Om)

	return ret, mask

###################################################################################################

def modelDiemer17PercentileValue(rspdef):
	
	if rspdef == 'mean':
		p = -1.0
	else:
		p = float(rspdef[-2:]) / 100.0

	return p

###################################################################################################

def modelDiemer17Gamma(nu, z):
	
	a0 = 1.222190
	a1 = 0.351460
	b0 = -0.286441
	b1 = 0.077767
	b2 = -0.056228
	b3 = 0.004100

	A = a0 + a1 * z
	B = b0 + b1 * z + b2 * z**2 + b3 * z**3
	Gamma = A * nu + B * nu**1.5
	
	return Gamma

###################################################################################################

def modelDiemer17RspR200m(qy, Gamma, nu, z, rspdef):

	if rspdef == 'mean':
		
		if qy == 'RspR200m':
			a0 = 0.649783
			b0 = 0.600362
			b_om = 0.091996
			b_nu = 0.061557
			c0 = -0.806288
			c_om = 17.520522
			c_nu = -0.293465
			c_om2 = -9.624342
			c_nu2 = 0.039196
			a0_p = 0.000000
			b0_p = 0.000000
			b_om_p = 0.000000
			b_om_p2 = 0.000000
			b_nu_p = 0.000000
			c_om_p = 0.000000
			c_om_p2 = 0.000000
			c_om2_p = 0.000000
			c_om2_p2 = 0.000000
			c_nu_p = 0.000000
			c_nu2_p = 0.000000
			
		elif qy == 'MspM200m':
			a0 = 0.679244
			b0 = 0.405083
			b_om = 0.291925
			b_nu = 0.000000
			c0 = 3.365943
			c_om = 1.469818
			c_nu = -0.075635
			c_om2 = 0.000000
			c_nu2 = 0.000000
			a0_p = 0.000000
			b0_p = 0.000000
			b_om_p = 0.000000
			b_om_p2 = 0.000000
			b_nu_p = 0.000000
			c_om_p = 0.000000
			c_om_p2 = 0.000000
			c_om2_p = 0.000000
			c_om2_p2 = 0.000000
			c_nu_p = 0.000000
			c_nu2_p = 0.000000
	else:
		
		if qy == 'RspR200m':
			a0 = 0.320332
			b0 = 0.267433
			b_om = 0.113389
			b_nu = 0.207989
			c0 = -0.959629
			c_om = 16.245894
			c_nu = 0.000000
			c_om2 = -9.497861
			c_nu2 = -0.018484
			a0_p = 0.614807
			b0_p = 0.545238
			b_om_p = 0.000000
			b_om_p2 = 0.000000
			b_nu_p = -0.223282
			c_om_p = 0.003941
			c_om_p2 = 8.969094
			c_om2_p = -0.000485
			c_om2_p2 = 10.613168
			c_nu_p = -0.451066
			c_nu2_p = 0.088029
			
		elif qy == 'MspM200m':
			a0 = 0.264765
			b0 = 0.666040
			b_om = 0.168814
			b_nu = 0.000000
			c0 = 4.728709
			c_om = 2.388866
			c_nu = -0.084108
			c_om2 = 0.000000
			c_nu2 = 0.000000
			a0_p = 0.843509
			b0_p = -0.639169
			b_om_p = 0.003195
			b_om_p2 = 4.939266
			b_nu_p = 0.225399
			c_om_p = -0.705712
			c_om_p2 = -1.241920
			c_om2_p = 0.000000
			c_om2_p2 = 0.000000
			c_nu_p = -0.391103
			c_nu2_p = 0.074216

	cosmo = cosmology.getCurrent()
	Om = cosmo.Om(z)
	p = modelDiemer17PercentileValue(rspdef)
		
	A0 = a0 + p * a0_p
	B0 = b0 + p * b0_p
	B_om = b_om + b_om_p * np.exp(p * b_om_p2)
	B_nu = b_nu + p * b_nu_p
	C0 = c0
	C_om = c_om + c_om_p * np.exp(p * c_om_p2)
	C_om2 = c_om2 + c_om2_p * np.exp(p * c_om2_p2)
	C_nu = c_nu + p * c_nu_p
	C_nu2 = c_nu2 + p * c_nu2_p
	
	A = A0
	B = (B0 + B_om * Om) * (1.0 + B_nu * nu)
	C = (C0 + C_om * Om + C_om2 * Om**2) * (1.0 + C_nu * nu + C_nu2 * nu**2)
	
	if utilities.isArray(C):
		C[C < 1E-4] = 1E-4
	
	ret = A + B * np.exp(-Gamma / C)
	
	return ret

###################################################################################################

def modelDiemer17Scatter(qy, Gamma, nu, z, rspdef):

	if rspdef == 'mean':
		if qy == 'RspR200m-1s':
			sigma_0 = 0.052645
			sigma_Gamma = 0.003846
			sigma_nu = -0.012054
			sigma_p = 0.000000
		elif qy == 'MspM200m-1s':
			sigma_0 = 0.052815
			sigma_Gamma = 0.002456
			sigma_nu = -0.011182
			sigma_p = 0.000000
	else:
		if qy == 'RspR200m-1s':
			sigma_0 = 0.044548
			sigma_Gamma = 0.004404
			sigma_nu = -0.014636
			sigma_p = 0.022637
		elif qy == 'MspM200m-1s':
			sigma_0 = 0.027594
			sigma_Gamma = 0.002330
			sigma_nu = -0.012491
			sigma_p = 0.047344

	p = modelDiemer17PercentileValue(rspdef)	
	ret = sigma_0 + sigma_Gamma * Gamma + sigma_nu * nu + sigma_p * p

	return ret

###################################################################################################

def modelShi16Delta(Gamma, Om):

	return 33.0 * Om**-0.45 * np.exp((0.88 - 0.14 * np.log(Om)) * Gamma**0.6)

###################################################################################################

def modelShi16RspR200m(Gamma, Om):

	return np.exp((0.24 + 0.074 * np.log(Gamma)) * np.log(Om) + 0.55 - 0.15 * Gamma)

###################################################################################################

def modelMore15RspR200m(nu200m = None, z = None, Gamma = None, statistic = 'median'):

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

def modelMore15MspM200m(nu200m = None, z = None, Gamma = None, statistic = 'median'):

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

# def Rsp(R, z, mdef, c = None, profile = defaults.HALO_MASS_CONVERSION_PROFILE):
# 	"""
# 	:math:`R_{sp}` as a function of spherical overdensity radius.
# 	
# 	Parameters
# 	-----------------------------------------------------------------------------------------------
# 	R: array_like
# 		Spherical overdensity radius in physical :math:`kpc/h`; can be a number or a numpy array.
# 	z: float
# 		Redshift
# 	mdef: str
# 		Mass definition in which R and c are given.
# 	c: array_like
# 		Halo concentration; must have the same dimensions as R, or be ``None`` in which case the 
# 		concentration is computed automatically.
# 	profile: str
# 		The functional form of the profile assumed in the conversion between mass definitions; 
# 		can be ``nfw`` or ``dk14``.
# 
# 	Returns
# 	-----------------------------------------------------------------------------------------------
# 	Rsp: array_like
# 		:math:`R_{sp}` in physical :math:`kpc/h`; has the same dimensions as R.
# 		
# 	See also
# 	-----------------------------------------------------------------------------------------------
# 	RspOverR200m: The ratio :math:`R_{sp} / R_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
# 	MspOverM200m: The ratio :math:`M_{sp} / M_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
# 	Msp: :math:`M_{sp}` as a function of spherical overdensity mass.
# 	"""
# 	
# 	if mdef == '200m':
# 		R200m = R
# 		M200m = mass_so.R_to_M(R200m, z, '200m')
# 	else:
# 		M = mass_so.R_to_M(R, z, mdef)
# 		if c is None:
# 			M200m, R200m, _ = mass_adv.changeMassDefinitionCModel(M, z, mdef, '200m', profile = profile)
# 		else:
# 			M200m, R200m, _ = mass_defs.changeMassDefinition(M, c, z, mdef, '200m', profile = profile)
# 			
# 	cosmo = cosmology.getCurrent()
# 	nu200m = cosmo.peakHeight(M200m, z)
# 	Rsp = R200m * RspOverR200m(nu200m = nu200m)
# 	
# 	return Rsp
# 
# ###################################################################################################
# 
# def Msp(M, z, mdef, c = None, profile = defaults.HALO_MASS_CONVERSION_PROFILE):
# 	"""
# 	:math:`M_{sp}` as a function of spherical overdensity mass.
# 	
# 	Parameters
# 	-----------------------------------------------------------------------------------------------
# 	M: array_like
# 		Spherical overdensity mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
# 	z: float
# 		Redshift
# 	mdef: str
# 		Mass definition in which M and c are given.
# 	c: array_like
# 		Halo concentration; must have the same dimensions as M, or be ``None`` in which case the 
# 		concentration is computed automatically.
# 	profile: str
# 		The functional form of the profile assumed in the conversion between mass definitions; 
# 		can be ``nfw`` or ``dk14``.
# 
# 	Returns
# 	-----------------------------------------------------------------------------------------------
# 	Msp: array_like
# 		:math:`M_{sp}` in :math:`M_{\odot}/h`; has the same dimensions as M.
# 		
# 	See also
# 	-----------------------------------------------------------------------------------------------
# 	RspOverR200m: The ratio :math:`R_{sp} / R_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
# 	MspOverM200m: The ratio :math:`M_{sp} / M_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
# 	Rsp: :math:`R_{sp}` as a function of spherical overdensity radius.
# 	"""
# 	
# 	if mdef == '200m':
# 		M200m = M
# 	else:
# 		if c is None:
# 			M200m, _, _ = mass_adv.changeMassDefinitionCModel(M, z, mdef, '200m', profile = profile)
# 		else:
# 			M200m, _, _ = mass_defs.changeMassDefinition(M, c, z, mdef, '200m', profile = profile)
# 	
# 	cosmo = cosmology.getCurrent()
# 	nu200m = cosmo.peakHeight(M200m, z)
# 	Msp = M200m * MspOverM200m(nu200m = nu200m)
# 	
# 	return Msp

###################################################################################################
