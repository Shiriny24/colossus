###################################################################################################
#
# power_spectrum.py         (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module implements models for the matter power spectrum, or more exactly, for the transfer 
function. Generally speaking, the transfer function should be evaluated using the 
:func:`cosmology.cosmology.Cosmology.matterPowerSpectrum` function.

---------------------------------------------------------------------------------------------------
Power spectrum models
---------------------------------------------------------------------------------------------------

The following models for the matter power spectrum are implemented in Colossus:

================== ======================================
ID                 Paper
================== ======================================
eisenstein98       `Eisenstein & Hu 1998 <http://adsabs.harvard.edu/abs/1998ApJ...496..605E>`_
eisenstein98_zb    `Eisenstein & Hu 1998 <http://adsabs.harvard.edu/abs/1998ApJ...496..605E>`_
================== ======================================

The _zb extension indicates the zero-baryon version of the Eisenstein & Hu 1998 model.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
from collections import OrderedDict

from colossus import defaults

###################################################################################################

class PowerSpectrumModel():
	"""
	This object contains certain characteristics of a power spectrum model. Currently, this object
	is empty, but the ``models`` variable is a dictionary of :class:`PowerSpectrumModel` objects 
	containing all available models.
	"""
		
	def __init__(self):
		return

###################################################################################################

models = OrderedDict()

models['eisenstein98'] = PowerSpectrumModel()
models['eisenstein98_zb'] = PowerSpectrumModel()

###################################################################################################

def transferFunction(k, h, Om0, Ob0, Tcmb0, model = defaults.POWER_SPECTRUM_MODEL):
	"""
	The transfer function.
	
	The transfer function transforms the spectrum of primordial fluctuations into the
	power spectrum of the initial matter density fluctuations. The primordial power spectrum is 
	usually described as a power law, leading to a power spectrum
	
	.. math::
		P(k) = T(k)^2 k^{n_s}
		
	where P(k) is the matter power spectrum, T(k) is the transfer function, and :math:`n_s` is 
	the tilt of the primordial power spectrum. See the :mod:`cosmology.cosmology` module for further 
	details on the cosmological parameters.

	Parameters
	-------------------------------------------------------------------------------------------
	k: array_like
		The wavenumber k (in comoving h/Mpc); can be a number or a numpy array.
	h: float
		The Hubble constant in units of 100 km/s/Mpc.
	Om0: float
		:math:`\Omega_{m}`, the matter density in units of the critical density at z = 0.
	Ob0: float
		:math:`\Omega_{b}`, the baryon density in units of the critical density at z = 0.
	Tcmb0: float
		The temperature of the CMB at z = 0 in Kelvin.

	Returns
	-------------------------------------------------------------------------------------------
	Tk: array_like
		The transfer function; has the same dimensions as k.
	"""
	
	if model == 'eisenstein98':
		T = modelEisenstein98(k, h, Om0, Ob0, Tcmb0)
	elif model == 'eisenstein98_zb':
		T = modelEisenstein98ZeroBaryon(k, h, Om0, Ob0, Tcmb0)
	else:
		raise Exception('Unknown model, %s.' % model)
	
	return T

###################################################################################################

def modelEisenstein98(k, h, Om0, Ob0, Tcmb0):
	"""
	The transfer function according to Eisenstein & Hu 1998.
	
	This function computes the Eisenstein & Hu 1998 approximation to the transfer function at a 
	scale k, and is based on Matt Becker's cosmocalc code.

	Parameters
	-------------------------------------------------------------------------------------------
	k: array_like
		The wavenumber k (in comoving h/Mpc); can be a number or a numpy array.
	h: float
		The Hubble constant in units of 100 km/s/Mpc.
	Om0: float
		:math:`\Omega_{m}`, the matter density in units of the critical density at z = 0.
	Ob0: float
		:math:`\Omega_{b}`, the baryon density in units of the critical density at z = 0.
	Tcmb0: float
		The temperature of the CMB at z = 0 in Kelvin.

	Returns
	-------------------------------------------------------------------------------------------
	Tk: array_like
		The transfer function; has the same dimensions as k.

	See also
	-------------------------------------------------------------------------------------------
	modelEisenstein98ZeroBaryon: The zero-baryon transfer function according to Eisenstein & Hu 1998.
	"""

	# Define shorter expressions
	omc = Om0 - Ob0
	ombom0 = Ob0 / Om0
	h2 = h**2
	om0h2 = Om0 * h2
	ombh2 = Ob0 * h2
	theta2p7 = Tcmb0 / 2.7
	theta2p72 = theta2p7**2
	theta2p74 = theta2p72**2
	
	# Convert kh from h/Mpc to 1/Mpc
	kh = k * h

	# Equation 2
	zeq = 2.50e4 * om0h2 / theta2p74

	# Equation 3
	keq = 7.46e-2 * om0h2 / theta2p72

	# Equation 4
	b1d = 0.313 * om0h2**-0.419 * (1.0 + 0.607 * om0h2**0.674)
	b2d = 0.238 * om0h2**0.223
	zd = 1291.0 * om0h2**0.251 / (1.0 + 0.659 * om0h2**0.828) * (1.0 + b1d * ombh2**b2d)

	# Equation 5
	Rd = 31.5 * ombh2 / theta2p74 / (zd / 1e3)
	Req = 31.5 * ombh2 / theta2p74 / (zeq / 1e3)

	# Equation 6
	s = 2.0 / 3.0 / keq * np.sqrt(6.0 / Req) * np.log((np.sqrt(1.0 + Rd) + \
		np.sqrt(Rd + Req)) / (1.0 + np.sqrt(Req)))

	# Equation 7
	ksilk = 1.6 * ombh2**0.52 * om0h2**0.73 * (1.0 + (10.4 * om0h2)**-0.95)

	# Equation 10
	q = kh / 13.41 / keq

	# Equation 11
	a1 = (46.9 * om0h2)**0.670 * (1.0 + (32.1 * om0h2)**-0.532)
	a2 = (12.0 * om0h2)**0.424 * (1.0 + (45.0 * om0h2)**-0.582)
	ac = a1**(-ombom0) * a2**(-ombom0**3)

	# Equation 12
	b1 = 0.944 / (1.0 + (458.0 * om0h2)**-0.708)
	b2 = (0.395 * om0h2)**-0.0266
	bc = 1.0 / (1.0 + b1 * ((omc / Om0)**b2 - 1.0))

	# Equation 15
	y = (1.0 + zeq) / (1.0 + zd)
	Gy = y * (-6.0 * np.sqrt(1.0 + y) + (2.0 + 3.0 * y) \
		* np.log((np.sqrt(1.0 + y) + 1.0) / (np.sqrt(1.0 + y) - 1.0)))

	# Equation 14
	ab = 2.07 * keq * s * (1.0 + Rd)**(-3.0 / 4.0) * Gy

	# Get CDM part of transfer function

	# Equation 18
	f = 1.0 / (1.0 + (kh * s / 5.4)**4)

	# Equation 20
	C = 14.2 / ac + 386.0 / (1.0 + 69.9 * q**1.08)

	# Equation 19
	T0t = np.log(np.e + 1.8 * bc * q) / (np.log(np.e + 1.8 * bc * q) + C * q * q)

	# Equation 17
	C1bc = 14.2 + 386.0 / (1.0 + 69.9 * q**1.08)
	T0t1bc = np.log(np.e + 1.8 * bc * q) / (np.log(np.e + 1.8 * bc * q) + C1bc * q * q)
	Tc = f * T0t1bc + (1.0 - f) * T0t

	# Get baryon part of transfer function

	# Equation 24
	bb = 0.5 + ombom0 + (3.0 - 2.0 * ombom0) * np.sqrt((17.2 * om0h2) * (17.2 * om0h2) + 1.0)

	# Equation 23
	bnode = 8.41 * om0h2**0.435

	# Equation 22
	st = s / (1.0 + (bnode / kh / s) * (bnode / kh / s) * (bnode / kh / s))**(1.0 / 3.0)

	# Equation 21
	C11 = 14.2 + 386.0 / (1.0 + 69.9 * q**1.08)
	T0t11 = np.log(np.e + 1.8 * q) / (np.log(np.e + 1.8 * q) + C11 * q * q)
	Tb = (T0t11 / (1.0 + (kh * s / 5.2)**2) + ab / (1.0 + (bb / kh / s)**3) * np.exp(-(kh / ksilk)**1.4)) \
		* np.sin(kh * st) / (kh * st)

	# Total transfer function
	Tk = ombom0 * Tb + omc / Om0 * Tc

	return Tk

###################################################################################################

def modelEisenstein98ZeroBaryon(k, h, Om0, Ob0, Tcmb0):
	"""
	The zero-baryon transfer function according to Eisenstein & Hu 1998.

	Parameters
	-------------------------------------------------------------------------------------------
	k: array_like
		The wavenumber k (in comoving h/Mpc); can be a number or a numpy array.
	h: float
		The Hubble constant in units of 100 km/s/Mpc.
	Om0: float
		:math:`\Omega_{m}`, the matter density in units of the critical density at z = 0.
	Ob0: float
		:math:`\Omega_{b}`, the baryon density in units of the critical density at z = 0.
	Tcmb0: float
		The temperature of the CMB at z = 0 in Kelvin.

	Returns
	-------------------------------------------------------------------------------------------
	Tk: array_like
		The transfer function; has the same dimensions as k.

	See also
	-------------------------------------------------------------------------------------------
	modelEisenstein98: The transfer function according to Eisenstein & Hu 1998.
	"""
	
	ombom0 = Ob0 / Om0
	h2 = h**2
	om0h2 = Om0 * h2
	ombh2 = Ob0 * h2
	theta2p7 = Tcmb0 / 2.7

	# Convert kh from hMpc^-1 to Mpc^-1
	kh = k * h

	# Equation 26
	s = 44.5 * np.log(9.83 / om0h2) / np.sqrt(1.0 + 10.0 * ombh2**0.75)

	# Equation 31
	alphaGamma = 1.0 - 0.328 * np.log(431.0 * om0h2) * ombom0 + 0.38 * np.log(22.3 * om0h2) * ombom0**2

	# Equation 30
	Gamma = Om0 * h * (alphaGamma + (1.0 - alphaGamma) / (1.0 + (0.43 * kh * s)**4))

	# Equation 28
	q = k * theta2p7 * theta2p7 / Gamma

	# Equation 29
	C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
	L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
	Tk = L0 / (L0 + C0 * q * q)

	return Tk

###################################################################################################
