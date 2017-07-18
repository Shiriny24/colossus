###################################################################################################
#
# lss.py                    (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module contains relatively general functions, such as translating halo mass into peak height.

---------------------------------------------------------------------------------------------------
Basic usage
---------------------------------------------------------------------------------------------------

The peak height of halos quantifies how likely they are to have collapsed by comparing their mass
to that of a typically collapsed structure, quantified by the variance of the linear density field.
The peak height of a halo is simple to evaluate in Colossus::
	
	setCosmology('planck15')
	nu = lss.peakHeight(M, z)

The inverse function converts peak height to mass. The non-linear mass is defined as the mass at
which peak height is unity at a given redshift, i.e., the mass of a halo that is typically 
collapsing at the current time.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
import scipy.integrate
import scipy.special

from colossus.utils import constants
from colossus.cosmology import cosmology

###################################################################################################

def lagrangianR(M):
	"""
	The lagrangian radius of a halo of mass M.

	Converts the mass of a halo (in comoving :math:`M_{\odot} / h`) to the radius of its 
	comoving Lagrangian volume (in comoving Mpc/h), that is the volume that encloses the halo's 
	mass given the mean density of the universe at z = 0.

	Parameters
	-------------------------------------------------------------------------------------------
	M: array_like
		Halo mass in :math:`M_{\odot} / h`; can be a number or a numpy array.

	Returns
	-------------------------------------------------------------------------------------------
	R: array_like
		The lagrangian radius in comoving Mpc/h; has the same dimensions as M.

	See also
	-------------------------------------------------------------------------------------------
	lagrangianM: The lagrangian mass of a halo of radius R.
	"""
	
	cosmo = cosmology.getCurrent()
	R = (3.0 * M / 4.0 / np.pi / cosmo.rho_m(0.0) / 1E9)**(1.0 / 3.0)
	
	return R

###################################################################################################

def lagrangianM(R):
	"""
	The lagrangian mass of a halo of radius R.

	Converts the radius of a halo (in comoving Mpc/h) to the mass in its comoving Lagrangian 
	volume (in :math:`M_{\odot} / h`), that is the volume that encloses the halo's mass given the 
	mean density of the universe at z = 0.

	Parameters
	-------------------------------------------------------------------------------------------
	R: array_like
		Halo radius in comoving Mpc/h; can be a number or a numpy array.

	Returns
	-------------------------------------------------------------------------------------------
	M: array_like
		The lagrangian mass; has the same dimensions as R.

	See also
	-------------------------------------------------------------------------------------------
	lagrangianR: The lagrangian radius of a halo of mass M.
	"""
	
	cosmo = cosmology.getCurrent()
	M = 4.0 / 3.0 * np.pi * R**3 * cosmo.rho_m(0.0) * 1E9
	
	return M

###################################################################################################

def collapseOverdensity(deltac_const = True, sigma = None):
	"""
	The threshold overdensity for halo collapse.
	
	For most applications, ``deltac_const = True`` works fine; in that case, this function
	simply returns the collapse overdensity predicted by the top-hat collapse model, 1.686. 
	Alternatively, a correction for the ellipticity of peaks can be applied according to Sheth 
	et al. 2001. In that case, the variance on the scale of a halo must also be passed.

	Parameters
	-------------------------------------------------------------------------------------------
	deltac_const: bool
		If True, the function returns the constant top-hat model collapse overdensity. If False,
		a correction due to the ellipticity of halos is applied.
	sigma: float
		The rms variance on the scale of the halo; only necessary if ``deltac_const == False``.

	Returns
	-------------------------------------------------------------------------------------------
	delta_c: float
		The threshold overdensity for collapse.
	"""
			
	if deltac_const:
		delta_c = constants.DELTA_COLLAPSE
	else:
		delta_c = constants.DELTA_COLLAPSE * (1.0 + 0.47 * (sigma / constants.DELTA_COLLAPSE)**1.23)
	
	return delta_c

###################################################################################################

def peakHeight(M, z, filt = 'tophat', Pk_source = 'eh98', deltac_const = True):
	"""
	Peak height, :math:`\\nu`, given a halo mass.
	
	Peak height is defined as :math:`\\nu \equiv \delta_c / \sigma(M)`. See the documentation 
	of the :func:`cosmology.cosmology.Cosmology.sigma` function for details on filters, power 
	spectrum source etc.
	
	Parameters
	-------------------------------------------------------------------------------------------
	M: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift.
	filt: str
		Either ``tophat`` or ``gaussian``.
	Pk_source: str
		Either ``eh98``, ``eh98smooth``, or the name of a user-supplied table.
	deltac_const: bool
		If True, the function returns the constant top-hat model collapse overdensity. If False,
		a correction due to the ellipticity of halos is applied.

	Returns
	-------------------------------------------------------------------------------------------
	nu: array_like
		Peak height; has the same dimensions as M.

	See also
	-------------------------------------------------------------------------------------------
	massFromPeakHeight: Halo mass from peak height, :math:`\\nu`.
	"""
			
	cosmo = cosmology.getCurrent()
	R = lagrangianR(M)
	sigma = cosmo.sigma(R, z, filt = filt, Pk_source = Pk_source)
	nu = collapseOverdensity(deltac_const, sigma) / sigma

	return nu

###################################################################################################

def massFromPeakHeight(nu, z, filt = 'tophat', Pk_source = 'eh98', deltac_const = True):
	"""
	Halo mass from peak height, :math:`\\nu`.
	
	Peak height is defined as :math:`\\nu \equiv \delta_c / \sigma(M)`. See the documentation 
	of the :func:`cosmology.cosmology.Cosmology.sigma` function for details on filters, power 
	spectrum source etc.
	
	Parameters
	-------------------------------------------------------------------------------------------
	nu: array_like
		Peak height; can be a number or a numpy array.
	z: float
		Redshift.
	filt: str
		Either ``tophat`` or ``gaussian``.
	Pk_source: str
		Either ``eh98``, ``eh98smooth``, or the name of a user-supplied table.
	deltac_const: bool
		If True, the function returns the constant top-hat model collapse overdensity. If False,
		a correction due to the ellipticity of halos is applied.

	Returns
	-------------------------------------------------------------------------------------------
	M: array_like
		Mass in :math:`M_{\odot}/h`; has the same dimensions as nu.

	See also
	-------------------------------------------------------------------------------------------
	peakHeight: Peak height, :math:`\\nu`, given a halo mass.
	"""
	
	cosmo = cosmology.getCurrent()
	sigma = collapseOverdensity(deltac_const = deltac_const) / nu
	R = cosmo.sigma(sigma, z, filt = filt, Pk_source = Pk_source, inverse = True)
	M = lagrangianM(R)
	
	return M

###################################################################################################

def nonLinearMass(z, filt = 'tophat', Pk_source = 'eh98'):
	"""
	The non-linear mass, :math:`M^*`.
	
	:math:`M^*` is the mass for which the variance is equal to the collapse threshold, i.e.
	:math:`\sigma(M^*) = \delta_c` and thus :math:`\\nu(M^*) = 1`. See the documentation 
	of the :func:`cosmology.cosmology.Cosmology.sigma` function for details on filters, power 
	spectrum source etc.
	
	Parameters
	-------------------------------------------------------------------------------------------
	z: float
		Redshift.
	filt: str
		Either ``tophat`` or ``gaussian``.
	Pk_source: str
		Either ``eh98``, ``eh98smooth``, or the name of a user-supplied table.

	Returns
	-------------------------------------------------------------------------------------------
	Mstar: float
		The non-linear mass in :math:`M_{\odot}/h`.

	See also
	-------------------------------------------------------------------------------------------
	massFromPeakHeight: Halo mass from peak height, :math:`\\nu`.
	"""
	
	return massFromPeakHeight(1.0, z = z, filt = filt, Pk_source = Pk_source, deltac_const = True)

###################################################################################################
# Peak curvature routines
###################################################################################################

# Get the mean peak curvature, <x>, at fixed nu from the integral of Bardeen et al. 1986 
# (BBKS). Note that this function is approximated very well by the _peakCurvatureApprox() 
# function below.

def _peakCurvatureExact(nu, gamma):

	# Equation A15 in BBKS. 
	
	def curvature_fx(x):

		f1 = np.sqrt(5.0 / 2.0) * x
		t1 = scipy.special.erf(f1) + scipy.special.erf(f1 / 2.0)

		b0 = np.sqrt(2.0 / 5.0 / np.pi)
		b1 = 31.0 * x ** 2 / 4.0 + 8.0 / 5.0
		b2 = x ** 2 / 2.0 - 8.0 / 5.0
		t2 = b0 * (b1 * np.exp(-5.0 * x ** 2 / 8.0) + b2 * np.exp(-5.0 * x ** 2 / 2.0))

		res = (x ** 3 - 3.0 * x) * t1 / 2.0 + t2

		return res

	# Equation A14 in BBKS, minus the normalization which is irrelevant here. If we need the 
	# normalization, the Rstar parameter also needs to be passed.
	
	def curvature_Npk(x, nu, gamma):

		#norm = np.exp(-nu**2 / 2.0) / (2 * np.pi)**2 / Rstar**3
		norm = 1.0
		fx = curvature_fx(x)
		xstar = gamma * nu
		g2 = 1.0 - gamma ** 2
		exponent = -(x - xstar) ** 2 / (2.0 * g2)
		res = norm * fx * np.exp(exponent) / np.sqrt(2.0 * np.pi * g2)

		return res

	# Average over Npk
	
	def curvature_Npk_x(x, nu, gamma):
		return curvature_Npk(x, nu, gamma) * x

	args = nu, gamma
	norm, _ = scipy.integrate.quad(curvature_Npk, 0.0, np.infty, args, epsrel = 1E-10)
	integ, _ = scipy.integrate.quad(curvature_Npk_x, 0.0, np.infty, args, epsrel = 1E-10)
	xav = integ / norm

	return xav

###################################################################################################

# Wrapper for the function above which takes tables of sigmas. This form can be more convenient 
# when computing many different nu's. 

def _peakCurvatureExactFromSigma(sigma0, sigma1, sigma2, deltac_const = True):

	nu = collapseOverdensity(deltac_const, sigma0) / sigma0
	gamma = sigma1 ** 2 / sigma0 / sigma2

	x = nu * 0.0
	for i in range(len(nu)):
		x[i] = _peakCurvatureExact(nu[i], gamma[i])

	return nu, gamma, x

###################################################################################################

# Get peak curvature from the approximate formula in BBKS. This approx. is excellent over the 
# relevant range of nu.

def _peakCurvatureApprox(nu, gamma):

	# Compute theta according to Equation 6.14 in BBKS
	g = gamma
	gn = g * nu
	theta1 = 3.0 * (1.0 - g ** 2) + (1.216 - 0.9 * g ** 4) * np.exp(-g * gn * gn / 8.0)
	theta2 = np.sqrt(3.0 * (1.0 - g ** 2) + 0.45 + (gn / 2.0) ** 2) + gn / 2.0
	theta = theta1 / theta2

	# Equation 6.13 in BBKS
	x = gn + theta
	
	# Equation 6.15 in BBKS
	nu_tilde = nu - theta * g / (1.0 - g ** 2)

	return theta, x, nu_tilde

###################################################################################################

# Wrapper for the function above which takes tables of sigmas. This form can be more convenient 
# when computing many different nu's. For convenience, various intermediate numbers are 
# returned as well.

def _peakCurvatureApproxFromSigma(sigma0, sigma1, sigma2, deltac_const = True):

	nu = collapseOverdensity(deltac_const, sigma0) / sigma0
	gamma = sigma1**2 / sigma0 / sigma2
	
	theta, x, nu_tilde = _peakCurvatureApprox(nu, gamma)
	
	return nu, gamma, x, theta, nu_tilde

###############################################################################################

def peakCurvature(M, z, filt = 'gaussian', Pk_source = 'eh98',
				deltac_const = True, exact = False):
	"""
	The average curvature of peaks for a halo mass M.
	
	In a Gaussian random field, :math:`\delta`, the peak height is defined as 
	:math:`\delta / \\sigma` where :math:`\\sigma = \\sigma_0` is the rms variance. The 
	curvature of the field is defined as :math:`x = -\\nabla^2 \delta / \\sigma_2` where 
	:math:`\\sigma_2` is the second moment of the variance.
	
	This function computes the average curvature of peaks in a Gaussian random field, <x>,
	according to Bardeen et al. 1986 (BBKS), for halos of a certain mass M. This mass is 
	converted to a Lagrangian scale R, and thus the variance and its moments. The evaluation
	can be performed by integration of Equation A14 in BBKS (if ``exact == True``), or using 
	their fitting function in Equation 6.13 (if ``exact == False``). The fitting function is 
	excellent over the relevant range of peak heights. 
	
	Parameters
	-------------------------------------------------------------------------------------------
	M: array_like
		Mass in in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift.
	filt: str
		Either ``tophat`` or ``gaussian``.
	Pk_source: str
		Either ``eh98``, ``eh98smooth``, or the name of a user-supplied table.
	deltac_const: bool
		If ``True``, the function returns the constant top-hat model collapse overdensity. If 
		``False``, a correction due to the ellipticity of halos is applied.
	exact: bool
		If ``True``, evaluate the integral exactly; if ``False``, use the BBKS approximation.	

	Returns
	-------------------------------------------------------------------------------------------
	nu: array_like
		Peak height; has the same dimensions as M.
	gamma: array_like
		An intermediate parameter, :math:`\\gamma = \\sigma_1^2 / (\\sigma_0 \\sigma_2)` (see
		Equation 4.6a in BBKS); has the same dimensions as M.
	x: array_like
		The mean peak curvature for halos of mass M (note the caveat discussed above); has the 
		same dimensions as M.
	theta: array_like
		An intermediate parameter (see Equation 6.14 in BBKS; only returned if ``exact == False``); 
		has the same dimensions as M.
	nu_tilde: array_like
		The modified peak height (see Equation 6.15 in BBKS; only returned if ``exact == False``); 
		has the same dimensions as M.
	
	Warnings
	-------------------------------------------------------------------------------------------		
	While peak height quantifies how high a fluctuation over the background a halo is, peak
	curvature tells us something about the shape of the initial peak. However, note the 
	cloud-in-cloud problem (BBKS): not all peaks end up forming halos, particularly small
	peaks will often get swallowed by other peaks. Thus, the average peak curvature is not
	necessarily equal to the average curvature of peaks that form halos.		
	"""

	cosmo = cosmology.getCurrent()

	R = lagrangianR(M)
	sigma0 = cosmo.sigma(R, z, j = 0, filt = filt, Pk_source = Pk_source)
	sigma1 = cosmo.sigma(R, z, j = 1, filt = filt, Pk_source = Pk_source)
	sigma2 = cosmo.sigma(R, z, j = 2, filt = filt, Pk_source = Pk_source)

	if exact:
		return _peakCurvatureExactFromSigma(sigma0, sigma1, sigma2, deltac_const = deltac_const)
	else:
		return _peakCurvatureApproxFromSigma(sigma0, sigma1, sigma2, deltac_const = deltac_const)

###################################################################################################
