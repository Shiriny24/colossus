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

We create a density profile object which has a range of functions. For example, let us create an 
NFW profile::
	
	profile = NFWProfile(M = 1E12, mdef = 'vir', z = 0.0, c = 10.0)
	Rvir = profile.RDelta(0.0, 'vir')
	rho = profile.density(Rvir)

See the documentation of the abstract base class :class:`profile_base.HaloDensityProfile` for the functionality 
of the profile objects. For documentation on spherical overdensity mass definitions, please see the 
documentation of the :mod:`halo.basics` module. The following functional forms for the density 
profile are implemented:

============================ =============================== ========================== =============
Class                        Explanation                     Paper                      Reference
============================ =============================== ========================== =============
:func:`SplineDensityProfile` A arbitrary density profile     ---                        ---
:func:`EinastoProfile`       Einasto profile                 Einasto 1965               TrAlm 5, 87
:func:`NFWProfile`           Navarro-Frenk-White profile     Navarro et al. 1997        ApJ 490, 493
:func:`DK14Profile`          Diemer & Kravtsov 2014 profile  Diemer & Kravtsov 2014     ApJ 789, 1
============================ =============================== ========================== =============

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
Profile fitting
---------------------------------------------------------------------------------------------------

Here, fitting refers to finding the parameters of a halo density profile which best describe a
given set of data points. Each point corresponds to a radius and a particular quantity, such as 
density, enclosed mass, or surface density. Optionally, the user can pass uncertainties on the 
data points, or even a full covariance matrix. All fitting should be done using the very general 
:func:`profile_base.HaloDensityProfile.fit` routine. For example, let us fit an NFW profile to some density 
data::

	profile = NFWProfile(M = 1E12, mdef = 'vir', z = 0.0, c = 10.0)
	profile.fit(r, rho, 'rho')
	
Here, r and rho are arrays of radii and densities. Note that the current parameters of the profile 
instance are used as an initial guess for the fit, and the profile object is set to the best-fit 
parameters after the fit. Under the hood, the fit function handles multiple different fitting 
methods. By default, the above fit is performed using a least-squares minimization, but we can also 
use an MCMC sampler, for example to fit the surface density profile::

	dict = profile.fit(r, Sigma, 'Sigma', method = 'mcmc', q_cov = covariance_matrix)
	best_fit_params = dict['x_mean']
	uncertainty = dict['percentiles'][0]
	
The :func:`profile_base.HaloDensityProfile.fit` function accepts many input options, some specific to the 
fitting method used. Please see the detailed documentation below.

---------------------------------------------------------------------------------------------------
Alternative mass definitions
---------------------------------------------------------------------------------------------------

Two alternative mass definitions (as in, not spherical overdensity masses) are described in 
More, Diemer & Kravtsov 2015. Those include:

* :math:`M_{sp}`: The mass contained within the radius of the outermost density caustic. 
  Caustics correspond to particles piling up at the apocenter of their orbits. The most pronounced
  caustic is due to the most recently accreted matter, and that caustic is also found at the
  largest radius which we call the splashback radius, :math:`R_{sp}`. This is designed as a 
  physically meaningful radius definition that encloses all the mass ever accreted by a halo.
* :math:`M_{<4r_s}`: The mass within 4 scale radii. This mass definition quantifies the mass in
  the inner part of the halo. During the fast accretion regime, this mass definition tracks
  :math:`M_{vir}`, but when the halo stops accreting it approaches a constant. 

:math:`M_{<4r_s}`: can be computed from both NFW and DK14 profiles. :math:`R_{sp}` and 
:math:`M_{sp}` can only be computed from DK14 profiles. Please see the :mod:`halo.mass_definitions`
module for convenient converter functions between mass definitions.

---------------------------------------------------------------------------------------------------
Units
---------------------------------------------------------------------------------------------------

Unless otherwise noted, all functions in this module use the following units:

================ =======================================
Variable         Unit
================ =======================================
Length           Physical kpc/h
Mass             :math:`M_{\odot}/h`
Density          Physical :math:`M_{\odot} h^2 / kpc^3`
Surface density  Physical :math:`M_{\odot} h / kpc^2`
================ =======================================

---------------------------------------------------------------------------------------------------
Module Reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.interpolate
import scipy.special

from colossus.utils import utilities
from colossus.utils import defaults
from colossus.cosmology import cosmology
from colossus.halo import basics
from colossus.halo import profile_base

###################################################################################################
# SPLINE DEFINED PROFILE
###################################################################################################

class SplineDensityProfile(profile_base.HaloDensityProfile):
	"""
	An arbitrary density profile using spline interpolation.
	
	This class takes an arbitrary array of radii and densities or enclosed masses as input, and 
	interpolates them using a splines (in log space). Note that there are three different ways of 
	specifying the density profile:
	
	* density and mass: Both density and mass are interpolated using splines.
	* density only: In order for the enclosed mass to be defined, the density must be specified 
	  all the way to r = 0. In that case, the mass is computed numerically, stored, and interpolated.
	* mass only: The density is computed as the derivative of the mass, stored, and interpolated.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	r: numpy array
		Radii in physical kpc/h.
	rho: array_like
		Density at radii r in physical :math:`M_{\odot} h^2 / kpc^3`. Does not have to be passed
		as long as M is passed.
	M: array_like
		Enclosed mass within radii r in :math:`M_{\odot} / h`. Does not have to be passed
		as long as rho is passed.

	Warnings
	-----------------------------------------------------------------------------------------------
	If both mass and density are supplied to the constructor, the consistency between the two is 
	not checked! 
	"""
	
	###############################################################################################
	# CONSTRUCTOR
	###############################################################################################
	
	def __init__(self, r, rho = None, M = None):
		
		self.par_names = []
		self.opt_names = []
		profile_base.HaloDensityProfile.__init__(self)
		
		self.rmin = np.min(r)
		self.rmax = np.max(r)
		self.r_guess = np.sqrt(self.rmin * self.rmax)
		self.min_RDelta = self.rmin
		self.max_RDelta = self.rmax

		if rho is None and M is None:
			msg = 'Either mass or density must be specified.'
			raise Exception(msg)
		
		self.rho_spline = None
		self.M_spline = None
		logr = np.log(r)
		
		if M is not None:
			logM = np.log(M)
			self.M_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logM)

		if rho is not None:
			logrho = np.log(rho)
			self.rho_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logrho)

		# Construct M(r) from density. For some reason, the spline integrator fails on the 
		# innermost bin, and the quad integrator fails on the outermost bin. 
		if self.M_spline is None:
			integrand = 4.0 * np.pi * r**2 * rho
			integrand_spline = scipy.interpolate.InterpolatedUnivariateSpline(r, integrand)
			logM = 0.0 * r
			for i in range(len(logM) - 1):
				logM[i], _ = scipy.integrate.quad(integrand_spline, 0.0, r[i])
			logM[-1] = integrand_spline.integral(0.0, r[-1])
			logM = np.log(logM)
			self.M_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logM)

		if self.rho_spline is None:
			deriv = self.M_spline(np.log(r), nu = 1) * M / r
			logrho = np.log(deriv / 4.0 / np.pi / r**2)
			self.rho_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logrho)

		return

	###############################################################################################
	# METHODS BOUND TO THE CLASS
	###############################################################################################

	def density(self, r):
		
		return np.exp(self.rho_spline(np.log(r)))

	###############################################################################################
	
	def densityDerivativeLin(self, r):

		log_deriv = self.rho_spline(np.log(r), nu = 1)
		deriv = log_deriv * self.density(r) / r
		
		return deriv

	###############################################################################################

	def densityDerivativeLog(self, r):
	
		return self.rho_spline(np.log(r), nu = 1)
	
	###############################################################################################

	def enclosedMass(self, r):

		return np.exp(self.M_spline(np.log(r)))

###################################################################################################
# NFW PROFILE
###################################################################################################

class NFWProfile(profile_base.HaloDensityProfile):
	"""
	The Navarro-Frenk-White profile.
	
	The NFW profile is defined by the density function
	
	.. math::
		\\rho(r) = \\frac{\\rho_s}{\\left(\\frac{r}{r_s}\\right) \\left(1 + \\frac{r}{r_s}\\right)^{2}}
		
	The constructor accepts either the free parameters in this formula, central density and scale 
	radius, or a spherical overdensity mass and concentration (in this case the mass definition 
	and redshift also need to be specified). The density and other commonly used routines are 
	implemented both as class and as static routines, meaning they can be called without 
	instantiating the class.

	Parameters
	-----------------------------------------------------------------------------------------------
	rhos: float
		The central density in physical :math:`M_{\odot} h^2 / kpc^3`.
	rs: float
		The scale radius in physical kpc/h.
	M: float
		A spherical overdensity mass in :math:`M_{\odot}/h` corresponding to the mass
		definition mdef at redshift z. 
	c: float
		The concentration, :math:`c = R / r_s`, corresponding to the given halo mass and mass 
		definition.
	z: float
		Redshift
	mdef: str
		The mass definition in which M and c are given.
	"""
	
	###############################################################################################
	# CONSTANTS
	###############################################################################################

	# See the xDelta function for the meaning of these constants
	XDELTA_GUESS_FACTORS = [5.0, 10.0, 20.0, 100.0, 10000.0]
	XDELTA_N_GUESS_FACTORS = len(XDELTA_GUESS_FACTORS)

	###############################################################################################
	# CONSTRUCTOR
	###############################################################################################

	def __init__(self, rhos = None, rs = None,
				M = None, c = None, z = None, mdef = None):
		
		self.par_names = ['rhos', 'rs']
		self.opt_names = []
		profile_base.HaloDensityProfile.__init__(self)

		# The fundamental way to define an NFW profile by the central density and scale radius
		if rhos is not None and rs is not None:
			self.par['rhos'] = rhos
			self.par['rs'] = rs

		# Alternatively, the user can give a mass and concentration, together with mass definition
		# and redshift.
		elif M is not None and c is not None and mdef is not None and z is not None:
			self.par['rhos'], self.par['rs'] = self.fundamentalParameters(M, c, z, mdef)
		
		else:
			msg = 'An NFW profile must be define either using rhos and rs, or M, c, mdef, and z.'
			raise Exception(msg)
		
		return

	###############################################################################################
	# STATIC METHODS
	###############################################################################################

	@classmethod
	def fundamentalParameters(cls, M, c, z, mdef):
		"""
		The fundamental NFW parameters, :math:`\\rho_s` and :math:`r_s`, from mass and 
		concentration.
		
		This routine is called in the constructor of the NFW profile class (unless :math:`\\rho_s` 
		and :math:`r_s` are passed by the user), but can also be called without instantiating an 
		NFWProfile object.
	
		Parameters
		-------------------------------------------------------------------------------------------
		M: array_like
			Spherical overdensity mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
		c: array_like
			The concentration, :math:`c = R / r_s`, corresponding to the given halo mass and mass 
			definition; must have the same dimensions as M.
		z: float
			Redshift
		mdef: str
			The mass definition in which M and c are given.
			
		Returns
		-------------------------------------------------------------------------------------------
		rhos: array_like
			The central density in physical :math:`M_{\odot} h^2 / kpc^3`; has the same dimensions
			as M.
		rs: array_like
			The scale radius in physical kpc/h; has the same dimensions as M.
		"""
				
		rs = basics.M_to_R(M, z, mdef) / c
		rhos = M / rs**3 / 4.0 / np.pi / cls.mu(c)
		
		return rhos, rs

	###############################################################################################

	@staticmethod
	def rho(rhos, x):
		"""
		The NFW density as a function of :math:`x=r/r_s`.
		
		This routine can be called without instantiating an NFWProfile object. In most cases, the 
		:func:`profile_base.HaloDensityProfile.density` function should be used instead.

		Parameters
		-------------------------------------------------------------------------------------------
		rhos: float
			The central density in physical :math:`M_{\odot} h^2 / kpc^3`.
		x: array_like
			The radius in units of the scale radius, :math:`x=r/r_s`; can be a number or a numpy
			array.
		
		Returns
		-------------------------------------------------------------------------------------------
		rho: array_like
			Density in physical :math:`M_{\odot} h^2 / kpc^3`; has the same dimensions as x.

		See also
		-------------------------------------------------------------------------------------------
		profile_base.HaloDensityProfile.density: Density as a function of radius.
		"""
		
		return rhos / x / (1.0 + x)**2
	
	###############################################################################################
	
	@staticmethod
	def mu(x):
		"""
		A function of :math:`x=r/r_s` that appears in the NFW enclosed mass.

		This routine can be called without instantiating an NFWProfile object.

		Parameters
		-------------------------------------------------------------------------------------------
		x: array_like
			The radius in units of the scale radius, :math:`x=r/r_s`; can be a number or a numpy
			array.
		
		Returns
		-------------------------------------------------------------------------------------------
		mu: array_like
			Has the same dimensions as x.

		See also
		-------------------------------------------------------------------------------------------
		M: The enclosed mass in an NFW profile as a function of :math:`x=r/r_s`.
		profile_base.HaloDensityProfile.enclosedMass: The mass enclosed within radius r.
		"""
		
		return np.log(1.0 + x) - x / (1.0 + x)
	
	###############################################################################################

	@classmethod
	def M(cls, rhos, rs, x):
		"""
		The enclosed mass in an NFW profile as a function of :math:`x=r/r_s`.

		This routine can be called without instantiating an NFWProfile object. In most cases, the 
		:func:`profile_base.HaloDensityProfile.enclosedMass` function should be used instead.

		Parameters
		-------------------------------------------------------------------------------------------
		rhos: float
			The central density in physical :math:`M_{\odot} h^2 / kpc^3`.
		rs: float
			The scale radius in physical kpc/h.
		x: array_like
			The radius in units of the scale radius, :math:`x=r/r_s`; can be a number or a numpy
			array.
		
		Returns
		-------------------------------------------------------------------------------------------
		M: array_like
			The enclosed mass in :math:`M_{\odot}/h`; has the same dimensions as x.

		See also
		-------------------------------------------------------------------------------------------
		mu: A function of :math:`x=r/r_s` that appears in the NFW enclosed mass.
		profile_base.HaloDensityProfile.enclosedMass: The mass enclosed within radius r.
		"""
		
		return 4.0 * np.pi * rs**3 * rhos * cls.mu(x)

	###############################################################################################

	@classmethod
	def _thresholdEquationX(cls, x, rhos, density_threshold):
		
		return rhos * cls.mu(x) * 3.0 / x**3 - density_threshold

	###############################################################################################
	
	@classmethod
	def xDelta(cls, rhos, rs, density_threshold, x_guess = 5.0):
		"""
		Find :math:`x=r/r_s` where the enclosed density has a particular value.
		
		This function is the basis for the :func:`RDelta` routine, but can 
		be used without instantiating an NFWProfile object. This is preferable when the function 
		needs to be evaluated many times, for example when converting a large number of mass 
		definitions.
		
		Parameters
		-------------------------------------------------------------------------------------------
		rhos: float
			The central density in physical :math:`M_{\odot} h^2 / kpc^3`.
		rs: float
			The scale radius in physical kpc/h.
		density_threshold: float
			The desired enclosed density threshold in physical :math:`M_{\odot} h^2 / kpc^3`. This 
			number can be generated from a mass definition and redshift using the 
			:func:`halo.basics.densityThreshold` function. 
		
		Returns
		-------------------------------------------------------------------------------------------
		x: float
			The radius in units of the scale radius, :math:`x=r/r_s`, where the enclosed density
			reaches ``density_threshold``. 

		See also
		-------------------------------------------------------------------------------------------
		RDelta: The spherical overdensity radius of a given mass definition.
		"""
		
		# A priori, we have no idea at what radius the result will come out, but we need to 
		# provide lower and upper limits for the root finder. To balance stability and performance,
		# we do so iteratively: if there is no result within relatively aggressive limits, we 
		# try again with more conservative limits.
		args = rhos, density_threshold
		x = None
		i = 0
		while x is None and i < cls.XDELTA_N_GUESS_FACTORS:
			try:
				xmin = x_guess / cls.XDELTA_GUESS_FACTORS[i]
				xmax = x_guess * cls.XDELTA_GUESS_FACTORS[i]
				x = scipy.optimize.brentq(cls._thresholdEquationX, xmin, xmax, args)
			except Exception:
				i += 1
		
		if x is None:
			msg = 'Could not determine x where the density threshold is satisfied.'
			raise Exception(msg)
		
		return x
		
	###############################################################################################
	# METHODS BOUND TO THE CLASS
	###############################################################################################
	
	def density(self, r):
	
		x = r / self.par['rs']
		density = self.rho(self.par['rhos'], x)
		
		return density

	###############################################################################################

	def densityDerivativeLin(self, r):

		x = r / self.par['rs']
		density_der = -self.par['rhos'] / self.par['rs'] * (1.0 / x**2 / (1.0 + x)**2 + 2.0 / x / (1.0 + x)**3)

		return density_der
	
	###############################################################################################

	def densityDerivativeLog(self, r):

		x = r / self.par['rs']
		density_der = -(1.0 + 2.0 * x / (1.0 + x))

		return density_der

	###############################################################################################

	def enclosedMass(self, r):
		
		x = r / self.par['rs']
		mass = self.M(self.par['rhos'], self.par['rs'], x)
		
		return mass
	
	###############################################################################################
	
	# The surface density of an NFW profile can be computed analytically which is much faster than
	# integration. The formula below is taken from Bartelmann (1996). The case r = rs is solved in 
	# Lokas & Mamon (2001), but in their notation the density at this radius looks somewhat 
	# complicated. In the notation used here, Sigma(rs) = 2/3 * rhos * rs.
	
	def surfaceDensity(self, r):
	
		xx = r / self.par['rs']
		x, is_array = utilities.getArray(xx)
		surfaceDensity = np.ones_like(x) * self.par['rhos'] * self.par['rs']
		
		# Solve separately for r < rs, r > rs, r = rs
		mask_rs = abs(x - 1.0) < 1E-4
		mask_lt = (x < 1.0) & (np.logical_not(mask_rs))
		mask_gt = (x > 1.0) & (np.logical_not(mask_rs))
		
		surfaceDensity[mask_rs] *= 2.0 / 3.0

		xi = x[mask_lt]		
		x2 = xi**2
		x2m1 = x2 - 1.0
		surfaceDensity[mask_lt] *= 2.0 / x2m1 \
			* (1.0 - 2.0 / np.sqrt(-x2m1) * np.arctanh(np.sqrt((1.0 - xi) / (xi + 1.0))))

		xi = x[mask_gt]		
		x2 = xi**2
		x2m1 = x2 - 1.0
		surfaceDensity[mask_gt] *= 2.0 / x2m1 \
			* (1.0 - 2.0 / np.sqrt(x2m1) * np.arctan(np.sqrt((xi - 1.0) / (xi + 1.0))))
			
		if not is_array:
			surfaceDensity = surfaceDensity[0]
	
		return surfaceDensity

	###############################################################################################
	
	# For the NFW profile, rmax is a constant multiple of the scale radius since vc is maximized
	# where ln(1+x) = (2x**2 + x) / (1+x)**2
	
	def Vmax(self):
		
		rmax = 2.16258 * self.par['rs']
		vmax = self.circularVelocity(rmax)
		
		return vmax, rmax

	###############################################################################################
	
	# This equation is 0 when the enclosed density matches the given density_threshold. This 
	# function matches the abstract interface in HaloDensityProfile, but for the NFW profile it is
	# easier to solve the equation in x (see the _thresholdEquationX() function).
		
	def _thresholdEquation(self, r, density_threshold):
		
		return self._thresholdEquationX(r / self.par['rs'], self.par['rhos'], density_threshold)

	###############################################################################################

	# Return the spherical overdensity radius (in kpc / h) for a given mass definition and redshift. 
	# This function is overwritten for the NFW profile as we have a better guess at the resulting
	# radius, namely the scale radius. Thus, the user can specify a minimum and maximum concentra-
	# tion that is considered.

	def RDelta(self, z, mdef):
	
		density_threshold = basics.densityThreshold(z, mdef)
		x = self.xDelta(self.par['rhos'], self.par['rs'], density_threshold)
		R = x * self.par['rs']
		
		return R

	###############################################################################################

	def M4rs(self):
		"""
		The mass within 4 scale radii, :math:`M_{<4rs}`.
		
		See the section on mass definitions for details.

		Returns
		-------------------------------------------------------------------------------------------
		M4rs: float
			The mass within 4 scale radii, :math:`M_{<4rs}`, in :math:`M_{\odot} / h`.
		"""
		
		M = self.enclosedMass(4.0 * self.par['rs'])
		
		return M
	
	###############################################################################################

	# When fitting the NFW profile, use log(rho_c) and log(rs); since both parameters are 
	# converted in the same way, we don't have to worry about the mask

	def _fitConvertParams(self, p, mask):
		
		return np.log(p)

	###############################################################################################
	
	def _fitConvertParamsBack(self, p, mask):
		
		return np.exp(p)

	###############################################################################################

	# Return and array of d rho / d ln(rhos) and d rho / d ln(rs)
	
	def _fitParamDeriv_rho(self, r, mask, N_par_fit):

		x = self.getParameterArray()
		deriv = np.zeros((N_par_fit, len(r)), np.float)
		rrs = r / x[1]
		rho_r = x[0] / rrs / (1.0 + rrs) ** 2

		counter = 0
		if mask[0]:
			deriv[counter] = rho_r
			counter += 1
		if mask[1]:
			deriv[counter] = rho_r * rrs * (1.0 / rrs + 2.0 / (1.0 + rrs))
			counter += 1
			
		return deriv

###################################################################################################
# EINASTO PROFILE
###################################################################################################

class EinastoProfile(profile_base.HaloDensityProfile):
	"""
	The Einasto 1965 density profile.

	The Einasto profile is defined by the density function
	
	.. math::
		\\rho(r) = \\rho_s \\exp \\left( -\\frac{2}{\\alpha} \\left[ \\left( \\frac{r}{r_s} \\right)^{\\alpha} - 1 \\right] \\right)

	or, alternatively, by a logarithmic slope that evolves with radius as 
	
	.. math::
		\\frac{d \\log(\\rho)}{d \\log(r)} = -2 \\left( \\frac{r}{r_s} \\right)^{\\alpha}
	
	The constructor accepts either the free parameters (the density at the scale radius, the scale 
	radius, and alpha), or a spherical overdensity mass and concentration (in this case the mass 
	definition and redshift also need to be specified). In the latter case, the user can specify 
	alpha or let the constructor compute it from its tight correlation with peak height 
	(Gao et al. 2008). In the latter case, a cosmology must be set before instantiating the 
	EinastoProfile object.

	Parameters
	-----------------------------------------------------------------------------------------------
	rhos: float
		The density at the scale radius in physical :math:`M_{\odot} h^2 / kpc^3`.
	rs: float
		The scale radius in physical kpc/h.
	alpha: float
		The radial dependence of the profile slope.
	M: float
		A spherical overdensity mass in :math:`M_{\odot}/h` corresponding to the mass
		definition mdef at redshift z. 
	c: float
		The concentration, :math:`c = R / r_s`, corresponding to the given halo mass and mass 
		definition.
	z: float
		Redshift
	mdef: str
		The mass definition in which M and c are given.		
	"""

	###############################################################################################
	# CONSTRUCTOR
	###############################################################################################

	def __init__(self, rhos = None, rs = None, alpha = None,
				M = None, c = None, z = None, mdef = None):
	
		self.par_names = ['rhos', 'rs', 'alpha']
		self.opt_names = []
		profile_base.HaloDensityProfile.__init__(self)

		# The fundamental way to define an Einasto profile by the density at the scale radius, 
		# the scale radius, and alpha.
		if rhos is not None and rs is not None and alpha is not None:
			self.par['rhos'] = rhos
			self.par['rs'] = rs
			self.par['alpha'] = alpha
			
		# Alternatively, the user can give a mass and concentration, together with mass definition
		# and redshift. Passing alpha is now optional since it can also be estimated from the
		# Gao et al. 2008 relation between alpha and peak height. This relation was calibrated for
		# nu_vir, so if the given mass definition is not 'vir' we convert the given mass to Mvir
		# assuming an NFW profile with the given mass and concentration. This leads to a negligible
		# inconsistency, but solving for the correct alpha iteratively would be much slower.
		elif M is not None and c is not None and mdef is not None and z is not None:
			self.fundamentalParameters(M, c, z, mdef, alpha)
					
		else:
			msg = 'An Einasto profile must be define either using rhos, rs, and alpha, or M, c, mdef, and z.'
			raise Exception(msg)

		# We need an initial radius to guess Rmax
		self.r_guess = self.par['rs']
		
		# Pre-compute the mass terms now that the parameters have been fixed
		self._setMassTerms()

		return
	
	###############################################################################################
	# METHODS BOUND TO THE CLASS
	###############################################################################################

	def fundamentalParameters(self, M, c, z, mdef, alpha = None):
		"""
		The fundamental Einasto parameters, :math:`\\rho_s`, :math:`r_s`, and :math:`\\alpha` from 
		mass and concentration.
		
		This routine is called in the constructor of the Einasto profile class (unless 
		:math:`\\rho_s`, :math:`r_s` and :math:`\\alpha` are passed by the user), and cannot be
		called without instantiating an EinastoProfile object.
	
		Parameters
		-------------------------------------------------------------------------------------------
		M: float
			Spherical overdensity mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
		c: float
			The concentration, :math:`c = R / r_s`, corresponding to the given halo mass and mass 
			definition; must have the same dimensions as M.
		z: float
			Redshift
		mdef: str
			The mass definition in which M and c are given.
		alpha: float
			The radial dependence of the profile slope; can be None in which case it is 
			approximated.
			
		Returns
		-------------------------------------------------------------------------------------------
		rhos: float
			The density at the scale radius in physical :math:`M_{\odot} h^2 / kpc^3`.
		rs: float
			The scale radius in physical kpc/h.
		alpha: float
			The radial dependence of the profile slope.
		"""

		R = basics.M_to_R(M, z, mdef)
		self.par['rs'] = R / c
		
		if alpha is None:
			if mdef == 'vir':
				Mvir = M
			else:
				Mvir, _, _ = changeMassDefinition(M, c, z, mdef, 'vir')
			cosmo = cosmology.getCurrent()
			nu_vir = cosmo.peakHeight(Mvir, z)
			alpha = 0.155 + 0.0095 * nu_vir**2
		
		self.par['alpha'] = alpha
		self.par['rhos'] = 1.0
		self._setMassTerms()
		M_unnorm = self.enclosedMass(R)
		self.par['rhos'] = M / M_unnorm
		
		return
	
	###############################################################################################

	# The enclosed mass for the Einasto profile is semi-analytical, in that it can be expressed
	# in terms of Gamma functions. We pre-compute some factors to speed up the computation 
	# later.
	
	def _setMassTerms(self):

		self.mass_norm = np.pi * self.par['rhos'] * self.par['rs']**3 * 2.0**(2.0 - 3.0 / self.par['alpha']) \
			* self.par['alpha']**(-1.0 + 3.0 / self.par['alpha']) * np.exp(2.0 / self.par['alpha']) 
		self.gamma_3alpha = scipy.special.gamma(3.0 / self.par['alpha'])
		
		return
	
	###############################################################################################

	# We need to overwrite the setParameterArray function because the mass terms need to be 
	# updated when the user changes the parameters.
	
	def setParameterArray(self, pars, mask = None):
		
		profile_base.HaloDensityProfile.setParameterArray(self, pars, mask = mask)
		self._setMassTerms()
		
		return

	###############################################################################################

	def density(self, r):
		
		rho = self.par['rhos'] * np.exp(-2.0 / self.par['alpha'] * \
										((r / self.par['rs'])**self.par['alpha'] - 1.0))
		
		return rho

	###############################################################################################
	
	def densityDerivativeLin(self, r):

		rho = self.density(r)
		drho_dr = rho * (-2.0 / self.par['rs']) * (r / self.par['rs'])**(self.par['alpha'] - 1.0)	
		
		return drho_dr

	###############################################################################################
	
	def densityDerivativeLog(self, r):

		der = -2.0 * (r / self.par['rs'])**self.par['alpha']
		
		return der

	###############################################################################################

	def enclosedMass(self, r):
		
		mass = self.mass_norm * self.gamma_3alpha * scipy.special.gammainc(3.0 / self.par['alpha'],
								2.0 / self.par['alpha'] * (r / self.par['rs'])**self.par['alpha'])
		
		return mass
	
	###############################################################################################

	# When fitting the Einasto profile, use log(rhos), log(rs) and log(alpha)

	def _fitConvertParams(self, p, mask):
		
		return np.log(p)

	###############################################################################################
	
	def _fitConvertParamsBack(self, p, mask):
		
		return np.exp(p)

	###############################################################################################

	# Return and array of d rho / d ln(rhos) and d rho / d ln(rs)
	
	def _fitParamDeriv_rho(self, r, mask, N_par_fit):

		x = self.getParameterArray()
		deriv = np.zeros((N_par_fit, len(r)), np.float)
		rrs = r / x[1]
		rho_r = self.density(r)
		
		counter = 0
		if mask[0]:
			deriv[counter] = rho_r
			counter += 1
		if mask[1]:
			deriv[counter] = 2.0 * rho_r * rrs**(x[2])
			counter += 1
		if mask[2]:
			deriv[counter] = rho_r * 2.0 / x[2] * rrs**x[2] * (1.0 - rrs**(-x[2]) - x[2] * np.log(rrs))
			counter += 1

		return deriv
	
###################################################################################################
# DIEMER & KRAVTSOV 2014 PROFILE
###################################################################################################

class DK14Profile(profile_base.HaloDensityProfileWithOuter):
	"""
	The Diemer & Kravtsov 2014 density profile.
	
	This profile corresponds to an Einasto profile at small radii, and steepens around the virial 
	radius. At large radii, the profile approaches a power-law in r. The profile formula has 8
	free parameters, but most of those are fixed to particular values that depend on the mass and
	mass accretion rate of a halo. This can be done automatically, the user only needs to pass the 
	mass of a halo, and optionally concentration. However, there are some further options.
	
	======= ================ ===================================================================================
	Param.  Symbol           Explanation	
	======= ================ ===================================================================================
	R200m	:math:`R_{200m}` The radius that encloses and average overdensity of 200 :math:`\\rho_m(z)`
	rhos	:math:`\\rho_s`   The central scale density, in physical :math:`M_{\odot} h^2 / kpc^3`
	rs      :math:`r_s`      The scale radius in physical kpc/h
	rt      :math:`r_t`      The radius where the profile steepens, in physical kpc/h
	alpha   :math:`\\alpha`   Determines how quickly the slope of the inner Einasto profile steepens
	beta    :math:`\\beta`    Sharpness of the steepening
	gamma	:math:`\\gamma`   Asymptotic negative slope of the steepening term
	be      :math:`b_e`      Normalization of the power-law outer profile
	se      :math:`s_e`      Slope of the power-law outer profile
	parts   ---              'inner', 'outer', or 'both' (default)
	======= ================ ===================================================================================
		
	The profile has two parts, the 1-halo term (``inner``) and the 2-halo term (``outer``). By default, 
	the function returns their sum (``both``).
	
	The profile was calibrated for the median and mean profiles of two types of halo samples, 
	namely samples selected by mass, and samples selected by both mass and mass accretion rate. 
	When a new profile object is created, the user can choose between those by setting 
	``selected = 'by_mass'`` or ``selected = 'by_accretion_rate'``. The latter option results 
	in a more accurate representation of the density profile, but the mass accretion rate must be 
	known. 
	
	Furthermore, the parameters for the power-law outer profile (be and se) exhibit a complicated 
	dependence on halo mass, redshift and cosmology. At the moment, they are not automatically 
	determined and must be set by the user if ``part == both``, i.e. if the outer profile is to
	be included which is recommended if the profile is to be reliable beyond the virial radius. 
	At low redshift, and for the cosmology considered in our paper, ``be = 1.0`` and ``se = 1.5`` 
	are good values over a wide range of masses (see Figure 18 in Diemer & Kravtsov 2014). 
	
	The parameter values, and their dependence on mass etc, are explained in Section 3.3 of
	Diemer & Kravtsov 2014.
	
	Get the native DK14 parameters given a halo mass, and possibly concentration.
	
	Get the DK14 parameters that correspond to a profile with a particular mass M in some mass
	definition mdef. Optionally, the user can define the concentration c; otherwise, it is 
	computed automatically. 
	
	Parameters
	-----------------------------------------------------------------------------------------------
	M: float
		Halo mass in :math:`M_{\odot}/h`.
	c: float
		Concentration in the same mass definition as M.
	z: float
		Redshift
	mdef: str
		The mass definition to which M corresponds.
	selected_by: str
		The halo sample to which this profile refers can be selected ``by_mass`` or 
		``by_accretion_rate``. This parameter influences how some of the fixed parameters in the 
		profile are set, in particular those that describe the steepening term.
	Gamma: float
		The mass accretion rate as defined in DK14. This parameter only needs to be passed if 
		``selected == by_accretion_rate``.
	part: str
		Can be ``both`` or ``inner``. This parameter is simply passed into the return structure. The 
		value ``outer`` makes no sense in this function, since the outer profile alone cannot be 
		normalized to have the mass M.
	be: float
		Normalization of the power-law outer profile. Only needs to be passed if ``part == both``.
		The best-fit be and se parameters depend on redshift, halo mass, cosmology etc,
		and there is no convenient formula to describe their values. At low redshift, ``be = 1.0``
		and ``se = 1.5`` are a good assumption (see Figure 18 in Diemer & Kravtsov 2014). 
	se: float
		Slope of the power-law outer profile (see parameter ``be`` above). Only needs to be 
		passed if ``part == both``.
	acc_warn: float
		If the function achieves a relative accuracy in matching M less than this value, a warning 
		is printed.
	acc_err: float
		If the function achieves a relative accuracy in matching MDelta less than this value, an 
		exception is raised.
	"""
	
	###############################################################################################
	# CONSTANTS
	###############################################################################################

	# This number determines the maximum overdensity that can be contributed by a power-law outer
	# profile. See the density function for details.
	max_outer_prof = 0.001

	###############################################################################################
	# CONSTRUCTOR
	###############################################################################################
	
	def __init__(self, rhos = None, rs = None, rt = None, alpha = None, beta = None, gamma = None, 
				R200m = None,
				M = None, c = None, z = None, mdef = None,
				selected_by = 'M', Gamma = None, 
				outer_terms = ['mean', 'pl'], 
				be = defaults.HALO_PROFILE_DK14_BE, se = defaults.HALO_PROFILE_DK14_SE,
				acc_warn = 0.01, acc_err = 0.05):
	
		# Set the fundamental variables par_names and opt_names
		self.par_names = ['rhos', 'rs', 'rt', 'alpha', 'beta', 'gamma']
		self.opt_names = ['selected_by', 'Gamma', 'R200m']
		#self.fit_log_mask = np.array([True, True, True, True, True, True])
		self.fit_log_mask = np.array([False, False, False, False, False, False])

		# Set outer terms
		self.outer_terms = []
		for i in range(len(outer_terms)):
			if outer_terms[i] == 'mean':
				if z is None:
					raise Exception('Redshift z must be set if a mean density outer term is chosen.')
				t = profile_base.OuterTermRhoMean(z)
			elif outer_terms[i] == 'pl':
				pl_max_rho = 1000.0
				t = profile_base.OuterTermPowerLaw(be, se, 'R200m', 5.0, pl_max_rho, z, norm_name = 'be', slope_name = 'se')
			self.outer_terms.append(t)
		
		# Run the constructor
		profile_base.HaloDensityProfileWithOuter.__init__(self)
		
		# The following parameters are not constants, they are temporarily changed by certain 
		# functions.
		self.accuracy_mass = 1E-4
		self.accuracy_radius = 1E-4

		self.opt['selected_by'] = selected_by
		self.opt['Gamma'] = Gamma
		self.opt['R200m'] = R200m
				
		if rhos is not None and rs is not None and rt is not None and alpha is not None \
			and beta is not None and gamma is not None and be is not None and se is not None \
			and R200m is not None:
			self.par['rhos'] = rhos
			self.par['rs'] = rs
			self.par['rt'] = rt
			self.par['alpha'] = alpha
			self.par['beta'] = beta
			self.par['gamma'] = gamma
		else:
			self.fundamentalParameters(M, c, z, mdef, selected_by, Gamma = Gamma,
									acc_warn = acc_warn, acc_err = acc_err)

		# We need to guess a radius when computing vmax
		self.r_guess = self.par['rs']

		return

	###############################################################################################
	# STATIC METHODS
	###############################################################################################

	# This function returns various calibrations of rt / R200m. Depending on selected, the chosen
	# beta and gamma are different, and thus rt is rather different. 
	#
	# If selected ==  by_nu, we use Equation 6 in DK14. Though this relation was originally 
	# calibrated for nu = nu_vir, the difference is small (<5%). 
	#
	# If selected == by_accretion_rate, there are multiple ways to calibrate the relation: from 
	# Gamma and z directly, or for the nu-selected samples but fitted like the accretion 
	# rate-selected samples (i.e., with beta = 6 and gamma = 4).

	@staticmethod
	def rtOverR200m(selected_by, nu200m = None, z = None, Gamma = None):
		
		if selected_by == 'M':
			ratio = 1.9 - 0.18 * nu200m
		
		elif selected_by == 'Gamma':
			if (Gamma is not None) and (z is not None):
				cosmo = cosmology.getCurrent()
				ratio =  0.43 * (1.0 + 0.92 * cosmo.Om(z)) * (1.0 + 2.18 * np.exp(-Gamma / 1.91))
			elif nu200m is not None:
				ratio = 0.79 * (1.0 + 1.63 * np.exp(-nu200m / 1.56))
			else:
				msg = 'Need either Gamma and z, or nu.'
				raise Exception(msg)
		
		else:
			msg = "Unknown sample selection, %s." % (selected_by)
			raise Exception(msg)

		return ratio

	###############################################################################################
	# METHODS BOUND TO THE CLASS
	###############################################################################################

	def fundamentalParameters(self, M, c, z, mdef, selected_by, Gamma = None, 
							acc_warn = 0.01, acc_err = 0.05):

		# Declare shared variables; these parameters are advanced during the iterations
		par2 = {}
		par2['RDelta'] = 0.0
		
		RTOL = 0.01
		MTOL = 0.01
		GUESS_TOL = 2.5
		
		self.accuracy_mass = MTOL
		self.accuracy_radius = RTOL
	
		# -----------------------------------------------------------------------------------------

		# Try a radius R200m, compute the resulting RDelta using the old RDelta as a starting guess
		
		def radius_diff(R200m, par2, Gamma, rho_target, R_target):
			
			self.opt['R200m'] = R200m
			M200m = basics.R_to_M(R200m, z, '200m')
			nu200m = cosmo.peakHeight(M200m, z)

			self.par['alpha'], self.par['beta'], self.par['gamma'], rt_R200m = \
				self.getFixedParameters(selected_by, nu200m = nu200m, z = z, Gamma = Gamma)
			self.par['rt'] = rt_R200m * R200m
			self.par['rhos'] *= self.normalizeInner(R200m, M200m)

			par2['RDelta'] = self._RDeltaLowlevel(par2['RDelta'], rho_target, guess_tolerance = GUESS_TOL)
			
			return par2['RDelta'] - R_target
		
		# -----------------------------------------------------------------------------------------
		
		# The user needs to set a cosmology before this function can be called
		cosmo = cosmology.getCurrent()
		R_target = basics.M_to_R(M, z, mdef)
		self.par['rs'] = R_target / c
		
		if mdef == '200m':
			
			# The user has supplied M200m, the parameters follow directly from the input
			M200m = M
			self.opt['R200m'] = basics.M_to_R(M200m, z, '200m')
			nu200m = cosmo.peakHeight(M200m, z)
			self.par['alpha'], self.par['beta'], self.par['gamma'], rt_R200m = \
				self.getFixedParameters(selected_by, nu200m = nu200m, z = z, Gamma = Gamma)
			self.par['rt'] = rt_R200m * self.opt['R200m']

			# Guess rhos = 1.0, then re-normalize			
			self.par['rhos'] = 1.0
			self.par['rhos'] *= self.normalizeInner(self.opt['R200m'], M200m)
			
		else:
			
			# The user has supplied some other mass definition, we need to iterate.
			_, R200m_guess, _ = changeMassDefinition(M, c, z, mdef, '200m')
			par2['RDelta'] = R_target
			self.par['rhos'] = 1.0

			# Iterate to find an M200m for which the desired mass is correct
			rho_target = basics.densityThreshold(z, mdef)
			args = par2, Gamma, rho_target, R_target
			self.opt['R200m'] = scipy.optimize.brentq(radius_diff, R200m_guess / 1.3, R200m_guess * 1.3,
								args = args, xtol = RTOL)

			# Check the accuracy of the result; M should be very close to MDelta now
			M_result = basics.R_to_M(par2['RDelta'], z, mdef)
			err = (M_result - M) / M
			
			if abs(err) > acc_warn:
				msg = 'WARNING: DK14 profile parameters converged to an accuracy of %.1f percent.' % (abs(err) * 100.0)
				print(msg)
			
			if abs(err) > acc_err:
				msg = 'DK14 profile parameters not converged (%.1f percent error).' % (abs(err) * 100.0)
				raise Exception(msg)
		
		return

	###############################################################################################

	# Get the parameter values for the DK14 profile that should be fixed, or can be determined from the 
	# peak height or mass accretion rate. If selected is 'by_mass', only nu must be passed. If selected 
	# is 'by_accretion_rate', then both z and Gamma must be passed.
	
	def getFixedParameters(self, selected_by, nu200m = None, z = None, Gamma = None):
	
		if selected_by == 'M':
			beta = 4.0
			gamma = 8.0
			rt_R200m = self.rtOverR200m('M', nu200m = nu200m)
		elif selected_by == 'Gamma':
			beta = 6.0
			gamma = 4.0
			rt_R200m = self.rtOverR200m('Gamma', z = z, Gamma = Gamma)
		else:
			msg = "Unknown sample selection, %s." % (selected_by)
			raise Exception(msg)
		
		# Gao et al. 2008 relation between alpha and nu. This function was originally calibrated for 
		# nu = nu_vir, but the difference is very small.
		alpha = 0.155 + 0.0095 * nu200m**2

		return alpha, beta, gamma, rt_R200m

	###############################################################################################
	
	def densityInner(self, r):
		
		inner = self.par['rhos'] * np.exp(-2.0 / self.par['alpha'] * ((r / self.par['rs'])**self.par['alpha'] - 1.0))
		fT = (1.0 + (r / self.par['rt'])**self.par['beta'])**(-self.par['gamma'] / self.par['gamma'])
		rho_1h = inner * fT

		return rho_1h

	###############################################################################################
	
	def densityDerivativeLinInner(self, r):
		
		drho_dr = r * 0.0
		
		rhos = self.par['rhos']
		rs = self.par['rs']
		rt = self.par['rt']
		alpha = self.par['alpha']
		beta = self.par['beta']
		gamma = self.par['gamma']
		
		inner = rhos * np.exp(-2.0 / alpha * ((r / rs) ** alpha - 1.0))
		d_inner = inner * (-2.0 / rs) * (r / rs)**(alpha - 1.0)	
		fT = (1.0 + (r / rt) ** beta) ** (-gamma / beta)
		d_fT = (-gamma / beta) * (1.0 + (r / rt) ** beta) ** (-gamma / beta - 1.0) * \
			beta / rt * (r / rt) ** (beta - 1.0)
		drho_dr += inner * d_fT + d_inner * fT
		
		return drho_dr

	###############################################################################################
	
	def densityDerivativeLog(self, r):
		
		drho_dr = self.densityDerivativeLin(r)
		rho = self.density(r)
		der = drho_dr * r / rho
		
		return der

	###############################################################################################
	
	# The surface density of the DK14 profile is a little tricky, since the profile approaches 
	# rho_m at large radii. Integrating to infinity would then give infinity. Instead, we subtract
	# the mean density if the outer profile is active. We don't need to check that r < rmax, since 
	# rmax is infinity for the DK14 profile.
	
	def surfaceDensity(self, r, accuracy = 1E-6):

		if np.max(r) >= self.rmax:
			msg = 'Cannot compute surface density at a radius (%.2e) greater than rmax (%.2e).' \
				% (np.max(r), self.rmax)
			raise Exception(msg)
		
		if self.opt['part'] in ['outer', 'both'] and self.opt['outer'] in ['mean', 'pl+mean']:
			subtract = self.par['rho_m']
		else:
			subtract = 0.0

		def integrand(r, R2):
			ret = r * (self.density(r) - subtract) / np.sqrt(r**2 - R2)
			return ret

		r_use, is_array = utilities.getArray(r)
		surfaceDensity = 0.0 * r_use
		for i in range(len(r_use)):	
			surfaceDensity[i], _ = scipy.integrate.quad(integrand, r_use[i], self.rmax, 
										args = (r_use[i]**2), epsrel = accuracy, limit = 1000)
			surfaceDensity[i] *= 2.0
			
		if not is_array:
			surfaceDensity = surfaceDensity[0]

		return surfaceDensity
	
	###############################################################################################

	# Low-level function to compute a spherical overdensity radius given the parameters of a DK14 
	# profile, the desired overdensity threshold, and an initial guess. A more user-friendly version
	# can be found above (DK14_getMR).
	
	def _RDeltaLowlevel(self, R_guess, density_threshold, guess_tolerance = 5.0):
			
		R = scipy.optimize.brentq(self._thresholdEquation, R_guess / guess_tolerance,
				R_guess * guess_tolerance, args = density_threshold, xtol = self.accuracy_radius)
		
		return R
	
	###############################################################################################

	# This function returns the spherical overdensity radius (in kpc / h) given a mass definition
	# and redshift. We know R200m and thus M200m for a DK14 profile, and use those parameters to
	# compute what R would be for an NFW profile and use this radius as an initial guess.
	
	def RDelta(self, z, mdef):
	
		M200m = basics.R_to_M(self.par['R200m'], z, mdef)
		_, R_guess, _ = changeMassDefinition(M200m, self.par['R200m'] / self.par['rs'], z, '200m', mdef)
		density_threshold = basics.densityThreshold(z, mdef)
		R = self._RDeltaLowlevel(R_guess, density_threshold)
	
		return R

	###############################################################################################

	def M4rs(self):
		"""
		The mass within 4 scale radii, :math:`M_{<4rs}`.
		
		See the section on mass definitions for details.

		Returns
		-------------------------------------------------------------------------------------------
		M4rs: float
			The mass within 4 scale radii, :math:`M_{<4rs}`, in :math:`M_{\odot} / h`.
		"""
		
		M = self.enclosedMass(4.0 * self.par['rs'])
		
		return M

	###############################################################################################

	def Rsp(self, search_range = 5.0):
		"""
		The splashback radius, :math:`R_{sp}`.
		
		See the section on mass definitions for details. Operationally, we define :math:`R_{sp}`
		as the radius where the profile reaches its steepest logarithmic slope.
		
		Parameters
		-------------------------------------------------------------------------------------------
		search_range: float
			When searching for the radius of steepest slope, search within this factor of 
			:math:`R_{200m}` (optional).
			
		Returns
		-------------------------------------------------------------------------------------------
		Rsp: float
			The splashback radius, :math:`R_{sp}`, in physical kpc/h.
			
		See also
		-------------------------------------------------------------------------------------------
		RMsp: The splashback radius and mass within, :math:`R_{sp}` and :math:`M_{sp}`.
		Msp: The mass enclosed within :math:`R_{sp}`, :math:`M_{sp}`.
		"""
		
		R200m = self.par['R200m']
		rc = scipy.optimize.fminbound(self.densityDerivativeLog, R200m / search_range, R200m * search_range)

		return rc
	
	###############################################################################################

	def RMsp(self, search_range = 5.0):
		"""
		The splashback radius and mass within, :math:`R_{sp}` and :math:`M_{sp}`.
		
		See the section on mass definitions for details.		
		
		Parameters
		-------------------------------------------------------------------------------------------
		search_range: float
			When searching for the radius of steepest slope, search within this factor of 
			:math:`R_{200m}` (optional).
			
		Returns
		-------------------------------------------------------------------------------------------
		Rsp: float
			The splashback radius, :math:`R_{sp}`, in physical kpc/h.
		Msp: float
			The mass enclosed within the splashback radius, :math:`M_{sp}`, in :math:`M_{\odot} / h`.
			
		See also
		-------------------------------------------------------------------------------------------
		Rsp: The splashback radius, :math:`R_{sp}`.
		Msp: The mass enclosed within :math:`R_{sp}`, :math:`M_{sp}`.
		"""
		
		Rsp = self.Rsp(search_range = search_range)
		Msp = self.enclosedMass(Rsp)

		return Rsp, Msp
	
	###############################################################################################

	def Msp(self, search_range = 5.0):
		"""
		The mass enclosed within :math:`R_{sp}`, :math:`M_{sp}`.
		
		See the section on mass definitions for details.		
		
		Parameters
		-------------------------------------------------------------------------------------------
		search_range: float
			When searching for the radius of steepest slope, search within this factor of 
			:math:`R_{200m}` (optional).
			
		Returns
		-------------------------------------------------------------------------------------------
		Msp: float
			The mass enclosed within the splashback radius, :math:`M_{sp}`, in :math:`M_{\odot} / h`.
			
		See also
		-------------------------------------------------------------------------------------------
		Rsp: The splashback radius, :math:`R_{sp}`.
		RMsp: The splashback radius and mass within, :math:`R_{sp}` and :math:`M_{sp}`.
		"""
		
		_, Msp = self.RMsp(search_range = search_range)

		return Msp
	
	###############################################################################################

	# When fitting the DK14 profile, use a mixture of linear and logarithmic parameters

	def _fitConvertParams(self, p, mask):

		p_fit = p
		log_mask = [self.fit_log_mask[mask]]
		p_fit[log_mask] = np.log(p_fit[log_mask])
		
		return p_fit

	###############################################################################################
	
	def _fitConvertParamsBack(self, p, mask):
		
		p_def = p.copy()
		log_mask = [self.fit_log_mask[mask]]
		p_def[log_mask] = np.exp(p_def[log_mask])
		
		return p_def

	###############################################################################################
	
	def _fitParamDeriv_rho(self, r, mask, N_par_fit):

		x = self.getParameterArray()
		deriv = np.zeros((N_par_fit, len(r)), np.float)
		
		temp_part = self.opt['part']
		self.opt['part'] = 'inner'
		rho_inner = self.density(r)
		self.opt['part'] = temp_part
		
		counter = 0
		
		rhos = x[0]
		rs = x[1]
		rt = x[2]
		alpha = x[3]
		beta = x[4]
		gamma = x[5]
		be = x[6]
		se = x[7]

		ro = 5.0 * x[8]
		rrs = r / rs
		rrt = r / rt
		rro = r / ro
		term1 = 1.0 + rrt**beta
		#rho_outer = x[9] * be * rro**-se
		rho_outer = x[9] * be / (self.max_outer_prof + (r / ro)**se)
		
		# rho_s
		if mask[0]:
			deriv[counter] = rho_inner / rhos
			counter += 1
		# rs
		if mask[1]:
			deriv[counter] = rho_inner / rs * rrs**alpha * 2.0
			counter += 1
		# rt
		if mask[2]:
			deriv[counter] = rho_inner * gamma / rt / term1 * rrt**beta
			counter += 1
		# alpha
		if mask[3]:
			deriv[counter] = rho_inner * 2.0 / alpha**2 * rrs**alpha * (1.0 - rrs**(-alpha) - alpha * np.log(rrs))
			counter += 1
		# beta
		if mask[4]:
			deriv[counter] = rho_inner * (gamma * np.log(term1) / beta**2 - gamma * \
										rrt**beta * np.log(rrt) / beta / term1)
			counter += 1
		# gamma
		if mask[5]:
			deriv[counter] = -rho_inner * np.log(term1) / beta
			counter += 1
		# be
		if mask[6]:
			deriv[counter] = rho_outer / be
			counter += 1
		# se
		if mask[7]:
			deriv[counter] = -rho_outer * np.log(rro)
			counter += 1

		# Correct for log parameters
		counter = 0
		for i in range(self.N_par):
			if self.fit_log_mask[i] and mask[i]:
				deriv[counter] *= x[i]
			if mask[i]:
				counter += 1
		
		return deriv
	
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
	changeMassDefinition: Change the spherical overdensity mass definition.
	"""
	
	M_i, is_array = utilities.getArray(M_i)
	c_i, _ = utilities.getArray(c_i)
	N = len(M_i)
	Rnew = np.zeros_like(M_i)
	cnew = np.zeros_like(M_i)

	if profile == 'nfw':
		
		# We do not instantiate NFW profile objects, but instead use the faster static functions
		rhos, rs = NFWProfile.fundamentalParameters(M_i, c_i, z_i, mdef_i)
		density_threshold = basics.densityThreshold(z_f, mdef_f)
		for i in range(N):
			cnew[i] = NFWProfile.xDelta(rhos[i], rs[i], density_threshold, x_guess = c_i[i])
		Rnew = rs * cnew

	elif profile == 'dk14':
		
		for i in range(N):
			prof = DK14Profile(M = M_i[i], mdef = mdef_i, z = z_i, c = c_i[i],
							selected = 'by_mass', part = 'inner')
			if mdef_f == '200m':
				Rnew[i] = prof.self.par['R200m']
			else:
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
		return NFWProfile.mu(c) - target
	
	def getX(c, p):
		
		target = NFWProfile.mu(c) * p
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
				target = NFWProfile.mu(c_bins[i]) * p
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
