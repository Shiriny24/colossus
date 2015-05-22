###################################################################################################
#
# Concentration.py 		(c) Benedikt Diemer
#						University of Chicago
#     				    bdiemer@oddjob.uchicago.edu
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

See the documentation of the abstract base class :class:`HaloDensityProfile.HaloDensityProfile` 
for the functionality of the profile objects. For documentation on spherical overdensity mass
definitions, please see the documentation of the :mod:`Halo` module. The following functional forms
for the density profile are implmented:

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
	changeMassDefinitionCModel
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
at z=1, and converted it to the 200m mass definition. If we do not know the concentration in
the initial mass definition, we can use a concentration model to estimate it::

	M200m, R200m, c200m = changeMassDefinitionCModel(1E12, 1.0, 'vir', '200m')
	
By default, the function uses the ``diemer_15`` concentration model (see the documentation of the
:mod:`HaloConcentration` module).

***************************************************************************************************
Alternative mass definitions
***************************************************************************************************

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
:math:`M_{sp}` can only be computed from DK14 profiles. For both mass definitions there are
converter functions:

.. autosummary::	
	RspOverR200m
	MspOverM200m
	Rsp
	Msp
	M4rs

***************************************************************************************************
Units
***************************************************************************************************

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
Detailed Documentation
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import math
import numpy
import scipy.misc
import scipy.optimize
import scipy.integrate
import scipy.interpolate
import abc

import Utilities
import Cosmology
import Halo
import HaloConcentration

###################################################################################################
# ABSTRACT BASE CLASS FOR HALO DENSITY PROFILES
###################################################################################################

class HaloDensityProfile(object):
	"""
	Abstract base class for a halo density profile in physical units.
	
	This class contains a set of quantities that can be computed from halo density profiles in 
	general. In principle, a particular functional form of the profile can be implemented by 
	inheriting this class and overwriting the constructor and density method. In practice there 
	are often faster implementations for particular forms of the profile.
	"""
	
	__metaclass__ = abc.ABCMeta

	def __init__(self):
		
		# The radial limits within which the profile is valid
		self.rmin = 0.0
		self.rmax = numpy.inf
		
		# The radial limits within which we search for spherical overdensity radii. These limits 
		# can be set much tighter for better performance.
		self.min_RDelta = 0.001
		self.max_RDelta = 10000.0
		
		return
	
	###############################################################################################

	@abc.abstractmethod
	def density(self, r):
		"""
		Density as a function of radius.
		
		Abstract function which must be overwritten by child classes.
		
		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		density: array_like
			Density in physical :math:`M_{\odot} h^2 / kpc^3`; has the same dimensions 
			as r.
		"""

		return

	###############################################################################################
	
	def densityDerivativeLin(self, r):
		"""
		The linear derivative of density, :math:`d \\rho / dr`. 

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		derivative: array_like
			The linear derivative in physical :math:`M_{\odot} h / kpc^2`; has the same 
			dimensions as r.
			
		See also
		-------------------------------------------------------------------------------------------
		densityDerivativeLog: The logarithmic derivative of density, :math:`d \log(\\rho) / d \log(r)`. 
		"""
		
		r_use, is_array = Utilities.getArray(r)
		density_der = 0.0 * r_use
		for i in range(len(r_use)):	
			density_der[i] = scipy.misc.derivative(self.density, r_use[i], dx = 0.001, n = 1, order = 3)
		if not is_array:
			density_der = density_der[0]
		
		return density_der

	###############################################################################################
	
	def densityDerivativeLog(self, r):
		"""
		The logarithmic derivative of density, :math:`d \log(\\rho) / d \log(r)`. 

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		derivative: array_like
			The dimensionless logarithmic derivative; has the same dimensions as r.
			
		See also
		-------------------------------------------------------------------------------------------
		densityDerivativeLin: The linear derivative of density, :math:`d \\rho / dr`.
		"""
				
		def logRho(logr):
			return numpy.log(self.density(numpy.exp(logr)))

		r_use, is_array = Utilities.getArray(r)
		density_der = 0.0 * r_use
		for i in range(len(r_use)):	
			density_der[i] = scipy.misc.derivative(logRho, numpy.log(r_use[i]), dx = 0.0001, n = 1, order = 3)
		if not is_array:
			density_der = density_der[0]

		return density_der
		
	###############################################################################################
	
	def enclosedMass(self, r, accuracy = 1E-6):
		"""
		The mass enclosed within radius r.

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
		accuracy: float
			The minimum accuracy of the integration.
			
		Returns
		-------------------------------------------------------------------------------------------
		M: array_like
			The mass enclosed within radius r, in :math:`M_{\odot}/h`; has the same dimensions as r.
		"""		

		def integrand(r):
			return self.density(r) * 4.0 * numpy.pi * r**2

		r_use, is_array = Utilities.getArray(r)
		M = 0.0 * r_use
		for i in range(len(r_use)):	
			M[i], _ = scipy.integrate.quad(integrand, self.rmin, r_use[i], epsrel = accuracy)
		if not is_array:
			M = M[0]
	
		return M

	###############################################################################################

	def cumulativePdf(self, r, Rmax = None, z = None, mdef = None):
		"""
		The cumulative distribution function of the profile.

		Some density profiles do not converge to a finite mass at large radius, and the distribution 
		thus needs to be cut off. The user can specify either a radius (in physical kpc/h) where 
		the profile is cut off, or a mass definition and redshift to compute this radius 
		(e.g., the virial radius :math:`R_{vir}` at z = 0).
		
		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
		Rmax: float
			The radius where to cut off the profile in physical kpc/h.
		z: float
			Redshift
		mdef: str
			The radius definition for the cut-off radius.
		
		Returns
		-------------------------------------------------------------------------------------------
		pdf: array_like
			The probability for mass to lie within radius r; has the same dimensions as r.
		"""		
		
		Rmax_use = None
		if Rmax is not None:
			Rmax_use = Rmax
		elif mdef is not None and z is not None:
			Rmax_use = self.RDelta(z, mdef)
		else:
			msg = 'The cumulative pdf function needs an outer radius for the profile.'
			raise Exception(msg)
			
		pdf = self.enclosedMass(r) / self.enclosedMass(Rmax_use)
		
		return pdf

	###############################################################################################
	
	def surfaceDensity(self, r, accuracy = 1E-6):
		"""
		The projected surface density at radius r.

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
		accuracy: float
			The minimum accuracy of the integration.
			
		Returns
		-------------------------------------------------------------------------------------------
		Sigma: array_like
			The surface density at radius r, in physical :math:`M_{\odot} h/kpc^2`; has the same 
			dimensions as r.
		"""		
		
		def integrand(r, R):
			ret = 2.0 * r * self.density(r) / numpy.sqrt(r**2 - R**2)
			return ret

		r_use, is_array = Utilities.getArray(r)
		surfaceDensity = 0.0 * r_use
		for i in range(len(r_use)):	
			
			if r_use[i] >= self.rmax:
				msg = 'Cannot compute surface density for radius %.2e since rmax is %.2e.' % (r_use[i], self.rmax)
				raise Exception(msg)
			
			surfaceDensity[i], _ = scipy.integrate.quad(integrand, r_use[i], self.rmax, args = r_use[i], \
											epsrel = accuracy, limit = 1000)
		if not is_array:
			surfaceDensity = surfaceDensity[0]

		return surfaceDensity
	
	###############################################################################################

	# This equation is 0 when the enclosed density matches the given density_threshold, and is used 
	# when numerically determining spherical overdensity radii.
	
	def _thresholdEquation(self, r, density_threshold):
		
		diff = self.enclosedMass(r) / 4.0 / math.pi * 3.0 / r**3 - density_threshold
		
		return diff

	###############################################################################################

	def RDelta(self, z, mdef):
		"""
		The spherical overdensity radius of a given mass definition.

		Parameters
		-------------------------------------------------------------------------------------------
		z: float
			Redshift
		mdef: str
			The mass definition for which the spherical overdensity radius is computed.
			
		Returns
		-------------------------------------------------------------------------------------------
		R: float
			Spherical overdensity radius in physical kpc/h.

		See also
		-------------------------------------------------------------------------------------------
		MDelta: The spherical overdensity mass of a given mass definition.
		RMDelta: The spherical overdensity radius and mass of a given mass definition.
		"""		

		density_threshold = Halo.densityThreshold(z, mdef)
		R = scipy.optimize.brentq(self._thresholdEquation, self.min_RDelta, self.max_RDelta, density_threshold)

		return R

	###############################################################################################

	def RMDelta(self, z, mdef):
		"""
		The spherical overdensity radius and mass of a given mass definition.
		
		This is a wrapper for the RDelta and MDelta functions which returns both radius and mass.

		Parameters
		-------------------------------------------------------------------------------------------
		z: float
			Redshift
		mdef: str
			The mass definition for which the spherical overdensity mass is computed.
			
		Returns
		-------------------------------------------------------------------------------------------
		R: float
			Spherical overdensity radius in physical kpc/h.
		M: float
			Spherical overdensity mass in :math:`M_{\odot} /h`.

		See also
		-------------------------------------------------------------------------------------------
		RDelta: The spherical overdensity radius of a given mass definition.
		MDelta: The spherical overdensity mass of a given mass definition.
		"""		
		
		R = self.RDelta(z, mdef)
		M = Halo.R_to_M(R, z, mdef)
		
		return R, M

	###############################################################################################

	# Return the spherical overdensity mass (in Msun / h) for a given mass definition and redshift.

	def MDelta(self, z, mdef):
		"""
		The spherical overdensity mass of a given mass definition.

		Parameters
		-------------------------------------------------------------------------------------------
		z: float
			Redshift
		mdef: str
			The mass definition for which the spherical overdensity mass is computed.
			
		Returns
		-------------------------------------------------------------------------------------------
		M: float
			Spherical overdensity mass in :math:`M_{\odot} /h`.

		See also
		-------------------------------------------------------------------------------------------
		RDelta: The spherical overdensity radius of a given mass definition.
		RMDelta: The spherical overdensity radius and mass of a given mass definition.
		"""		
				
		_, M = self.RMDelta(z, mdef)
		
		return M

###################################################################################################
# SPLINE DEFINED PROFILE
###################################################################################################

class SplineDensityProfile(HaloDensityProfile):
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
		
		HaloDensityProfile.__init__(self)
		
		self.rmin = numpy.min(r)
		self.rmax = numpy.max(r)
		self.min_RDelta = self.rmin
		self.max_RDelta = self.rmax

		if rho is None and M is None:
			msg = 'Either mass or density must be specified.'
			raise Exception(msg)
		
		self.rho_spline = None
		self.M_spline = None
		logr = numpy.log(r)
		
		if M is not None:
			logM = numpy.log(M)
			self.M_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logM)

		if rho is not None:
			logrho = numpy.log(rho)
			self.rho_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logrho)

		# Construct M(r) from density. For some reason, the spline integrator fails on the 
		# innermost bin, and the quad integrator fails on the outermost bin. 
		if self.M_spline is None:
			integrand = 4.0 * numpy.pi * r**2 * rho
			integrand_spline = scipy.interpolate.InterpolatedUnivariateSpline(r, integrand)
			logM = 0.0 * r
			for i in range(len(logM) - 1):
				logM[i], _ = scipy.integrate.quad(integrand_spline, 0.0, r[i])
			logM[-1] = integrand_spline.integral(0.0, r[-1])
			logM = numpy.log(logM)
			self.M_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logM)

		if self.rho_spline is None:
			deriv = self.M_spline(numpy.log(r), nu = 1) * M / r
			logrho = numpy.log(deriv / 4.0 / numpy.pi / r**2)
			self.rho_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logrho)

		return

	###############################################################################################
	# METHODS BOUND TO THE CLASS
	###############################################################################################

	def density(self, r):

		return numpy.exp(self.rho_spline(numpy.log(r)))

	###############################################################################################
	
	def densityDerivativeLin(self, r):

		log_deriv = self.rho_spline(numpy.log(r), nu = 1)
		deriv = log_deriv * self.density(r) / r
		
		return deriv

	###############################################################################################

	def densityDerivativeLog(self, r):
	
		return self.rho_spline(numpy.log(r), nu = 1)
	
	###############################################################################################

	def enclosedMass(self, r):

		return numpy.exp(self.M_spline(numpy.log(r)))
	
###################################################################################################
# NFW PROFILE
###################################################################################################

class NFWProfile(HaloDensityProfile):
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
	xdelta_guess_factors = [5.0, 10.0, 20.0, 100.0, 10000.0]
	xdelta_n_guess_factors = len(xdelta_guess_factors)

	###############################################################################################
	# CONSTRUCTOR
	###############################################################################################

	def __init__(self, rhos = None, rs = None, \
				M = None, c = None, z = None, mdef = None):
		
		HaloDensityProfile.__init__(self)

		# The fundamental way to define an NFW profile by the central density and scale radius
		if rhos is not None and rs is not None:
			self.rhos = rhos
			self.rs = rs
			
		# Alternatively, the user can give a mass and concentration, together with mass definition
		# and redshift.
		elif M is not None and c is not None and mdef is not None and z is not None:
			self.rhos, self.rs = self.fundamentalParameters(M, c, z, mdef)
		
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
		rhos: float
			The central density in physical :math:`M_{\odot} h^2 / kpc^3`; has the same dimensions
			as M.
		rs: float
			The scale radius in physical kpc/h; has the same dimensions as M.
		"""
				
		rs = Halo.M_to_R(M, z, mdef) / c
		rhos = M / rs**3 / 4.0 / math.pi / cls.mu(c)
		
		return rhos, rs

	###############################################################################################

	@staticmethod
	def rho(rhos, x):
		"""
		The NFW density as a function of :math:`x=r/r_s`.
		
		This routine can be called without instantiating an NFWProfile object. In most cases, the 
		:func:`HaloDensityProfile.density` function should be used instead.

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
		HaloDensityProfile.density: Density as a function of radius.
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
		HaloDensityProfile.enclosedMass: The mass enclosed within radius r.
		"""
		
		return numpy.log(1.0 + x) - x / (1.0 + x)
	
	###############################################################################################

	@classmethod
	def M(cls, rhos, rs, x):
		"""
		The enclosed mass in an NFW profile as a function of :math:`x=r/r_s`.

		This routine can be called without instantiating an NFWProfile object. In most cases, the 
		:func:`HaloDensityProfile.enclosedMass` function should be used instead.

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
		HaloDensityProfile.enclosedMass: The mass enclosed within radius r.
		"""
		
		return 4.0 * math.pi * rs**3 * rhos * cls.mu(x)

	###############################################################################################

	@classmethod
	def _thresholdEquationX(cls, x, rhos, density_threshold):
		
		return rhos * cls.mu(x) * 3.0 / x**3 - density_threshold

	###############################################################################################
	
	@classmethod
	def xDelta(cls, rhos, rs, density_threshold, x_guess = 5.0):
		"""
		Find :math:`x=r/r_s` where the enclosed density has a particular value.
		
		This function is the basis for the :func:`HaloDensityProfile.RDelta` routine, but can 
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
			:func:`Halo.densityThreshold` function. 
		
		Returns
		-------------------------------------------------------------------------------------------
		x: float
			The radius in units of the scale radius, :math:`x=r/r_s`, where the enclosed density
			reaches ``density_threshold``. 

		See also
		-------------------------------------------------------------------------------------------
		HaloDensityProfile.RDelta: The spherical overdensity radius of a given mass definition.
		"""
		
		# A priori, we have no idea at what radius the result will come out, but we need to 
		# provide lower and upper limits for the root finder. To balance stability and performance,
		# we do so iteratively: if there is no result within relatively aggressive limits, we 
		# try again with more conservative limits.
		args = rhos, density_threshold
		x = None
		i = 0
		while x is None and i < cls.xdelta_n_guess_factors:
			try:
				xmin = x_guess / cls.xdelta_guess_factors[i]
				xmax = x_guess * cls.xdelta_guess_factors[i]
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
	
		x = r / self.rs
		density = self.rho(self.rhos, x)
		
		return density

	###############################################################################################

	def densityDerivativeLin(self, r):

		x = r / self.rs
		density_der = -self.rhos / self.rs * (1.0 / x**2 / (1.0 + x)**2 + 2.0 / x / (1.0 + x)**3)

		return density_der
	
	###############################################################################################

	def densityDerivativeLog(self, r):

		x = r / self.rs
		density_der = -(1.0 + 2.0 * x / (1.0 + x))

		return density_der

	###############################################################################################

	def enclosedMass(self, r):
		
		x = r / self.rs
		mass = self.M(self.rhos, self.rs, x)
		
		return mass
	
	###############################################################################################
	
	def surfaceDensity(self, r):
	
		x = r / self.rs
		
		if not Utilities.isArray(x):
			x_use = numpy.array([x])
		else:
			x_use = x
		
		surfaceDensity = 0.0 * x_use
		for i in range(len(x_use)):
			
			xx = x_use[i]
			xx2 = xx**2
			
			if abs(xx - 1.0) < 1E-2:
				fx = 0.0
			else:
				if xx > 1.0:
					fx = 1.0 - 2.0 / math.sqrt(xx2 - 1.0) * math.atan(math.sqrt((xx - 1.0) / (xx + 1.0)))
				else:
					fx = 1.0 - 2.0 / math.sqrt(1.0 - xx2) * math.atanh(math.sqrt((1.0 - xx) / (1.0 + xx)))
		
			surfaceDensity[i] = 2.0 * self.rhos * self.rs / (x_use[i]**2 - 1.0) * fx
	
		if not Utilities.isArray(x):
			surfaceDensity = surfaceDensity[0]
	
		return surfaceDensity

	###############################################################################################
	
	# This equation is 0 when the enclosed density matches the given density_threshold. This 
	# function matches the abstract interface in HaloDensityProfile, but for the NFW profile it is
	# easier to solve the equation in x (see the _thresholdEquationX() function).
		
	def _thresholdEquation(self, r, density_threshold):
		
		return self._thresholdEquationX(r / self.rs, self.rhos, density_threshold)

	###############################################################################################

	# Return the spherical overdensity radius (in kpc / h) for a given mass definition and redshift. 
	# This function is overwritten for the NFW profile as we have a better guess at the resulting
	# radius, namely the scale radius. Thus, the user can specify a minimum and maximum concentra-
	# tion that is considered.

	def RDelta(self, z, mdef):
	
		density_threshold = Halo.densityThreshold(z, mdef)
		x = self.xDelta(self.rhos, self.rs, density_threshold)
		R = x * self.rs
		
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
		
		M = self.enclosedMass(4.0 * self.rs)
		
		return M

###################################################################################################
# EINASTO PROFILE
###################################################################################################

class EinastoProfile(HaloDensityProfile):
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

	def __init__(self, rhos = None, rs = None, alpha = None, \
				M = None, c = None, z = None, mdef = None):
	
		HaloDensityProfile.__init__(self)

		# The fundamental way to define an Einasto profile by the density at the scale radius, 
		# the scale radius, and alpha.
		if rhos is not None and rs is not None and alpha is not None:
			
			self.rhos = rhos
			self.rs = rs
			self.alpha = alpha
			
		# Alternatively, the user can give a mass and concentration, together with mass definition
		# and redshift. Passing alpha is now optional since it can also be estimated from the
		# Gao et al. 2008 relation between alpha and peak height. This relation was calibrated for
		# nu_vir, so if the given mass definition is not 'vir' we convert the given mass to Mvir
		# assuming an NFW profile with the given mass and concentration. This leads to a negligible
		# inconsistency, but solving for the correct alpha iteratively would be much slower.
		elif M is not None and c is not None and mdef is not None and z is not None:
			
			R = Halo.M_to_R(M, z, mdef)
			self.rs = R / c
			
			if alpha is None:
				if mdef == 'vir':
					Mvir = M
				else:
					Mvir, _, _ = changeMassDefinition(M, c, z, mdef, 'vir')
				cosmo = Cosmology.getCurrent()
				nu_vir = cosmo.peakHeight(Mvir, z)
				alpha = 0.155 + 0.0095 * nu_vir**2
			
			self.alpha = alpha
			self.rhos = 1.0
			M_unnorm = self.enclosedMass(R)
			self.rhos = M / M_unnorm
					
		else:
			msg = 'An Einasto profile must be define either using rhos, rs, and alpha, or M, c, mdef, and z.'
			raise Exception(msg)

		return

	###############################################################################################
	# METHODS BOUND TO THE CLASS
	###############################################################################################

	def density(self, r):
		
		rho = self.rhos * numpy.exp(-2.0 / self.alpha * ((r / self.rs) ** self.alpha - 1.0))
		
		return rho

	###############################################################################################
	
	def densityDerivativeLin(self, r):

		rho = self.density(r)
		drho_dr = rho * (-2.0 / self.rs) * (r / self.rs)**(self.alpha - 1.0)	
		
		return drho_dr

	###############################################################################################
	
	def densityDerivativeLog(self, r):

		der = -2.0 * (r / self.rs)**self.alpha
		
		return der

###################################################################################################
# DIEMER & KRAVTSOV 2014 PROFILE
###################################################################################################

class DK14Parameters(object):
	"""
	The parameters of the Diemer & Kravtsov 2014 profile
	
	This object specifies the parameters of the Diemer & Kravtsov 2014 halo density profile.
	Note that the majority of these parameters are not free, but fixed depending on either halo
	mass or mass accretion rate.
	
	======= ================ ===================================================================================
	Param.  Symbol           Explanation	
	======= ================ ===================================================================================
	R200m	:math:`R_{200m}` The radius that encloses and average overdensity of 200 :math:`\\rho_m(z)`
	rho_s	:math:`\\rho_s`   The central scale density, in physical :math:`M_{\odot} h^2 / kpc^3`
	rho_m   :math:`\\rho_m`   The mean matter density of the universe, in physical :math:`M_{\odot} h^2 / kpc^3`
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
	"""
	
	def __init__(self):
		
		self.R200m = 0.0
		self.rho_s = 0.0
		self.rho_m = 0.0
		self.rs = 0.0
		self.rt = 0.0
		self.alpha = 0.0
		self.beta = 0.0
		self.gamma = 0.0
		self.be = 0.0
		self.se = 0.0
		self.part = 'both'
		
		return

###################################################################################################

class DK14Profile(HaloDensityProfile):
	"""
	The Diemer & Kravtsov 2014 density profile.
	
	This profile corresponds to an Einasto profile at small radii, and steepens around the virial 
	radius. At large radii, the profile approaches a power-law in r. The profile formula has 8
	free parameters, but most of those are fixed to particular values. This is done automatically,
	the user only needs to pass the mass of a halo, and optionally concentration. However, there are
	some further options.
	
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
	
	Parameters
	-----------------------------------------------------------------------------------------------
	par: DK14Parameters
		A parameters object
	**kwargs:
		The parameters of the DK14 profile as keyword args. See the deriveParameters function. 
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
	
	def __init__(self, par = None, **kwargs):
	
		HaloDensityProfile.__init__(self)

		# The following parameters are not constants, they are temporarily changed by certain 
		# functions.
		self.accuracy_mass = 1E-4
		self.accuracy_radius = 1E-4

		if par is not None:
			self.par = par
		else:
			self.deriveParameters(**kwargs)

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
	def rtOverR200m(selected, nu200m = None, z = None, Gamma = None):
		
		if selected == 'by_mass':
			ratio = 1.9 - 0.18 * nu200m
		
		elif selected == 'by_accretion_rate':
			if (Gamma is not None) and (z is not None):
				cosmo = Cosmology.getCurrent()
				ratio =  0.43 * (1.0 + 0.92 * cosmo.Om(z)) * (1.0 + 2.18 * numpy.exp(-Gamma / 1.91))
			elif nu200m is not None:
				ratio = 0.79 * (1.0 + 1.63 * numpy.exp(-nu200m / 1.56))
			else:
				msg = 'Need either Gamma and z, or nu.'
				raise Exception(msg)

		return ratio

	###############################################################################################
	# METHODS BOUND TO THE CLASS
	###############################################################################################

	def deriveParameters(self, M = None, c = None, z = None, mdef = None, \
			selected = 'by_mass', Gamma = None, part = 'both', be = None, se = None, \
			acc_warn = 0.01, acc_err = 0.05):
		"""
		Get the native DK14 parameters given a halo mass, and possibly concentration.
		
		Get the DK14 parameters that correspond to a profile with a particular mass M in some mass
		definition mdef. Optionally, the user can define the concentration c; otherwise, it is 
		computed automatically. 
		
		Parameters
		-----------------------------------------------------------------------------------------------
		M: float
			Halo mass in :math:`M_{\odot}/h`.
		c: float
			Concentration. If this parameter is None, c is estimated using the model of 
			Diemer & Kravtsov 2015. 
		z: float
			Redshift
		mdef: str
			The mass definition to which M corresponds.
		selected: str
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
		
		# Declare shared variables; these parameters are advanced during the iterations
		par2 = {}
		par2['RDelta'] = 0.0
		self.par = DK14Parameters()
		
		RTOL = 0.01
		MTOL = 0.01
		GUESS_TOL = 2.5
		self.accuracy_mass = MTOL
		self.accuracy_radius = RTOL
	
		# -----------------------------------------------------------------------------------------

		# Try a radius R200m, compute the resulting RDelta using the old RDelta as a starting guess
		
		def radius_diff(R200m, par2, Gamma, rho_target, R_target):
			
			self.par.R200m = R200m
			M200m = Halo.R_to_M(R200m, z, '200m')
			nu200m = cosmo.peakHeight(M200m, z)

			self.par.alpha, self.par.beta, self.par.gamma, rt_R200m = \
				self.getFixedParameters(selected, nu200m = nu200m, z = z, Gamma = Gamma)
			self.par.rt = rt_R200m * R200m
			self.normalize(R200m, M200m, mtol = MTOL)

			par2['RDelta'] = self._RDeltaLowlevel(par2['RDelta'], rho_target, guess_tolerance = GUESS_TOL)
			
			return par2['RDelta'] - R_target
		
		# -----------------------------------------------------------------------------------------
		
		# Test for wrong user input
		if part in ['outer', 'both'] and (be is None or se is None):
			msg = "Since part = %s, the parameters be and se must be set. The recommended values are 1.0 and 1.5." % (part)
			raise Exception(msg)
		
		# The user needs to set a cosmology before this function can be called
		cosmo = Cosmology.getCurrent()
		
		# Get concentration if the user hasn't supplied it, compute scale radius
		if c is None:
			c = HaloConcentration.concentration(M, mdef, z, statistic = 'median')
		R_target = Halo.M_to_R(M, z, mdef)

		self.par.rs = R_target / c
		self.par.rho_m = cosmo.rho_m(z)
		self.par.be = be
		self.par.se = se
		self.par.part = part
		
		if mdef == '200m':
			
			# The user has supplied M200m, the parameters follow directly from the input
			M200m = M
			self.par.R200m = Halo.M_to_R(M200m, z, '200m')
			nu200m = cosmo.peakHeight(M200m, z)
			self.par.alpha, self.par.beta, self.par.gamma, rt_R200m = \
				self.getFixedParameters(selected, nu200m = nu200m, z = z, Gamma = Gamma)
			self.par.rt = rt_R200m * self.par.R200m
			self.normalize(self.par.R200m, M200m, mtol = MTOL)
			
		else:
			
			# The user has supplied some other mass definition, we need to iterate.
			_, R200m_guess, _ = changeMassDefinition(M, c, z, mdef, '200m')
			par2['RDelta'] = R_target

			# Iterate to find an M200m for which the desired mass is correct
			rho_target = Halo.densityThreshold(z, mdef)
			args = par2, Gamma, rho_target, R_target
			self.par.R200m = scipy.optimize.brentq(radius_diff, R200m_guess / 1.3, R200m_guess * 1.3, \
								args = args, xtol = RTOL)

			# Check the accuracy of the result; M should be very close to MDelta now
			M_result = Halo.R_to_M(par2['RDelta'], z, mdef)
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
	
	def getFixedParameters(self, selected, nu200m = None, z = None, Gamma = None):
	
		if selected == 'by_mass':
			beta = 4.0
			gamma = 8.0
			rt_R200m = self.rtOverR200m('by_mass', nu200m = nu200m)
		elif selected == 'by_accretion_rate':
			beta = 6.0
			gamma = 4.0
			rt_R200m = self.rtOverR200m('by_accretion_rate', z = z, Gamma = Gamma)
		else:
			msg = "HaloDensityProfile.DK14_getFixedParameters: Unknown sample selection, %s." % (selected)
			raise Exception(msg)
		
		# Gao et al. 2008 relation between alpha and nu. This function was originally calibrated for 
		# nu = nu_vir, but the difference is very small.
		alpha = 0.155 + 0.0095 * nu200m**2

		return alpha, beta, gamma, rt_R200m

	###############################################################################################

	# Set the rhos parameter such that the profile encloses mass M at radius R

	def normalize(self, R, M, mtol = 0.01):

		part_true = self.par.part
		self.par.part = 'inner'
		self.par.rho_s = 1.0

		Mr_inner = self.enclosedMass(R)
		if part_true == 'both':
			self.par.part = 'outer'
			Mr_outer = self.enclosedMass(R)
		elif part_true == 'inner':
			Mr_outer = 0.0
		else:
			msg = "Invalid value for part, %s." % (part_true)
			raise Exception(msg)
			
		self.par.rho_s = (M - Mr_outer) / Mr_inner
		self.par.part = part_true
		
		return

	###############################################################################################

	# The power-law outer profile is cut off at 1 / max_outer_prof to avoid a spurious density 
	# spike at very small radii if the slope of the power-law (se) is steep.
	
	def density(self, r):
		
		rho = r * 0.0
		par = self.par
		
		if par.part in ['inner', 'both']:
			inner = par.rho_s * numpy.exp(-2.0 / par.alpha * ((r / par.rs) ** par.alpha - 1.0))
			fT = (1.0 + (r / par.rt) ** par.beta) ** (-par.gamma / par.beta)
			rho += inner * fT
		
		if par.part in ['outer', 'both']:
			outer = par.rho_m * (1.0 + par.be / (self.max_outer_prof + (r / 5.0 / par.R200m)**par.se))
			rho += outer
		
		return rho

	###############################################################################################
	
	def densityDerivativeLin(self, r):
		
		drho_dr = r * 0.0
		par = self.par
		
		if par.part in ['inner', 'both']:
			inner = par.rho_s * numpy.exp(-2.0 / par.alpha * ((r / par.rs) ** par.alpha - 1.0))
			d_inner = inner * (-2.0 / par.rs) * (r / par.rs)**(par.alpha - 1.0)	
			fT = (1.0 + (r / par.rt) ** par.beta) ** (-par.gamma / par.beta)
			d_fT = (-par.gamma / par.beta) * (1.0 + (r / par.rt) ** par.beta) ** (-par.gamma / par.beta - 1.0) * \
				par.beta / par.rt * (r / par.rt) ** (par.beta - 1.0)
			drho_dr += inner * d_fT + d_inner * fT
	
		if par.part in ['outer', 'both']:
			t1 = 1.0 / 5.0 / par.R200m
			t2 = r * t1
			drho_dr += -par.rho_m * par.be * par.se * t1 * (self.max_outer_prof + t2**par.se)**-2 \
				* t2**(par.se - 1.0)
		
		return drho_dr

	###############################################################################################
	
	def densityDerivativeLog(self, r):
		
		drho_dr = self.densityDerivativeLin(r)
		rho = self.density(r)
		der = drho_dr * r / rho
		
		return der

	###############################################################################################

	# Low-level function to compute a spherical overdensity radius given the parameters of a DK14 
	# profile, the desired overdensity threshold, and an initial guess. A more user-friendly version
	# can be found above (DK14_getMR).
	
	def _RDeltaLowlevel(self, R_guess, density_threshold, guess_tolerance = 5.0):
			
		R = scipy.optimize.brentq(self._thresholdEquation, R_guess / guess_tolerance, \
				R_guess * guess_tolerance, args = density_threshold, xtol = self.accuracy_radius)
		
		return R
	
	###############################################################################################

	# This function returns the spherical overdensity radius (in kpc / h) given a mass definition
	# and redshift. We know R200m and thus M200m for a DK14 profile, and use those parameters to
	# compute what R would be for an NFW profile and use this radius as an initial guess.
	
	def RDelta(self, z, mdef):
	
		M200m = Halo.R_to_M(self.par.R200m, z, mdef)
		_, R_guess, _ = changeMassDefinition(M200m, self.par.R200m / self.par.rs, z, '200m', mdef)
		density_threshold = Halo.densityThreshold(z, mdef)
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
		
		M = self.enclosedMass(4.0 * self.par.rs)
		
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
		
		R200m = self.par.R200m
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
	
	M_i, is_array = Utilities.getArray(M_i)
	c_i, _ = Utilities.getArray(c_i)
	N = len(M_i)
	Rnew = numpy.zeros((N), dtype = float)
	cnew = numpy.zeros((N), dtype = float)

	if profile == 'nfw':
		
		# We do not instantiate NFW profile objects, but instead use the faster static functions
		rhos, rs = NFWProfile.fundamentalParameters(M_i, c_i, z_i, mdef_i)
		density_threshold = Halo.densityThreshold(z_f, mdef_f)
		for i in range(N):
			cnew[i] = NFWProfile.xDelta(rhos[i], rs[i], density_threshold, x_guess = c_i[i])
		Rnew = rs * cnew

	elif profile == 'dk14':
		
		for i in range(N):
			prof = DK14Profile(M = M_i[i], mdef = mdef_i, z = z_i, c = c_i[i], \
							selected = 'by_mass', part = 'inner')
			if mdef_f == '200m':
				Rnew[i] = prof.par.R200m
			else:
				Rnew[i] = prof.RDelta(z_f, mdef_f)
			cnew[i] = Rnew[i] / prof.rs

	else:
		msg = 'This function is not defined for profile %s.' % (profile)
		raise Exception(msg)

	if not is_array:
		Rnew = Rnew[0]
		cnew = cnew[0]

	Mnew = Halo.R_to_M(Rnew, z_f, mdef_f)
	
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
	changeMassDefinitionCModel: Change the spherical overdensity mass definition, using a model for the concentration.
	"""
	
	return pseudoEvolve(M, c, z, mdef_in, z, mdef_out, profile = profile)

###################################################################################################

def changeMassDefinitionCModel(M, z, mdef_in, mdef_out, profile = 'nfw', c_model = 'diemer15'):
	"""
	Change the spherical overdensity mass definition, using a model for the concentration.
	
	This function is a wrapper for the :func:`changeMassDefinition` function. Instead of forcing 
	the user to provide concentrations, they are computed from a model indicated by the ``c_model``
	parameter.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	M_i: array_like
		The initial halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z_i: float
		The initial redshift.
	mdef_i: str
		The initial mass definition.
	mdef_f: str
		The final mass definition (can be the same as mdef_i, or different).
	profile: str
		The functional form of the profile assumed in the computation; can be ``nfw`` or ``dk14``.
	c_model: str
		The identifier of a concentration model (see :mod:`HaloConcentration` for valid inputs).

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
	changeMassDefinition: Change the spherical overdensity mass definition.
	"""
	
	c = HaloConcentration.concentration(M, mdef_in, z, model = c_model)
	
	return pseudoEvolve(M, c, z, mdef_in, z, mdef_out, profile = profile)

###################################################################################################

def M4rs(M, z, mdef, c = None):
	"""
	Convert a spherical overdensity mass to :math:`M_{<4rs}`.
	
	See the section on mass definitions for the definition of :math:`M_{<4rs}`.

	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Spherical overdensity halo mass in :math:`M_{\odot} / h`; can be a number or a numpy
		array.
	z: float
		Redshift
	mdef: str
		The spherical overdensity mass definition in which M (and optionally c) are given.
	c: array_like
		Concentration. If this parameter is not passed, concentration is automatically 
		computed. Must have the same dimensions as M.
		
	Returns
	-----------------------------------------------------------------------------------------------
	M4rs: array_like
		The mass within 4 scale radii, :math:`M_{<4rs}`, in :math:`M_{\odot} / h`; has the 
		same dimensions as M.
	"""

	if c is None:
		c = HaloConcentration.concentration(M, mdef, z)
	
	Mfrs = M * NFWProfile.mu(4.0) / NFWProfile.mu(c)
	
	return Mfrs

###################################################################################################

def RspOverR200m(nu200m = None, z = None, Gamma = None):
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

	if (Gamma is not None) and (z is not None):
		cosmo = Cosmology.getCurrent()
		ratio =  0.54 * (1 + 0.53 * cosmo.Om(z)) * (1 + 1.36 * numpy.exp(-Gamma / 3.04))
	elif nu200m is not None:
		ratio = 0.81 * (1.0 + 0.97 * numpy.exp(-nu200m / 2.44))
	else:
		msg = 'Need either Gamma and z, or nu.'
		raise Exception(msg)

	return ratio

###################################################################################################

def MspOverM200m(nu200m = None, z = None, Gamma = None):
	"""
	The ratio :math:`M_{sp} / M_{200m}` from either the accretion rate, :math:`\\Gamma`, or
	the peak height, :math:`\\nu`.
	
	This function implements the relations calibrated in More, Diemer & Kravtsov 2015. Either
	the accretion rate :math:`\\Gamma` and redshift, or the peak height :math:`\\nu`, must not 
	be ``None``. 

	Parameters
	-----------------------------------------------------------------------------------------------
	nu_vir: array_like
		The peak height as computed from :math:`M_{200m}`; can be a number or a numpy array.
	z: array_like
		Redshift; can be a number or a numpy array.
	Gamma: array_like
		The mass accretion rate, as defined in Diemer & Kravtsov 2014; can be a number or a 
		numpy array.
	
	Returns
	-----------------------------------------------------------------------------------------------
	ratio: array_like
		:math:`M_{sp} / M_{200m}`; has the same dimensions as z, Gamma, or nu, depending
		on which of those parameters is an array.
		
	See also
	-----------------------------------------------------------------------------------------------
	RspOverR200m: The ratio :math:`R_{sp} / R_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
	Rsp: :math:`R_{sp}` as a function of spherical overdensity radius.
	Msp: :math:`M_{sp}` as a function of spherical overdensity mass.
	"""
	
	if (Gamma is not None) and (z is not None):
		cosmo = Cosmology.getCurrent()
		ratio =  0.59 * (1 + 0.35 * cosmo.Om(z)) * (1 + 0.92 * numpy.exp(-Gamma / 4.54))
	elif nu200m is not None:
		ratio = 0.82 * (1.0 + 0.63 * numpy.exp(-nu200m / 3.52))
	else:
		msg = 'Need either Gamma and z, or nu.'
		raise Exception(msg)
	
	return ratio

###################################################################################################

def Rsp(R, z, mdef, c = None, profile = 'nfw'):
	"""
	:math:`R_{sp}` as a function of spherical overdensity radius.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	R: array_like
		Spherical overdensity radius in physical :math:`kpc/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		Mass definition in which R and c are given.
	c: array_like
		Halo concentration; must have the same dimensions as R, or be ``None`` in which case the 
		concentration is computed automatically.
	profile: str
		The functional form of the profile assumed in the conversion between mass definitions; 
		can be ``nfw`` or ``dk14``.

	Returns
	-----------------------------------------------------------------------------------------------
	Rsp: array_like
		:math:`R_{sp}` in physical :math:`kpc/h`; has the same dimensions as R.
		
	See also
	-----------------------------------------------------------------------------------------------
	RspOverR200m: The ratio :math:`R_{sp} / R_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
	MspOverM200m: The ratio :math:`M_{sp} / M_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
	Msp: :math:`M_{sp}` as a function of spherical overdensity mass.
	"""
	
	if mdef == '200m':
		R200m = R
		M200m = Halo.R_to_M(R200m, z, '200m')
	else:
		M = Halo.R_to_M(R, z, mdef)
		M200m, R200m, _ = changeMassDefinition(M, c, z, mdef, '200m', profile = profile)

	cosmo = Cosmology.getCurrent()
	nu200m = cosmo.peakHeight(M200m, z)
	Rsp = R200m * RspOverR200m(nu200m = nu200m)
	
	return Rsp

###################################################################################################

def Msp(M, z, mdef, c = None, profile = 'nfw'):
	"""
	:math:`M_{sp}` as a function of spherical overdensity mass.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Spherical overdensity mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		Mass definition in which M and c are given.
	c: array_like
		Halo concentration; must have the same dimensions as M, or be ``None`` in which case the 
		concentration is computed automatically.
	profile: str
		The functional form of the profile assumed in the conversion between mass definitions; 
		can be ``nfw`` or ``dk14``.

	Returns
	-----------------------------------------------------------------------------------------------
	Msp: array_like
		:math:`M_{sp}` in :math:`M_{\odot}/h`; has the same dimensions as M.
		
	See also
	-----------------------------------------------------------------------------------------------
	RspOverR200m: The ratio :math:`R_{sp} / R_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
	MspOverM200m: The ratio :math:`M_{sp} / M_{200m}` from either the accretion rate, :math:`\\Gamma`, or the peak height, :math:`\\nu`.
	Rsp: :math:`R_{sp}` as a function of spherical overdensity radius.
	"""
	
	if mdef == '200m':
		M200m = M
	else:
		M200m, _, _ = changeMassDefinition(M, c, z, mdef, '200m', profile = profile)
	
	cosmo = Cosmology.getCurrent()
	nu200m = cosmo.peakHeight(M200m, z)
	Msp = M200m * MspOverM200m(nu200m = nu200m)
	
	return Msp

###################################################################################################

def radiusFromPdf(M, z, mdef, cumulativePdf, c = None, c_model = 'diemer15', \
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
	z: float
		Redshift
	mdef: str
		The mass definition in which the halo mass M is given. 
	cumulativePdf: array_like
		The cumulative pdf that we are seeking. If an array, this array needs to have the same 
		dimensions as the M array.
	c: array_like
		If ``c is None``, the ``c_model`` concentration model is used to determine the mean c for each
		mass. The user can also supply concentrations in an array of the same dimensions as the
		M array.
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
	
	M_array, is_array = Utilities.getArray(M)
	R = Halo.M_to_R(M, z, mdef)
	N = len(M_array)
	x = 0.0 * M_array
	if c is None:
		c = HaloConcentration.concentration(M, mdef, z, model = c_model)
	c_array, _ = Utilities.getArray(c)
	p_array, _ = Utilities.getArray(cumulativePdf)
	
	if interpolate:

		# Create an interpolator on a regular grid in c-p space.
		bin_width_c = 0.1
		c_min = numpy.min(c_array) * 0.99
		c_max = numpy.max(c_array) * 1.01
		c_bins = numpy.arange(c_min, c_max + bin_width_c, bin_width_c)
		
		p_bins0 = numpy.arange(0.0, 0.01, 0.001)
		p_bins1 = numpy.arange(0.01, 0.1, 0.01)
		p_bins2 = numpy.arange(0.1, 1.1, 0.1)
		p_bins = numpy.concatenate((p_bins0, p_bins1, p_bins2))
		
		N_c = len(c_bins)
		N_p = len(p_bins)

		x_ = numpy.zeros((N_c, N_p), dtype = float)
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
