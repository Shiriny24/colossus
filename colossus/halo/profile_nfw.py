###################################################################################################
#
# profile_nfw.py                (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import numpy as np
import scipy.optimize

from colossus.utils import utilities
from colossus.halo import basics
from colossus.halo import profile_base

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
				M = None, c = None, z = None, mdef = None, **kwargs):
		
		self.par_names = ['rhos', 'rs']
		self.opt_names = []
		profile_base.HaloDensityProfile.__init__(self, **kwargs)

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
	
	def densityInner(self, r):
	
		x = r / self.par['rs']
		density = self.rho(self.par['rhos'], x)
		
		return density

	###############################################################################################

	def densityDerivativeLinInner(self, r):

		x = r / self.par['rs']
		density_der = -self.par['rhos'] / self.par['rs'] * (1.0 / x**2 / (1.0 + x)**2 + 2.0 / x / (1.0 + x)**3)

		return density_der
	
	###############################################################################################

	def densityDerivativeLogInner(self, r):

		x = r / self.par['rs']
		density_der = -(1.0 + 2.0 * x / (1.0 + x))

		return density_der

	###############################################################################################

	def enclosedMassInner(self, r):
		
		x = r / self.par['rs']
		mass = self.M(self.par['rhos'], self.par['rs'], x)
		
		return mass
	
	###############################################################################################
	
	# The surface density of an NFW profile can be computed analytically which is much faster than
	# integration. The formula below is taken from Bartelmann (1996). The case r = rs is solved in 
	# Lokas & Mamon (2001), but in their notation the density at this radius looks somewhat 
	# complicated. In the notation used here, Sigma(rs) = 2/3 * rhos * rs.
	
	def surfaceDensityInner(self, r):
	
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
