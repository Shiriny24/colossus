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
for the density profile are implemented:

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
Profile fitting
***************************************************************************************************

Here, fitting refers to finding the parameters of a halo density profile which best describe a
given set of data points. Each point corresponds to a radius and a particular quantity, such as 
density, enclosed mass, or surface density. Optionally, the user can pass uncertainties on the 
data points, or even a full covariance matrix. All fitting should be done using the very general 
:func:`HaloDensityProfile.fit` routine. For example, let us fit an NFW profile to some density 
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
	
The :func:`HaloDensityProfile.fit` function accepts many input options, some specific to the 
fitting method used. Please see the detailed documentation below.

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
import scipy.special
import abc
import collections

from utils import MCMC
from utils import Utilities
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
		
		# For some functions, such as Vmax, we need an intial guess for a radius.
		self.r_guess = 100.0
		
		# The parameters of the profile are stored in a dictionary
		self.par = collections.OrderedDict()
		self.N_par = len(self.par_names)
		for name in self.par_names:
			self.par[name] = 0.0

		# Additionally to the numerical parameters, there can be options
		self.opt = collections.OrderedDict()
		self.N_opt = len(self.opt_names)
		for name in self.opt_names:
			self.opt[name] = None
			
		# Function pointers to various physical quantities. This can be overwritten or extended
		# by child classes.
		self.quantities = {}
		self.quantities['rho'] = self.density
		self.quantities['M'] = self.enclosedMass
		self.quantities['Sigma'] = self.surfaceDensity

		return

	###############################################################################################

	def getParameterArray(self, mask = None):
		"""
		Returns an array of the profile parameters.
		
		The profile parameters are internally stored in an ordered dictionary. For some 
		applications (e.g., fitting), a simply array is more appropriate.
		
		Parameters
		-------------------------------------------------------------------------------------------
		mask: array_like
			Optional; must be a numpy array (not a list) of booleans, with the same length as the
			parameter vector of the profile class (profile.N_par). Only those parameters that 
			correspond to ``True`` values are returned.

		Returns
		-------------------------------------------------------------------------------------------
		par: array_like
			A numpy array with the profile's parameter values.
		"""
		
		par = numpy.array(self.par.values())
		if mask is not None:
			par = par[mask]
			
		return par
	
	###############################################################################################
	
	def setParameterArray(self, pars, mask = None):
		"""
		Set the profile parameters from an array.
		
		The profile parameters are internally stored in an ordered dictionary. For some 
		applications (e.g., fitting), setting them directly from an array might be necessary. If 
		the profile contains values that depend on the parameters, the profile class must overwrite
		this function and update according to the new parameters.
		
		Parameters
		-------------------------------------------------------------------------------------------
		pars: array_like
			The new parameter array.
		mask: array_like
			Optional; must be a numpy array (not a list) of booleans, with the same length as the
			parameter vector of the profile class (profile.N_par). If passed, only those 
			parameters that correspond to ``True`` values are set (meaning the pars parameter must
			be shorter than profile.N_par).
		"""
		
		if mask is None:		
			for i in range(self.N_par):
				self.par[self.par_names[i]] = pars[i]
		else:
			counter = 0
			for i in range(self.N_par):
				if mask[i]:
					self.par[self.par_names[i]] = pars[counter]
					counter += 1
					
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

	def circularVelocity(self, r):
		"""
		The circular velocity, :math:`v_c \\equiv \\sqrt{GM(<r)/r}`.

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
			
		Returns
		-------------------------------------------------------------------------------------------
		vc: float
			The circular velocity in km / s; has the same dimensions as r.

		See also
		-------------------------------------------------------------------------------------------
		Vmax: The maximum circular velocity, and the radius where it occurs.
		"""		
	
		M = self.enclosedMass(r)
		v = numpy.sqrt(Cosmology.AST_G * M / r)
		
		return v

	###############################################################################################

	# This helper function is used for Vmax where we need to minimize -vc.

	def _circularVelocity_negative(self, r):
		
		return -self.circularVelocity(r)

	###############################################################################################

	def Vmax(self):
		"""
		The maximum circular velocity, and the radius where it occurs.
			
		Returns
		-------------------------------------------------------------------------------------------
		vmax: float
			The maximum circular velocity in km / s.
		rmax: float
			The radius where fmax occurs, in physical kpc/h.

		See also
		-------------------------------------------------------------------------------------------
		circularVelocity: The circular velocity, :math:`v_c \\equiv \\sqrt{GM(<r)/r}`.
		"""		
		
		res = scipy.optimize.minimize(self._circularVelocity_negative, self.r_guess)
		rmax = res.x[0]
		vmax = self.circularVelocity(rmax)
		
		return vmax, rmax

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
	
	###############################################################################################

	# Return a numpy array of fitting parameters, given the standard profile parameters. By default, 
	# the parameters used in a fit are the same as the fundamental parameters. Derived classes 
	# might want to change that, for example to fit log(p) instead of p if the value can only be 
	# positive. 

	def _fit_convertParams(self, p, mask):
		
		return p

	###############################################################################################
	
	def _fit_convertParamsBack(self, p, mask):
		
		return p

	###############################################################################################

	# This function is evaluated before any derivatives etc. Thus, we set the new set of 
	# parameters here. Note that the matrix Q is the matrix that is dot-multiplied with the 
	# difference vector; this is not the same as the inverse covariance matrix.	

	def _fit_diff_function(self, x, r, q, f, fder, Q, mask, N_par_fit, verbose):

		print 'diff'
		print x
		self.setParameterArray(self._fit_convertParamsBack(x, mask), mask = mask)
		q_fit = f(r)
		q_diff = q_fit - q
		mf = numpy.dot(Q, q_diff)

		return mf

	###############################################################################################

	# Evaluate the derivative of the parameters, and multiply with the same matrix as in the diff
	# function. This function should only be called if fp is not None, i.e. if the analytical 
	# derivative is implemented.

	def _fit_param_deriv_highlevel(self, x, r, q, f, fder, Q, mask, N_par_fit, verbose):
		
		deriv = fder(self, r, mask, N_par_fit)		
		for j in range(N_par_fit):
			deriv[j] = numpy.dot(Q, deriv[j])
			
		return deriv

	###############################################################################################

	def _fit_chi2(self, r, q, f, covinv):

		q_model = f(r)
		diff = q_model - q
		chi2 = numpy.dot(numpy.dot(diff, covinv), diff)
		
		return chi2

	###############################################################################################
	
	# Evaluate the likelihood for a vector of parameter sets x. In this case, the vector is 
	# evaluated element-by-element, but the function is expected to handle a vector since this 
	# could be much faster for a simpler likelihood.
	
	def _fit_likelihood(self, x, r, q, f, covinv, mask):

		n_eval = len(x)
		res = numpy.zeros((n_eval), numpy.float)
		for i in range(n_eval):
			self.setParameterArray(x[i], mask = mask)
			res[i] = numpy.exp(-0.5 * self._fit_chi2(r, q, f, covinv))
		
		return res

	###############################################################################################

	# Note that the MCMC fitter does NOT use the converted fitting parameters, but just the 
	# parameters themselves. Otherwise, interpreting the chain becomes very complicated.

	def _fit_method_mcmc(self, r, q, f, covinv, mask, N_par_fit, verbose, \
				converged_GR, nwalkers, best_fit, initial_step, random_seed, convergence_step):
		
		x0 = self.getParameterArray(mask = mask)
		args = r, q, f, covinv, mask
		walkers = MCMC.initWalkers(x0, initial_step = initial_step, nwalkers = nwalkers, random_seed = random_seed)
		xi = numpy.reshape(walkers, (len(walkers[0]) * 2, len(walkers[0, 0])))
		chain_thin, chain_full, R = MCMC.runChain(self._fit_likelihood, walkers, convergence_step = convergence_step, \
							args = args, converged_GR = converged_GR, verbose = verbose)
		mean, median, stddev, p = MCMC.analyzeChain(chain_thin, self.par_names, verbose = verbose)

		dict = {}
		dict['x_initial'] = xi
		dict['chain_full'] = chain_full
		dict['chain_thin'] = chain_thin
		dict['R'] = R
		dict['x_mean'] = mean
		dict['x_median'] = median
		dict['x_stddev'] = stddev
		dict['x_percentiles'] = p
		
		if best_fit == 'mean':
			x = mean
		elif best_fit == 'median':
			x = median

		self.setParameterArray(x, mask = mask)
		
		return x, dict

	###############################################################################################

	def _fit_method_leastsq(self, r, q, f, fder, Q, mask, N_par_fit, verbose, tolerance):
		
		# Prepare arguments
		if fder is None:
			deriv_func = None
		else:
			deriv_func = self._fit_param_deriv_highlevel	
		args = r, q, f, fder, Q, mask, N_par_fit, verbose

		# Run the actual fit
		ini_guess = self._fit_convertParams(self.getParameterArray(mask = mask), mask)
		x_fit, cov, dict, fit_msg, err_code = scipy.optimize.leastsq(self._fit_diff_function, ini_guess, \
							Dfun = deriv_func, col_deriv = 1, args = args, full_output = 1, \
							xtol = tolerance)
		
		# Check the output
		if not err_code in [1, 2, 3, 4]:
			msg = 'Fitting failed, message: %s' % (fit_msg)
			raise Warning(msg)

		# Set the best-fit parameters
		x = self._fit_convertParamsBack(x_fit, mask)
		self.setParameterArray(x, mask = mask)

		# The fitter sometimes fails to derive a covariance matrix
		if cov is not None:
			
			# The covariance matrix is in relative units, i.e. needs to be multiplied with the 
			# residual chi2
			diff = self._fit_diff_function(x_fit, *args)
			residual = numpy.sum(diff**2) / (len(r) - N_par_fit)
			cov *= residual

			# Derive an estimate of the uncertainty from the covariance matrix. We need to take into
			# account that cov refers to the fitting parameters which may not be the same as the 
			# standard profile parameters.
			sigma = numpy.sqrt(numpy.diag(cov))
			err = numpy.zeros((2, N_par_fit), numpy.float)
			err[0] = self._fit_convertParamsBack(x_fit - sigma, mask)
			err[1] = self._fit_convertParamsBack(x_fit + sigma, mask)

		else:
			
			msg = 'WARNING: Could not determine uncertainties on fitted parameters. Set all uncertainties to zero.'
			print(msg)
			err = numpy.zeros((2, N_par_fit), numpy.float)
			
		dict['x_err'] = err

		# Print solution
		if verbose:
			msg = 'Found solution in %d steps. Best-fit parameters:' % (dict['nfev'])
			print(msg)
			counter = 0
			for i in range(self.N_par):
				if mask is None or mask[i]:
					msg = 'Parameter %10s = %7.2e [%7.2e .. %7.2e]' \
						% (self.par_names[i], x[counter], err[0, counter], err[1, counter])
					counter += 1
					print(msg)
					
		return x, dict

	###############################################################################################

	# This function represents a general interface for fitting, and should not have to be 
	# overwritten by child classes.

	def fit(self, 
		# Input data
		r, q, quantity, q_err = None, q_cov = None, \
		# General fitting options: method, parameters to vary
		method = 'leastsq', mask = None, verbose = True, \
		# Options specific to leastsq
		tolerance = 1E-5, \
		# Options specific to the MCMC initialization
		initial_step = 0.1, nwalkers = 100, random_seed = None, \
		# Options specific to running the MCMC chain and its analysis
		convergence_step = 100, converged_GR = 0.01, best_fit = 'median'):
		"""
		Fit the density, mass, or surface density profile to a given set of data points.
		
		This function represents a general interface for finding the best-fit parameters of a 
		halo density profile given a set of data points. These points can represent a number of
		different physical quantities: ``quantity`` can either be density, enclosed mass, or 
		surface density (``rho``, ``M``, or ``Sigma``). The data points q at radii r can optionally 
		have error bars, and the user can pass a full covariance matrix.
		
		There are two fundamental methods for performing the fit, a least-squares minimization 
		(method = leastsq) and a Markov-Chain Monte Carlo (method = mcmc). The MCMC method has some
		specific options (see below). In either case, the current parameters of the profile instance 
		serve as an initial guess. Finally, the user can choose to vary only a sub-set of the
		profile parameters through the ``mask`` parameter.
		
		The function returns a dictionary with outputs that depend on which method is chosen. After
		this function has completed, the profile instance represents the best-fit profile to the 
		data points (i.e., its parameters are the best-fit parameters). Note that all output 
		parameters are bundled into one dictionary. The explanations below refer to the entries in
		this dictionary.

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			The radii of the data points, in physical kpc/h.
		q: array_like
			The data to fit; can either be density in physical :math:`M_{\odot} h^2 / kpc^3`, 
			enclosed mass in :math:`M_{\odot} /h`, or surface density in physical 
			:math:`M_{\odot} h/kpc^2`. Must have the same dimensions as r.
		quantity: str
			Indicates which quantity is given in the q input, can be ``rho``, ``M``, or ``Sigma``.
		q_err: array_like
			Optional; the uncertainty on the values in q in the same units. If ``method==mcmc``, 
			either q_err or q_cov must be passed. If ``method==leastsq`` and neither q_err nor 
			q_cov are passed, the absolute different between data points and fit is minimized. In 
			this case, the returned chi2 is in units of absolute difference, meaning its value 
			will depend on the units of q.
		q_cov: array_like
			Optional; the covariance matrix of the elements in q, as a 2-dimensional numpy array. 
			This array must have dimensions of q**2 and be in units of the square of the units of 
			q. If q_cov is passed, q_err is ignored since the diagonal elements of q_cov correspond 
			to q_err**2.
		method: str
			The fitting method; can be ``leastsq`` for a least-squares minimization of ``mcmc``
			for a Markov-Chain Monte Carlo.
		mask: array_like
			Optional; a numpy array of booleans that has the same length as the variables vector
			of the density profile class. Only variables where ``mask == True`` are varied in the
			fit, all others are kept constant. Important: this has to be a numpy array rather than
			a list.
		verbose: bool
			If true, output information about the fitting process.
		tolerance: float
			Only active when ``method==leastsq``. The accuracy to which the best-fit parameters
			are found.
		initial_step: array_like
			Only active when ``method==mcmc``. The MCMC samples ("walkers") are initially 
			distributed in a Gaussian around the initial guess. The width of the Gaussian is given
			by initial_step, either as an array of length N_par (giving the width of each Gaussian)
			or as a float number, in which case the width is set to initial_step times the initial
			value of the parameter.
		nwalkers: int
			Only active when ``method==mcmc``. The number of MCMC samplers that are run in parallel.
		random_seed: int
			Only active when ``method==mcmc``. If random_seed is not None, it is used to initialize
			the random number generator. This can be useful for reproducing results.
		convergence_step: int
			Only active when ``method==mcmc``. The convergence criteria are computed every
			convergence_step steps (and output is printed if ``verbose==True``). 
		converged_GR: float
			Only active when ``method==mcmc``. The maximum difference between different chains, 
			according to the Gelman-Rubin criterion. Once the GR indicator is lower than this 
			number in all parameters, the chain is ended. Setting this number too low leads to
			very long runtimes, but setting it too high can lead to inaccurate results.
		best_fit: str
			Only active when ``method==mcmc``. This parameter determines whether the ``mean`` or 
			``median`` value of the likelihood distribution is used as the output parameter set.
			
		Returns
		-------------------------------------------------------------------------------------------
		results: dict
			A dictionary bundling the various fit results. Regardless of the fitting method, the 
			dictionary always contains the following entries:
			
			``x``: array_like
				The best-fit result vector. If mask is passed, this vector only contains those 
				variables that were varied in the fit. 
			``q_fit``: array_like
				The fitted profile at the radii of the data points; has the same units as q and the 
				same dimensions as r.
			``chi2``: float
				The chi^2 of the best-fit profile. If a covariance matrix was passed, the 
				covariances are taken into account. If no uncertainty was passed at all, chi2 is
				in units of absolute difference, meaning its value will depend on the units of q.
			``chi2_ndof``: float
				The chi^2 per degree of freedom.
		
			If ``method==leastsq``, the dictionary additionally contains the following entries:
			
			``nfev``: int
				The number of function calls used in the fit.
			``x_err``: array_like
				An array of dimensions [2, nparams] which contains an estimate of the lower and 
				upper uncertainties on the fitted parameters. These uncertainties are computed 
				from the covariance matrix estimated by the fitter. Please note that this estimate
				does not exactly correspond to a 68% likelihood. In order to get more statistically
				meaningful uncertainties, please use the MCMC samples instead of least-squares. In
				some cases, the fitter fails to return a covariance matrix, in which case x_err is
				None.
				
			as well as the other entries returned by scipy.optimize.leastsq. If ``method==mcmc`,
			the dictionary contains the following entries:
			
			``x_initial``: array_like
				The initial positions of the walkers, in an array of dimensions [nwalkers, nparams].
			``chain_full``: array_like
				A numpy array of dimensions [n_independent_samples, nparams] with the parameters 
				at each step in the chain. In this thin chain, only every nth step is output, 
				where n is the auto-correlation time, meaning that the samples in this chain are 
				truly independent.
			``chain_thin``: array_like
				Like the thin chain, but including all steps. Thus, the samples in this chain are 
				not indepedent from each other. However, the full chain often gives better plotting 
				results.
			``R``: array_like
				A numpy array containing the GR indicator at each step when it was saved.
			``x_mean``: array_like
				The mean of the chain for each parameter; has length nparams.
			``x_median``: array_like
				The median of the chain for each parameter; has length nparams.
			``x_stddev``: array_like
				The standard deviation of the chain for each parameter; has length nparams.
			``x_percentiles``: array_like
				The lower and upper values of each parameter that contain a certain percentile of 
				the probability; has dimensions [n_percentages, 2, nparams] where the second 
				dimension contains the lower/upper values. 
		"""						
		
		# Check whether this profile has any parameters that can be optimized. If not, throw an
		# error.
		if self.N_par == 0:
			raise Exception('This profile has no parameters that can be fitted.')

		if verbose:
			Utilities.printLine()

		# Check whether the parameter mask makes sense
		if mask is None:
			mask = numpy.ones((self.N_par), numpy.bool)
		else:
			if len(mask) != self.N_par:
				msg = 'Mask has %d elements, expected %d.' % (len(mask), self.N_par)
				raise Exception(msg)
		N_par_fit = numpy.count_nonzero(mask)
		if N_par_fit < 1:
			raise Exception('The mask contains no True elements, meaning there are no parameters to vary.')
		if verbose:
			msg = 'Profile fit: Varying %d / %d parameters.' % (N_par_fit, self.N_par)
			print(msg)
		
		# Set the correct function to evaluate during the fitting process. We could just pass
		# quantity, but that would mean many evaluations of the dictionary entry.
		f = self.quantities[quantity]

		# Compute the inverse covariance matrix covinv. If no covariance has been passed, this 
		# matrix is diagonal, with covinv_ii = 1/sigma_i^2. If sigma has not been passed either,
		# the matrix is the identity matrix. 
		N = len(r)
		if q_cov is not None:
			covinv = numpy.linalg.inv(q_cov)
		elif q_err is not None:
			covinv = numpy.zeros((N, N), numpy.float)
			numpy.fill_diagonal(covinv, 1.0 / q_err**2)
		else:
			covinv = numpy.identity((N), numpy.float)

		# Perform the fit
		if method == 'mcmc':
			
			if q_cov is None and q_err is None:
				raise Exception('MCMC cannot be run without uncertainty vector or covariance matrix.')
			
			x, dict = self._fit_method_mcmc(r, q, f, covinv, mask, N_par_fit, verbose, \
				converged_GR, nwalkers, best_fit, initial_step, random_seed, convergence_step)
			
		elif method == 'leastsq':
		
			# If an analytical parameter derivative is implemented for this class, use it.
			deriv_name = '_fit_param_deriv_%s' % (quantity)
			if deriv_name in self.__class__.__dict__:
				fder = self.__class__.__dict__[deriv_name]
				if verbose:
					print(('Found analytical derivative function for quantity %s.' % (quantity)))
			else:
				fder = None
				if verbose:
					print(('Could not find analytical derivative function for quantity %s.' % (quantity)))

			# If the covariance matrix is given, things get a little complicated because we are not
			# just minimizing chi2 = C^-1 diff C, but have to return a vector of diffs for each 
			# data point. Thus, we decompose C^-1 into its eigenvalues and vectors:
			#
			# C^-1 = V^T Lambda V
			#
			# where V is a matrix of eigenvectors and Lambda is the matrix of eigenvalues. In the 
			# diff function, we want 
			#
			# diff -> V . diff / sigma
			#
			# Since Lambda has 1/sigma_i^2 on the diagonals, we create Q = V * root(Lambda) so that
			# 
			# diff -> Q . diff.
			#
			# If only sigma has been passed, Q has q/sigma_i on the diagonal.
			
			if q_cov is not None:
				Lambda, Q = numpy.linalg.eig(covinv)
				for i in range(N):
					Q[:, i] *= numpy.sqrt(Lambda[i])
			elif q_err is not None:
				Q = numpy.zeros((N, N), numpy.float)
				numpy.fill_diagonal(Q, 1.0 / q_err)
			else:
				Q = covinv
				
			x, dict = self._fit_method_leastsq(r, q, f, fder, Q, mask, N_par_fit, verbose, tolerance)
			
		else:
			msg = 'Unknown fitting method, %s.' % method
			raise Exception(msg)
		
		# Compute a few convenient outputs
		dict['x'] = x
		dict['q_fit'] = f(r)
		dict['chi2'] = self._fit_chi2(r, q, f, covinv)
		dict['chi2_ndof'] = dict['chi2'] / (len(r) - N_par_fit)
		
		if verbose:
			Utilities.printLine()

		return dict

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
		
		self.par_names = []
		self.opt_names = []
		HaloDensityProfile.__init__(self)
		
		self.rmin = numpy.min(r)
		self.rmax = numpy.max(r)
		self.r_guess = numpy.sqrt(self.rmin * self.rmax)
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
		
		self.par_names = ['rhos', 'rs']
		self.opt_names = []
		HaloDensityProfile.__init__(self)

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
		x, is_array = Utilities.getArray(xx)
		surfaceDensity = numpy.ones_like(x) * self.par['rhos'] * self.par['rs']
		
		# Solve separately for r < rs, r > rs, r = rs
		mask_rs = abs(x - 1.0) < 1E-4
		mask_lt = (x < 1.0) & (numpy.logical_not(mask_rs))
		mask_gt = (x > 1.0) & (numpy.logical_not(mask_rs))
		
		surfaceDensity[mask_rs] *= 2.0 / 3.0

		xi = x[mask_lt]		
		x2 = xi**2
		x2m1 = x2 - 1.0
		surfaceDensity[mask_lt] *= 2.0 / x2m1 \
			* (1.0 - 2.0 / numpy.sqrt(-x2m1) * numpy.arctanh(numpy.sqrt((1.0 - xi) / (xi + 1.0))))

		xi = x[mask_gt]		
		x2 = xi**2
		x2m1 = x2 - 1.0
		surfaceDensity[mask_gt] *= 2.0 / x2m1 \
			* (1.0 - 2.0 / numpy.sqrt(x2m1) * numpy.arctan(numpy.sqrt((xi - 1.0) / (xi + 1.0))))
			
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
	
		density_threshold = Halo.densityThreshold(z, mdef)
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

	def _fit_convertParams(self, p, mask):
		
		return numpy.log(p)

	###############################################################################################
	
	def _fit_convertParamsBack(self, p, mask):
		
		return numpy.exp(p)

	###############################################################################################

	# Return and array of d rho / d ln(rhos) and d rho / d ln(rs)
	
	def _fit_param_deriv_rho(self, r, mask, N_par_fit):

		x = self.getParameterArray()
		deriv = numpy.zeros((N_par_fit, len(r)), numpy.float)
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
	
		self.par_names = ['rhos', 'rs', 'alpha']
		self.opt_names = []
		HaloDensityProfile.__init__(self)

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

		R = Halo.M_to_R(M, z, mdef)
		self.par['rs'] = R / c
		
		if alpha is None:
			if mdef == 'vir':
				Mvir = M
			else:
				Mvir, _, _ = changeMassDefinition(M, c, z, mdef, 'vir')
			cosmo = Cosmology.getCurrent()
			nu_vir = cosmo.peakHeight(Mvir, z)
			alpha = 0.155 + 0.0095 * nu_vir**2
		
		self.par['alpha'] = alpha
		self.par['rhos'] = 1.0
		self._setMassTerms()
		M_unnorm = self.enclosedMass(R)
		self.par['rhos'] = M / M_unnorm
		
		return
	
	###############################################################################################

	# The enclosed mass for the Einasto profile is semi-analytical, in that it cna be expressed
	# in terms of Gamma functions. We pre-compute some factors to speed up the computation 
	# later.
	
	def _setMassTerms(self):

		self.mass_norm = numpy.pi * self.par['rhos'] * self.par['rs']**3 * 2.0**(2.0 - 3.0 / self.par['alpha']) \
			* self.par['alpha']**(-1.0 + 3.0 / self.par['alpha']) * numpy.exp(2.0 / self.par['alpha']) 
		self.gamma_3alpha = scipy.special.gamma(3.0 / self.par['alpha'])
		
		return
	
	###############################################################################################

	# We need to overwrite the setParameterArray function because the mass terms need to be 
	# updated when the user changes the parameters.
	
	def setParameterArray(self, pars, mask = None):
		
		HaloDensityProfile.setParameterArray(self, pars, mask = mask)
		self._setMassTerms()
		
		return

	###############################################################################################

	def density(self, r):
		
		rho = self.par['rhos'] * numpy.exp(-2.0 / self.par['alpha'] * \
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
		
		mass = self.mass_norm * self.gamma_3alpha * scipy.special.gammainc(3.0 / self.par['alpha'], \
								2.0 / self.par['alpha'] * (r / self.par['rs'])**self.par['alpha'])
		
		return mass
	
	###############################################################################################

	# When fitting the Einasto profile, use log(rhos), log(rs) and log(alpha)

	def _fit_convertParams(self, p, mask):
		
		return numpy.log(p)

	###############################################################################################
	
	def _fit_convertParamsBack(self, p, mask):
		
		return numpy.exp(p)

	###############################################################################################

	# Return and array of d rho / d ln(rhos) and d rho / d ln(rs)
	
	def _fit_param_deriv_rho(self, r, mask, N_par_fit):

		x = self.getParameterArray()
		deriv = numpy.zeros((N_par_fit, len(r)), numpy.float)
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
			deriv[counter] = rho_r * 2.0 / x[2] * rrs**x[2] * (1.0 - rrs**(-x[2]) - x[2] * numpy.log(rrs))
			counter += 1

		return deriv
	
###################################################################################################
# DIEMER & KRAVTSOV 2014 PROFILE
###################################################################################################

class DK14Profile(HaloDensityProfile):
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
	
	def __init__(self, rhos = None, rs = None, rt = None, alpha = None, beta = None, gamma = None, be = None, se = None, R200m = None, rho_m = None, \
				M = None, c = None, z = None, mdef = None, \
				selected = 'by_mass', Gamma = None, part = 'both', outer = 'powerlaw'):
	
		self.par_names = ['rhos', 'rs', 'rt', 'alpha', 'beta', 'gamma', 'be', 'se', 'R200m', 'rho_m']
		self.opt_names = ['part', 'selected', 'Gamma', 'outer']
		self.fit_log_mask = numpy.array([False, False, True, True, True, True, False, False, False, False])
		HaloDensityProfile.__init__(self)
		
		# The following parameters are not constants, they are temporarily changed by certain 
		# functions.
		self.accuracy_mass = 1E-4
		self.accuracy_radius = 1E-4

		self.opt['part'] = part
		self.opt['selected'] = selected
		self.opt['Gamma'] = Gamma
		self.opt['outer'] = outer

		if rhos is not None and rs is not None and rt is not None and alpha is not None and beta is not None and gamma is not None and be is not None and se is not None and R200m is not None and rho_m is not None:
			self.par['rhos'] = rhos
			self.par['rs'] = rs
			self.par['rt'] = rt
			self.par['alpha'] = alpha
			self.par['beta'] = beta
			self.par['gamma'] = gamma
			self.par['be'] = be
			self.par['se'] = se
			self.par['R200m'] = R200m
			self.par['rho_m'] = rho_m
		else:
			self.fundamentalParameters(M, c, z, mdef, be, se, \
						part = part, selected = selected, Gamma = Gamma, outer = outer)

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

	def fundamentalParameters(self, M, c, z, mdef, be, se, \
			part = 'both', selected = 'by_mass', Gamma = None, outer = 'powerlaw', \
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
		
		RTOL = 0.01
		MTOL = 0.01
		GUESS_TOL = 2.5
		self.accuracy_mass = MTOL
		self.accuracy_radius = RTOL
	
		# -----------------------------------------------------------------------------------------

		# Try a radius R200m, compute the resulting RDelta using the old RDelta as a starting guess
		
		def radius_diff(R200m, par2, Gamma, rho_target, R_target):
			
			self.par['R200m'] = R200m
			M200m = Halo.R_to_M(R200m, z, '200m')
			nu200m = cosmo.peakHeight(M200m, z)

			self.par['alpha'], self.par['beta'], self.par['gamma'], rt_R200m = \
				self.getFixedParameters(selected, nu200m = nu200m, z = z, Gamma = Gamma)
			self.par['rt'] = rt_R200m * R200m
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

		self.par['rs'] = R_target / c
		self.par['rho_m'] = cosmo.rho_m(z)
		self.par['be'] = be
		self.par['se'] = se
		self.opt['part'] = part
		
		if mdef == '200m':
			
			# The user has supplied M200m, the parameters follow directly from the input
			M200m = M
			self.par['R200m'] = Halo.M_to_R(M200m, z, '200m')
			nu200m = cosmo.peakHeight(M200m, z)
			self.par['alpha'], self.par['beta'], self.par['gamma'], rt_R200m = \
				self.getFixedParameters(selected, nu200m = nu200m, z = z, Gamma = Gamma)
			self.par['rt'] = rt_R200m * self.par['R200m']
			self.normalize(self.par['R200m'], M200m, mtol = MTOL)
			
		else:
			
			# The user has supplied some other mass definition, we need to iterate.
			_, R200m_guess, _ = changeMassDefinition(M, c, z, mdef, '200m')
			par2['RDelta'] = R_target

			# Iterate to find an M200m for which the desired mass is correct
			rho_target = Halo.densityThreshold(z, mdef)
			args = par2, Gamma, rho_target, R_target
			self.par['R200m'] = scipy.optimize.brentq(radius_diff, R200m_guess / 1.3, R200m_guess * 1.3, \
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

		part_true = self.opt['part']
		self.opt['part'] = 'inner'
		self.par['rhos'] = 1.0

		Mr_inner = self.enclosedMass(R)
		if part_true == 'both':
			self.opt['part'] = 'outer'
			Mr_outer = self.enclosedMass(R)
		elif part_true == 'inner':
			Mr_outer = 0.0
		else:
			msg = "Invalid value for part, %s." % (part_true)
			raise Exception(msg)
			
		self.par['rhos'] = (M - Mr_outer) / Mr_inner
		self.opt['part'] = part_true
		
		return

	###############################################################################################

	# The power-law outer profile is cut off at 1 / max_outer_prof to avoid a spurious density 
	# spike at very small radii if the slope of the power-law (se) is steep.
	
	def density(self, r):
		
		rho = r * 0.0

		rhos = self.par['rhos']
		rs = self.par['rs']
		rt = self.par['rt']
		alpha = self.par['alpha']
		beta = self.par['beta']
		gamma = self.par['gamma']
		
		if self.opt['part'] in ['inner', 'both']:
			inner = rhos * numpy.exp(-2.0 / alpha * ((r / rs) ** alpha - 1.0))
			fT = (1.0 + (r / rt) ** beta) ** (-gamma / beta)
			rho += inner * fT
		
		if self.opt['part'] in ['outer', 'both']:
			be = self.par['be']
			se = self.par['se']
			outer = self.par['rho_m'] * (1.0 + be / (self.max_outer_prof + (r / 5.0 / self.par['R200m'])**se))
			rho += outer
		
		return rho

	###############################################################################################
	
	def densityDerivativeLin(self, r):
		
		drho_dr = r * 0.0
		
		rhos = self.par['rhos']
		rs = self.par['rs']
		rt = self.par['rt']
		alpha = self.par['alpha']
		beta = self.par['beta']
		gamma = self.par['gamma']
				
		if self.opt['part'] in ['inner', 'both']:
			inner = rhos * numpy.exp(-2.0 / alpha * ((r / rs) ** alpha - 1.0))
			d_inner = inner * (-2.0 / rs) * (r / rs)**(alpha - 1.0)	
			fT = (1.0 + (r / rt) ** beta) ** (-gamma / beta)
			d_fT = (-gamma / beta) * (1.0 + (r / rt) ** beta) ** (-gamma / beta - 1.0) * \
				beta / rt * (r / rt) ** (beta - 1.0)
			drho_dr += inner * d_fT + d_inner * fT
	
		if self.opt['part'] in ['outer', 'both']:
			be = self.par['be']
			se = self.par['se']
			t1 = 1.0 / 5.0 / self.par['R200m']
			t2 = r * t1
			drho_dr += -self.par['rho_m'] * be * se * t1 * (self.max_outer_prof + t2**se)**-2 * t2**(se - 1.0)
		
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

		if self.opt['part'] in ['outer', 'both']:
			subtract = self.par['rho_m']
		else:
			subtract = 0.0

		def integrand(r, R):
			ret = 2.0 * r * (self.density(r) - subtract) / numpy.sqrt(r**2 - R**2)
			return ret

		r_use, is_array = Utilities.getArray(r)
		surfaceDensity = 0.0 * r_use
		for i in range(len(r_use)):	
			surfaceDensity[i], _ = scipy.integrate.quad(integrand, r_use[i], self.rmax, args = r_use[i], \
											epsrel = accuracy, limit = 1000)
			
		if not is_array:
			surfaceDensity = surfaceDensity[0]

		return surfaceDensity
	
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
	
		M200m = Halo.R_to_M(self.par['R200m'], z, mdef)
		_, R_guess, _ = changeMassDefinition(M200m, self.par['R200m'] / self.par['rs'], z, '200m', mdef)
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

	def _fit_convertParams(self, p, mask):

		p_fit = p
		log_mask = [self.fit_log_mask[mask]]
		p_fit[log_mask] = numpy.log(p_fit[log_mask])
		
		return p_fit

	###############################################################################################
	
	def _fit_convertParamsBack(self, p, mask):

		p_def = p
		log_mask = [self.fit_log_mask[mask]]
		p_def[log_mask] = numpy.exp(p_def[log_mask])
		
		return p_def

	###############################################################################################

	# Return and array of d rho / d ln(rhos) and d rho / d ln(rs)
	
	def _fit_param_deriv_rho_(self, r, mask, N_par_fit):

		x = self.getParameterArray()
		deriv = numpy.zeros((N_par_fit, len(r)), numpy.float)
		rho_r = self.density(r)
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
		term1 = 1.0 + rrt ** beta
		outer = x[9] * be * rro**-se
		
		# rho_s
		if mask[0]:
			deriv[counter] = rho_r / rhos
			counter += 1
		# rs
		if mask[1]:
			deriv[counter] = rho_r / rs * rrs**alpha * 2.0
			counter += 1
		# rt
		if mask[2]:
			deriv[counter] = rho_r * gamma / rt / term1 * rrt ** beta
			counter += 1
		# alpha
		if mask[3]:
			deriv[counter] = rho_r * 2.0 / alpha ** 2 * rrs ** alpha * (1.0 - rrs ** (-alpha) - alpha * numpy.log(rrs))
			counter += 1
		# beta
		if mask[4]:
			deriv[counter] = rho_r * (gamma * numpy.log(term1) / beta ** 2 - gamma * \
										rrt ** beta * numpy.log(rrt) / beta / term1)
			counter += 1
		# gamma
		if mask[5]:
			deriv[counter] = -rho_r * numpy.log(term1) / beta
			counter += 1
		# be
		if mask[6]:
			deriv[counter] = outer / be
			counter += 1
		# se
		if mask[7]:
			deriv[counter] = -outer * numpy.log(rro)
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
	
	M_i, is_array = Utilities.getArray(M_i)
	c_i, _ = Utilities.getArray(c_i)
	N = len(M_i)
	Rnew = numpy.zeros_like(M_i)
	cnew = numpy.zeros_like(M_i)

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
		if c is None:
			M200m, R200m, _ = changeMassDefinitionCModel(M, z, mdef, '200m', profile = profile)
		else:
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
		if c is None:
			M200m, _, _ = changeMassDefinitionCModel(M, z, mdef, '200m', profile = profile)
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
