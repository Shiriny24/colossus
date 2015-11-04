###################################################################################################
#
# profile_base.py           (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module contains abstract base classes for halo density profiles.

---------------------------------------------------------------------------------------------------
Module Reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
import scipy.misc
import scipy.optimize
import scipy.integrate
import abc
import collections
import six

from colossus.utils import utilities
from colossus.utils import constants
from colossus.utils import mcmc
from colossus.cosmology import cosmology
from colossus.halo import basics

###################################################################################################
# ABSTRACT BASE CLASS FOR HALO DENSITY PROFILES
###################################################################################################

@six.add_metaclass(abc.ABCMeta)
class HaloDensityProfile():
	"""
	Abstract base class for a halo density profile in physical units.
	
	This class contains a set of quantities that can be computed from halo density profiles in 
	general. In principle, a particular functional form of the profile can be implemented by 
	inheriting this class and overwriting the constructor and density method. In practice there 
	are often faster implementations for particular forms of the profile.
	"""

	def __init__(self):
		
		# The radial limits within which the profile is valid. These can be used as integration
		# limits for surface density, for example.
		self.rmin = 0.0
		self.rmax = np.inf
		
		# The radial limits within which we search for spherical overdensity radii. These limits 
		# can be set much tighter for better performance.
		self.min_RDelta = 0.001
		self.max_RDelta = 10000.0
		
		# For some functions, such as Vmax, we need an intial guess for a radius (in kpc/h).
		self.r_guess = 100.0
		
		# The parameters of the profile are stored in a dictionary
		self.par = collections.OrderedDict()
		self.N_par = len(self.par_names)
		for name in self.par_names:
			self.par[name] = None

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
		
		par = np.array(list(self.par.values()))
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

	def _densityDerivativeLin(self, r, density_function):
		
		r_use, is_array = utilities.getArray(r)
		density_der = 0.0 * r_use
		for i in range(len(r_use)):	
			density_der[i] = scipy.misc.derivative(density_function, r_use[i], dx = 0.001, n = 1, order = 3)
		if not is_array:
			density_der = density_der[0]
		
		return density_der

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
		
		return self._densityDerivativeLin(r, self.density)

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
			return np.log(self.density(np.exp(logr)))

		r_use, is_array = utilities.getArray(r)
		density_der = 0.0 * r_use
		for i in range(len(r_use)):	
			density_der[i] = scipy.misc.derivative(logRho, np.log(r_use[i]), dx = 0.0001, n = 1, order = 3)
		if not is_array:
			density_der = density_der[0]

		return density_der
		
	###############################################################################################
	
	def _enclosedMass(self, r, accuracy, density_function):
		
		def integrand(r):
			return density_function(r) * 4.0 * np.pi * r**2

		r_use, is_array = utilities.getArray(r)
		M = 0.0 * r_use
		for i in range(len(r_use)):	
			M[i], _ = scipy.integrate.quad(integrand, self.rmin, r_use[i], epsrel = accuracy)
		if not is_array:
			M = M[0]

		return M

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

		return self._enclosedMass(r, accuracy, self.density)

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
			ret = 2.0 * r * self.density(r) / np.sqrt(r**2 - R**2)
			return ret

		if np.max(r) >= self.rmax:
			msg = 'Cannot compute surface density at a radius (%.2e) greater than rmax (%.2e).' \
				% (np.max(r), self.rmax)
			raise Exception(msg)

		r_use, is_array = utilities.getArray(r)
		surfaceDensity = 0.0 * r_use
		for i in range(len(r_use)):	
			
			if r_use[i] >= self.rmax:
				msg = 'Cannot compute surface density for radius %.2e since rmax is %.2e.' % (r_use[i], self.rmax)
				raise Exception(msg)
			
			surfaceDensity[i], _ = scipy.integrate.quad(integrand, r_use[i], self.rmax, args = r_use[i],
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
		v = np.sqrt(constants.G * M / r)
		
		return v

	###############################################################################################

	# This helper function is used for Vmax where we need to minimize -vc.

	def _circularVelocityNegative(self, r):
		
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
		
		res = scipy.optimize.minimize(self._circularVelocityNegative, self.r_guess)
		rmax = res.x[0]
		vmax = self.circularVelocity(rmax)
		
		return vmax, rmax

	###############################################################################################

	# This equation is 0 when the enclosed density matches the given density_threshold, and is used 
	# when numerically determining spherical overdensity radii.
	
	def _thresholdEquation(self, r, density_threshold):
		
		diff = self.enclosedMass(r) / 4.0 / np.pi * 3.0 / r**3 - density_threshold
		
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

		density_threshold = basics.densityThreshold(z, mdef)
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
		M = basics.R_to_M(R, z, mdef)
		
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
	#
	# The p array passed to _fitConvertParams and _fitConvertParamsBack is a copy, meaning these
	# functions are allowed to manipulate it. 

	def _fitConvertParams(self, p, mask):
		
		return p

	###############################################################################################
	
	def _fitConvertParamsBack(self, p, mask):
		
		return p

	###############################################################################################

	# This function is evaluated before any derivatives etc. Thus, we set the new set of 
	# parameters here. For this purpose, we pass a copy of x so that the _fitConvertParamsBack 
	# does not manipulate the actual parameter vector x.
	#
	# Note that the matrix Q is the matrix that is dot-multiplied with the difference vector; this 
	# is not the same as the inverse covariance matrix.	

	def _fitDiffFunction(self, x, r, q, f, fder, Q, mask, N_par_fit, verbose):

		self.setParameterArray(self._fitConvertParamsBack(x.copy(), mask), mask = mask)
		q_fit = f(r)
		q_diff = q_fit - q
		mf = np.dot(Q, q_diff)
		#print('mf')
		#print(mf)
		
		return mf

	###############################################################################################

	# Evaluate the derivative of the parameters, and multiply with the same matrix as in the diff
	# function. This function should only be called if fp is not None, i.e. if the analytical 
	# derivative is implemented.

	def _fitParamDerivHighlevel(self, x, r, q, f, fder, Q, mask, N_par_fit, verbose):
		
		deriv = fder(self, r, mask, N_par_fit)		
		for j in range(N_par_fit):
			deriv[j] = np.dot(Q, deriv[j])
		#print(deriv)
		#print(Q)
		#exit()
		
		return deriv

	###############################################################################################

	def _fitChi2(self, r, q, f, covinv):

		q_model = f(r)
		diff = q_model - q
		chi2 = np.dot(np.dot(diff, covinv), diff)
		
		return chi2

	###############################################################################################
	
	# Evaluate the likelihood for a vector of parameter sets x. In this case, the vector is 
	# evaluated element-by-element, but the function is expected to handle a vector since this 
	# could be much faster for a simpler likelihood.
	
	def _fitLikelihood(self, x, r, q, f, covinv, mask):

		n_eval = len(x)
		res = np.zeros((n_eval), np.float)
		for i in range(n_eval):
			self.setParameterArray(x[i], mask = mask)
			res[i] = np.exp(-0.5 * self._fitChi2(r, q, f, covinv))
		
		return res

	###############################################################################################

	# Note that the MCMC fitter does NOT use the converted fitting parameters, but just the 
	# parameters themselves. Otherwise, interpreting the chain becomes very complicated.

	def _fitMethodMCMC(self, r, q, f, covinv, mask, N_par_fit, verbose,
				converged_GR, nwalkers, best_fit, initial_step, random_seed,
				convergence_step, output_every_n):
		
		x0 = self.getParameterArray(mask = mask)
		args = r, q, f, covinv, mask
		walkers = mcmc.initWalkers(x0, initial_step = initial_step, nwalkers = nwalkers, random_seed = random_seed)
		xi = np.reshape(walkers, (len(walkers[0]) * 2, len(walkers[0, 0])))
		chain_thin, chain_full, R = mcmc.runChain(self._fitLikelihood, walkers, convergence_step = convergence_step,
							args = args, converged_GR = converged_GR, verbose = verbose, output_every_n = output_every_n)
		mean, median, stddev, p = mcmc.analyzeChain(chain_thin, self.par_names, verbose = verbose)

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

	def _fitMethodLeastsq(self, r, q, f, fder, Q, mask, N_par_fit, verbose, tolerance):
		
		# Prepare arguments
		if fder is None:
			deriv_func = None
		else:
			deriv_func = self._fitParamDerivHighlevel	
		args = r, q, f, fder, Q, mask, N_par_fit, verbose

		# Run the actual fit
		ini_guess = self._fitConvertParams(self.getParameterArray(mask = mask), mask)
		x_fit, cov, dict, fit_msg, err_code = scipy.optimize.leastsq(self._fitDiffFunction, ini_guess,
							Dfun = deriv_func, col_deriv = 1, args = args, full_output = 1,
							xtol = tolerance)
		
		# Check the output
		if not err_code in [1, 2, 3, 4]:
			msg = 'Fitting failed, message: %s' % (fit_msg)
			raise Warning(msg)

		# Set the best-fit parameters
		x = self._fitConvertParamsBack(x_fit, mask)
		self.setParameterArray(x, mask = mask)

		# The fitter sometimes fails to derive a covariance matrix
		if cov is not None:
			
			# The covariance matrix is in relative units, i.e. needs to be multiplied with the 
			# residual chi2
			diff = self._fitDiffFunction(x_fit, *args)
			residual = np.sum(diff**2) / (len(r) - N_par_fit)
			cov *= residual

			# Derive an estimate of the uncertainty from the covariance matrix. We need to take into
			# account that cov refers to the fitting parameters which may not be the same as the 
			# standard profile parameters.
			sigma = np.sqrt(np.diag(cov))
			err = np.zeros((2, N_par_fit), np.float)
			err[0] = self._fitConvertParamsBack(x_fit - sigma, mask)
			err[1] = self._fitConvertParamsBack(x_fit + sigma, mask)

		else:
			
			msg = 'WARNING: Could not determine uncertainties on fitted parameters. Set all uncertainties to zero.'
			print(msg)
			err = np.zeros((2, N_par_fit), np.float)
			
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
		r, q, quantity, q_err = None, q_cov = None,
		# General fitting options: method, parameters to vary
		method = 'leastsq', mask = None, verbose = True,
		# Options specific to leastsq
		tolerance = 1E-5,
		# Options specific to the MCMC initialization
		initial_step = 0.1, nwalkers = 100, random_seed = None,
		# Options specific to running the MCMC chain and its analysis
		convergence_step = 100, converged_GR = 0.01, best_fit = 'median', output_every_n = 100):
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
		output_every_n: int
			Only active when ``method==mcmc``. This parameter determines how frequently the MCMC
			chain outputs information. Only effective if ``verbose == True``.
		
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
				
			as well as the other entries returned by scipy.optimize.leastsq. If ``method==mcmc``,
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
			utilities.printLine()

		# Check whether the parameter mask makes sense
		if mask is None:
			mask = np.ones((self.N_par), np.bool)
		else:
			if len(mask) != self.N_par:
				msg = 'Mask has %d elements, expected %d.' % (len(mask), self.N_par)
				raise Exception(msg)
		N_par_fit = np.count_nonzero(mask)
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
			covinv = np.linalg.inv(q_cov)
		elif q_err is not None:
			covinv = np.zeros((N, N), np.float)
			np.fill_diagonal(covinv, 1.0 / q_err**2)
		else:
			covinv = np.identity((N), np.float)

		# Perform the fit
		if method == 'mcmc':
			
			if q_cov is None and q_err is None:
				raise Exception('MCMC cannot be run without uncertainty vector or covariance matrix.')
			
			x, dict = self._fitMethodMCMC(r, q, f, covinv, mask, N_par_fit, verbose,
				converged_GR, nwalkers, best_fit, initial_step, random_seed, convergence_step, output_every_n)
			
		elif method == 'leastsq':
		
			# If an analytical parameter derivative is implemented for this class, use it.
			deriv_name = '_fitParamDeriv_%s' % (quantity)
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
				Lambda, Q = np.linalg.eig(covinv)
				for i in range(N):
					Q[:, i] *= np.sqrt(Lambda[i])
				Q = Q.T
			elif q_err is not None:
				Q = np.zeros((N, N), np.float)
				np.fill_diagonal(Q, 1.0 / q_err)
			else:
				Q = covinv
				
			x, dict = self._fitMethodLeastsq(r, q, f, fder, Q, mask, N_par_fit, verbose, tolerance)
			
		else:
			msg = 'Unknown fitting method, %s.' % method
			raise Exception(msg)
		
		# Compute a few convenient outputs
		dict['x'] = x
		dict['q_fit'] = f(r)
		dict['chi2'] = self._fitChi2(r, q, f, covinv)
		dict['chi2_ndof'] = dict['chi2'] / (len(r) - N_par_fit)
		
		if verbose:
			utilities.printLine()

		return dict

###################################################################################################
# ABSTRACT BASE CLASS FOR 2-HALO TERMS
###################################################################################################

# The parameter and option names in this class are up to the user. This enables two kinds of 
# behavior: first, potentially conflicting standard names can be changed, and second, the user can
# "point" to an already existing parameter.

@six.add_metaclass(abc.ABCMeta)
class OuterTerm():
	
	def __init__(self, par_array, opt_array, par_names, opt_names):
		
		if len(par_array) != len(par_names):
			msg = 'Arrays with parameters and parameter names must have the same length (%d, %d).' % \
				(len(par_array), len(par_names))
			raise Exception(msg)
		
		if len(opt_array) != len(opt_names):
			msg = 'Arrays with options and option names must have the same length (%d, %d).' % \
				(len(opt_array), len(opt_names))
			raise Exception(msg)

		self.par_names = par_names
		self.opt_names = opt_names

		# The parameters of the profile are stored in a dictionary
		self.term_par = collections.OrderedDict()
		self.N_par = len(self.par_names)
		for i in range(self.N_par):
			self.term_par[self.par_names[i]] = par_array[i]

		# Additionally to the numerical parameters, there can be options
		self.term_opt = collections.OrderedDict()
		self.N_opt = len(self.opt_names)
		for i in range(self.N_opt):
			self.term_opt[self.opt_names[i]] = opt_array[i]
		
		return

	###############################################################################################

	# Return the density of at an array r

	@abc.abstractmethod
	def _density(self, r):
		return

	###############################################################################################

	def density(self, r):

		r_array, is_array = utilities.getArray(r)
		rho = self._density(r_array)
		if not is_array:
			rho = rho[0]
		
		return rho

	###############################################################################################

	def densityDerivativeLin(self, r):
		
		r_use, is_array = utilities.getArray(r)
		density_der = 0.0 * r_use
		for i in range(len(r_use)):	
			density_der[i] = scipy.misc.derivative(self.density, r_use[i], dx = 0.001, n = 1, order = 3)
		if not is_array:
			density_der = density_der[0]
			
		return density_der
		
###################################################################################################
# 2-HALO TERM: MEAN DENSITY
###################################################################################################

class OuterTermRhoMean(OuterTerm):
	
	def __init__(self, z):
		
		if z is None:
			raise Exception('Redshift cannot be None.')
		
		OuterTerm.__init__(self, [], [z], [], ['z'])
		
		self.z = z
		cosmo = cosmology.getCurrent()
		self.rho_m = cosmo.rho_m(z)
		
		return

	###############################################################################################

	def _density(self, r):
		
		return np.ones((len(r)), np.float) * self.rho_m

###################################################################################################
# 2-HALO TERM: POWER LAW
###################################################################################################

# This class implements a power-law outer profile with a free normalization and slope, but a fixed
# pivot radius. Note that the slope is inverted, i.e. a more positive slope means a steeper 
# profile.
#
# The max_rho factor provides a convenient upper limit, since steep power-law profiles can lead to
# a very high, unphysical density contribution at the halo center.

class OuterTermPowerLaw(OuterTerm):
	
	def __init__(self, norm, slope, pivot, pivot_factor, max_rho, z,
				norm_name = 'norm', slope_name = 'slope', 
				pivot_name = 'pivot', pivot_factor_name = 'pivot_factor', 
				max_rho_name = 'pl_max_rho', z_name = 'z'):

		if norm is None:
			raise Exception('Normalization of power law cannot be None.')
		if slope is None:
			raise Exception('Slope of power law cannot be None.')
		if pivot is None:
			raise Exception('Pivot of power law cannot be None.')
		if pivot_factor is None:
			raise Exception('Pivot factor of power law cannot be None.')
		if max_rho is None:
			raise Exception('Maximum of power law cannot be None.')
		if z is None:
			raise Exception('Redshift of power law cannot be None.')
		
		OuterTerm.__init__(self, [norm, slope], [pivot, pivot_factor, max_rho, z],
						[norm_name, slope_name], [pivot_name, pivot_factor_name, max_rho_name, z_name])

		return

	###############################################################################################

	def _getParameters(self):

		r_pivot_id = self.opt[self.opt_names[0]]
		if r_pivot_id in self.par:
			r_pivot = self.par[r_pivot_id]
		elif r_pivot_id in self.opt:
			r_pivot = self.opt[r_pivot_id]
		else:
			msg = 'Could not find the parameter or option %s.' % (r_pivot_id)
			raise Exception(msg)

		norm = self.par[self.par_names[0]]
		slope = self.par[self.par_names[1]]
		r_pivot *= self.opt[self.opt_names[1]]
		max_rho = self.opt[self.opt_names[2]]
		z = self.opt[self.opt_names[3]]
		rho_m = cosmology.getCurrent().rho_m(z)
		
		return norm, slope, r_pivot, max_rho, rho_m

	###############################################################################################

	def _density(self, r):
		
		norm, slope, r_pivot, max_rho, rho_m = self._getParameters()
		rho = rho_m * norm * (1.0 / max_rho + r / r_pivot)**-slope

		return rho

	###############################################################################################

	def densityDerivativeLin(self, r):

		norm, slope, r_pivot, max_rho, rho_m = self._getParameters()
		t1 = 1.0 / r_pivot
		t2 = r * t1
		drho_dr = -rho_m * norm * slope * t1 * (1.0 / max_rho + t2**slope)**-2 * t2**(slope - 1.0)

		return drho_dr
	
###################################################################################################
# ABSTRACT BASE CLASS FOR HALO DENSITY PROFILES WITH A DESCRIPTION OF THE 2-HALO TERM
###################################################################################################

@six.add_metaclass(abc.ABCMeta)
class HaloDensityProfileWithOuter(HaloDensityProfile):
	"""
	Abstract base class for a halo density profile composed of both a 1-halo (inner) and a 2-halo 
	(outer) term.
	
	This class extends HaloDensityProfile by a set of additional terms that describe the outer 
	profile, or 2-halo term. The user can choose an arbitrary number of terms to add, for example
	the mean density, a power law, and an estimate of the 2-halo term based on the matter-matter
	correlation function.
	
	Derived classes must set the outer_terms variable with a list of OuterTerm classes, as well as
	the usual par_names and opt_names variables.
	"""
	
	def __init__(self):
		
		# In the constructor of HaloDensityProfile, the par and opt dictionaries are initialized
		HaloDensityProfile.__init__(self)
		
		# Now we also add any parameters for the 2-halo term(s)
		self.N_outer = len(self.outer_terms)
		for i in range(self.N_outer):
			self.par.update(self.outer_terms[i].term_par)
			self.opt.update(self.outer_terms[i].term_opt)
			
			# Set pointers to the par and opt dictionaries of the profile class that owns the terms
			self.outer_terms[i].par = self.par
			self.outer_terms[i].opt = self.opt
		
		return

	###############################################################################################

	# Add 1 halo and 2 halo densities

	def density(self, r):
		
		return self.densityInner(r) + self.densityOuter(r)

	###############################################################################################

	@abc.abstractmethod
	def densityInner(self, r):
		
		return
	
	###############################################################################################

	# Add the contributions from all 2h terms
	
	def densityOuter(self, r):
		
		r_array, is_array = utilities.getArray(r)
		rho_outer = np.zeros((len(r_array)), np.float)
		for i in range(self.N_outer):
			rho_outer += self.outer_terms[i].density(r)
		if not is_array:
			rho_outer = rho_outer[0]
		
		return rho_outer

	###############################################################################################

	def densityDerivativeLin(self, r):
		
		return self.densityDerivativeLinInner(r) + self.densityDerivativeLinOuter(r)

	###############################################################################################

	def densityDerivativeLinInner(self, r):

		return self._densityDerivativeLin(r, self.densityInner)

	###############################################################################################

	# Add the contributions from all 2h terms
	
	def densityDerivativeLinOuter(self, r):
		
		r_array, is_array = utilities.getArray(r)
		rho_der_outer = np.zeros((len(r_array)), np.float)
		for i in range(self.N_outer):
			rho_der_outer += self.outer_terms[i].densityDerivativeLin(r)
		if not is_array:
			rho_der_outer = rho_der_outer[0]
		
		return rho_der_outer

	###############################################################################################
	
	def enclosedMass(self, r, accuracy = 1E-6):

		return self._enclosedMass(r, accuracy, self.density)

	###############################################################################################

	def enclosedMassInner(self, r, accuracy = 1E-6):

		return self._enclosedMass(r, accuracy, self.densityInner)

	###############################################################################################

	def enclosedMassOuter(self, r, accuracy = 1E-6):

		return self._enclosedMass(r, accuracy, self.densityOuter)

	###############################################################################################

	# Find a number by which the inner profile needs to be multiplied in order to give a particular
	# total enclosed mass at a particular radius.

	def normalizeInner(self, R, M):
		
		Mr_inner = self.enclosedMassInner(R)
		Mr_outer = self.enclosedMassOuter(R)
		norm = (M - Mr_outer) / Mr_inner
		
		return norm
	
###################################################################################################
