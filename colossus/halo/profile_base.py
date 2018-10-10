###################################################################################################
#
# profile_base.py           (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
The majority of all profile functions are implemented in this base class. Please see 
:doc:`halo_profile` for a general introduction and :doc:`tutorials` for coding examples.
"""

###################################################################################################

import numpy as np
import scipy.misc
import scipy.optimize
import scipy.integrate
import scipy.interpolate
import abc
import collections
import six
import copy

from colossus import defaults
from colossus.utils import utilities
from colossus.utils import constants
from colossus.utils import mcmc
from colossus.halo import mass_so

###################################################################################################
# ABSTRACT BASE CLASS FOR HALO DENSITY PROFILES
###################################################################################################

@six.add_metaclass(abc.ABCMeta)
class HaloDensityProfile():
	"""
	Abstract base class for a halo density profile in physical units.
	
	A particular functional form for the density profile can be implemented by inheriting this 
	class and overwriting the constructor and :func:`density` method. In practice, a number of 
	other functions should also be overwritten for speed and convenience.
	
	Furthermore, this base class provides a general implementation of outer profile terms, 
	i.e. descriptions of the outer profile beyond the virial radius. Thus, these terms can be
	added to any derived density profile class without adding new code.

	Parameters
	-------------------------------------------------------------------------------------------
	outer_terms: list
		A list of OuterTerm objects to add to the density profile. 
	"""

	def __init__(self, outer_terms = []):
		
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
		
		# The parameters of the profile are stored in a dictionary. We separately store the number
		# of parameters for the inner profile because the total number may include additional 
		# parameters of the outer profile.
		self.par = collections.OrderedDict()
		self.N_par = len(self.par_names)
		self.N_par_inner = len(self.par_names)
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
		self.quantities['DeltaSigma'] = self.deltaSigma

		# Now we also add any parameters for the outer term(s)
		self._outer_terms = copy.copy(outer_terms)
		self.N_outer = len(self._outer_terms)
		self._outer_par_positions = []
		
		for i in range(self.N_outer):
			
			# Check which parameters are not yet existent in the parameters array
			added_parameters = []
			for p in self._outer_terms[i].term_par.keys():
				if not p in self.par:
					added_parameters.append(p)
			N_added = len(added_parameters)
			
			# Update the par/opt and par/opt-names dictionaries
			self.par.update(self._outer_terms[i].term_par)
			self.opt.update(self._outer_terms[i].term_opt)
		
			self.par_names.extend(self._outer_terms[i].term_par_names)
			self.opt_names.extend(self._outer_terms[i].term_opt_names)
			
			# Set pointers to the par and opt dictionaries of the profile class that owns the terms
			self._outer_terms[i].par = self.par
			self._outer_terms[i].opt = self.opt
			
			# Set pointer to the profile itself
			self._outer_terms[i].owner = self
		
			# For convenience, also store at what positions in the par array the outer parameters
			# were inserted.
			pp = np.zeros((N_added), np.int)
			key_list = list(self.par.keys())
			for j in range(N_added):
				pp[j] = key_list.index(added_parameters[j])
			self._outer_par_positions.append(pp)
			
		# We need to update the par and opt counters since the super constructor did not know 
		# about the parameters for the outer terms
		self.N_par = len(self.par)
		self.N_opt = len(self.opt)

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
			if len(mask) != self.N_par:
				msg = 'Received %d mask elements for %d parameters.' % \
					(np.count_nonzero(mask), self.N_par)
				raise Exception(msg)
				
			if len(pars) != np.count_nonzero(mask):
				msg = 'Received %d parameters and %d mask elements that are True.' % \
					(len(pars), np.count_nonzero(mask))
				raise Exception(msg)
			
			counter = 0
			for i in range(self.N_par):
				if mask[i]:
					self.par[self.par_names[i]] = pars[counter]
					counter += 1

		return

	###############################################################################################

	def update(self):
		"""
		Update the profile object after a change in parameters.
		
		If the parameters dictionary has been changed (e.g. by the user or during fitting), this 
		function must be called to ensure consistency within the profile object. This involves
		deleting any pre-computed quantities (e.g., tabulated enclosed masses) and re-computing
		profile properties that depend on the parameters.
		"""
		
		return

	###############################################################################################

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
			Density in physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`; has the same dimensions 
			as ``r``.
		"""

		return self.densityInner(r) + self.densityOuter(r)

	###############################################################################################

	@abc.abstractmethod
	def densityInner(self, r):
		"""
		Density of the inner profile as a function of radius.
		
		Abstract function which must be overwritten by child classes.
		
		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		density: array_like
			Density in physical :math:`M_{\odot} h^2 / kpc^3`; has the same dimensions 
			as ``r``.
		"""		
		
		return
	
	###############################################################################################
	
	def densityOuter(self, r):
		"""
		Density of the outer profile as a function of radius.
		
		This function should generally not be overwritten by child classes since it handles the 
		general case of adding up the contributions from all outer profile terms.
		
		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		density: array_like
			Density in physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`; has the same dimensions 
			as ``r``.
		"""		
				
		r_array, is_array = utilities.getArray(r)
		rho_outer = np.zeros((len(r_array)), np.float)
		for i in range(self.N_outer):
			rho_outer += self._outer_terms[i].density(r)
		if not is_array:
			rho_outer = rho_outer[0]
		
		return rho_outer

	###############################################################################################

	def densityDerivativeLin(self, r):
		"""
		The linear derivative of density, :math:`d \\rho / dr`. 

		This function should generally not be overwritten by child classes since it handles the 
		general case of adding up the contributions from the inner and outer terms.

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		derivative: array_like
			The linear derivative in physical :math:`M_{\odot} h / {\\rm kpc}^2`; has the same 
			dimensions as ``r``.
		"""
		
		return self.densityDerivativeLinInner(r) + self.densityDerivativeLinOuter(r)

	###############################################################################################

	def densityDerivativeLinInner(self, r):
		"""
		The linear derivative of the inner density, :math:`d \\rho_{\\rm inner} / dr`. 

		This function provides a numerical approximation to the derivative of the inner term, and
		should be overwritten by child classes if the derivative can be expressed analytically.
		
		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		derivative: array_like
			The linear derivative in physical :math:`M_{\odot} h / {\\rm kpc}^2`; has the same 
			dimensions as ``r``.
		"""
		
		r_use, is_array = utilities.getArray(r)
		rho_der = 0.0 * r_use
		for i in range(len(r_use)):	
			rho_der[i] = scipy.misc.derivative(self.densityInner, r_use[i], dx = 0.001, n = 1, order = 3)
		if not is_array:
			rho_der = rho_der[0]

		return rho_der

	###############################################################################################
	
	def densityDerivativeLinOuter(self, r):
		"""
		The linear derivative of the outer density, :math:`d \\rho_{\\rm outer} / dr`. 

		This function should generally not be overwritten by child classes since it handles the 
		general case of adding up the contributions from all outer profile terms.
		
		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		derivative: array_like
			The linear derivative in physical :math:`M_{\odot} h / {\\rm kpc}^2`; has the same 
			dimensions as ``r``.
		"""
		
		r_array, is_array = utilities.getArray(r)
		rho_der_outer = np.zeros((len(r_array)), np.float)
		for i in range(self.N_outer):
			rho_der_outer += self._outer_terms[i].densityDerivativeLin(r)
		if not is_array:
			rho_der_outer = rho_der_outer[0]
		
		return rho_der_outer

	###############################################################################################
	
	def densityDerivativeLog(self, r):
		"""
		The logarithmic derivative of density, :math:`d \log(\\rho) / d \log(r)`. 

		This function should generally not be overwritten by child classes since it handles the 
		general case of adding up the contributions from the inner and outer profile terms.

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		derivative: array_like
			The dimensionless logarithmic derivative; has the same dimensions as ``r``.
		"""

		drho_dr = self.densityDerivativeLin(r)
		rho = self.density(r)
		der = drho_dr * r / rho

		return der
		
	###############################################################################################
	
	def densityDerivativeLogInner(self, r):
		"""
		The logarithmic derivative of the inner density, :math:`d \log(\\rho_{\\rm inner}) / d \log(r)`. 

		This function evaluates the logarithmic derivative based on the linear derivative. If there
		is an analytic expression for the logarithmic derivative, child classes should overwrite 
		this function.

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		derivative: array_like
			The dimensionless logarithmic derivative; has the same dimensions as ``r``.
		"""
		
		drho_dr = self.densityDerivativeLinInner(r)
		rho = self.density(r)
		der = drho_dr * r / rho

		return der
		
	###############################################################################################
	
	def densityDerivativeLogOuter(self, r):
		"""
		The logarithmic derivative of the outer density, :math:`d \log(\\rho_{\\rm outer}) / d \log(r)`. 

		This function should generally not be overwritten by child classes since it handles the 
		general case of adding up the contributions from outer profile terms.

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		derivative: array_like
			The dimensionless logarithmic derivative; has the same dimensions as ``r``.
		"""		
		drho_dr = self.densityDerivativeLinOuter(r)
		rho = self.density(r)
		der = drho_dr * r / rho

		return der

	###############################################################################################
	
	# General function to integrate density.
	
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
	
	def enclosedMass(self, r, accuracy = defaults.HALO_PROFILE_ENCLOSED_MASS_ACCURACY):
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
			The mass enclosed within radius ``r``, in :math:`M_{\odot}/h`; has the same dimensions 
			as ``r``.
		"""		

		return self._enclosedMass(r, accuracy, self.density)

	###############################################################################################

	def enclosedMassInner(self, r, accuracy = defaults.HALO_PROFILE_ENCLOSED_MASS_ACCURACY):
		"""
		The mass enclosed within radius r due to the inner profile term.

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
		accuracy: float
			The minimum accuracy of the integration.
			
		Returns
		-------------------------------------------------------------------------------------------
		M: array_like
			The mass enclosed within radius ``r``, in :math:`M_{\odot}/h`; has the same dimensions 
			as ``r``.
		"""		

		return self._enclosedMass(r, accuracy, self.densityInner)

	###############################################################################################

	def enclosedMassOuter(self, r, accuracy = defaults.HALO_PROFILE_ENCLOSED_MASS_ACCURACY):
		"""
		The mass enclosed within radius r due to the outer profile term.

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
		accuracy: float
			The minimum accuracy of the integration.
			
		Returns
		-------------------------------------------------------------------------------------------
		M: array_like
			The mass enclosed within radius ``r``, in :math:`M_{\odot}/h`; has the same dimensions 
			as ``r``.
		"""		

		return self._enclosedMass(r, accuracy, self.densityOuter)
	
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
			The radius definition for the cut-off radius. See :doc:`halo_mass` for details.
		
		Returns
		-------------------------------------------------------------------------------------------
		pdf: array_like
			The probability for mass to lie within radius ``r``; has the same dimensions as ``r``.
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

	def _surfaceDensity(self, r, density_func, interpolate, accuracy, max_r_interpolate,
					max_r_integrate):
		
		# We evaluate the integral in log space, regardless of whether we are using interpolated
		# or exact densities. This leads to the r^2 factor rather than the usual r in the 
		# integrands.
		
		def integrand_interp(logr, R2, interp):
			r2 = np.exp(logr)**2
			ret = r2 * np.exp(interp(logr)) / np.sqrt(r2 - R2)
			return ret

		def integrand_exact(logr, R2, interp):
			r = np.exp(logr)
			r2 = r**2
			ret = r2 * density_func(r) / np.sqrt(r2 - R2)
			return ret
		
		# The upper limit of the integration is a bit tricky. If we are evaluating density exactly,
		# we can in principle integrate to infinity. However, in practice that leads to nan
		# results. Thus, we limit the upper radius to some very large number. 
		# 
		# If we are interpolating the density, we do not want to go to too large a radius in order 
		# to avoid a huge interpolation table. 
		#
		# In all cases, an upper limit is set by the rmax of the profile which may or may not be
		# infinity. Finally, we always check that the largest requested radius is not larger than
		# the upper integration limit divided by a safety factor of 10.
		
		if interpolate:
			log_min_r = np.log(np.min(r) * 0.99)
			log_max_r = np.log(min(self.rmax, max_r_interpolate))
		else:
			log_max_r = np.log(self.rmax)
			log_max_r = min(log_max_r, np.log(max_r_integrate))

		if np.max(r) > np.exp(log_max_r) / 10.0:
			msg = 'Cannot evaluate surface density for radius %.2e, must be smaller than %.2e. ' \
				% (np.max(r), np.exp(log_max_r) / 10.0)
			msg += 'You may want to turn the interpolate option off or change the integration limits.'
			raise Exception(msg)

		if interpolate:
			table_log_r = np.arange(log_min_r, log_max_r + 0.01, 0.1)
			rho = density_func(np.exp(table_log_r))
			if np.min(rho) <= 0.0:
				msg = 'Found negative or zero value in density, cannot create interpolation table. Try computing surface density with interpolate = False.'
				raise Exception(msg)
			table_log_rho = np.log(rho)
			interp = scipy.interpolate.InterpolatedUnivariateSpline(table_log_r, table_log_rho)
			integrand = integrand_interp
		else:
			interp = None
			integrand = integrand_exact
			
		r_use, is_array = utilities.getArray(r)
		surfaceDensity = 0.0 * r_use
		log_r_use = np.log(r_use)
		for i in range(len(r_use)):
			surfaceDensity[i], _ = scipy.integrate.quad(integrand, log_r_use[i], log_max_r, 
										args = (r_use[i]**2, interp), epsrel = accuracy, limit = 1000)
			surfaceDensity[i] *= 2.0

		if np.count_nonzero(surfaceDensity < 0.0) > 0:
			raise Exception('Found negative surface density. Please check the integration limits.')

		if not is_array:
			surfaceDensity = surfaceDensity[0]

		return surfaceDensity

	###############################################################################################
	
	# The surface density of the outer profile can be tricky, since some outer terms lead to a 
	# diverging integral. Thus, constant terms such as rho_m need to be ignored in this function.
	#
	# Note that this function returns the sum of the separately evaluated surface densities for
	# the inner and outer profiles, rather than integrating the combined inner and outer density.
	# This ensures that potentially overwritten inner or outer routines are used if available which
	# is typically much faster than integrating.
	
	def surfaceDensity(self, r,
					interpolate = True,
					accuracy = defaults.HALO_PROFILE_SURFACE_DENSITY_ACCURACY, 
					max_r_interpolate = defaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTERPOLATE,
					max_r_integrate = defaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTEGRATE):
		"""
		The projected surface density at radius r.
		
		The surface density is computed by projecting the 3D density along the line of sight,

		.. math::
			\\Sigma(R) = 2 \\int_R^{\\infty} \\frac{r \\rho(r)}{\\sqrt{r^2-R^2}} dr
		
		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
		interpolate: bool
			Use an interpolation table for density during the integration. This should make the
			evaluation somewhat faster, depending on how large the radius array is. 
		accuracy: float
			The minimum accuracy of the integration.
		max_r_interpolate: float
			The maximum radius in physical kpc/h to which the density profile is integrated when 
			using interpolating density.
		max_r_integrate: float
			The maximum radius in physical kpc/h to which the density profile is integrated when 
			using exact densities.
		
		Returns
		-------------------------------------------------------------------------------------------
		Sigma: array_like
			The surface density at radius ``r``, in physical :math:`M_{\odot} h/{\\rm kpc}^2`; has the 
			same dimensions as ``r``.
		"""
		
		sigma = self.surfaceDensityInner(r, interpolate = interpolate, accuracy = accuracy, 
						max_r_interpolate = max_r_interpolate, max_r_integrate = max_r_integrate)
		if self.N_outer > 0:
			sigma += self.surfaceDensityOuter(r, interpolate = interpolate, accuracy = accuracy,
						max_r_interpolate = max_r_interpolate, max_r_integrate = max_r_integrate)
		
		return sigma

	###############################################################################################

	def surfaceDensityInner(self, r,
					interpolate = True,
					accuracy = defaults.HALO_PROFILE_SURFACE_DENSITY_ACCURACY,
					max_r_interpolate = defaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTERPOLATE,
					max_r_integrate = defaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTEGRATE):
		"""
		The projected surface density at radius r due to the inner profile.

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
		interpolate: bool
			Use an interpolation table for density during the integration. This should make the
			evaluation somewhat faster, depending on how large the radius array is. 
		accuracy: float
			The minimum accuracy of the integration.
		max_r_interpolate: float
			The maximum radius in physical kpc/h to which the density profile is integrated when 
			using interpolating density.
		max_r_integrate: float
			The maximum radius in physical kpc/h to which the density profile is integrated when 
			using exact densities.
			
		Returns
		-------------------------------------------------------------------------------------------
		Sigma: array_like
			The surface density at radius ``r``, in physical :math:`M_{\odot} h/{\\rm kpc}^2`; has the 
			same dimensions as ``r``.
		"""
		
		return self._surfaceDensity(r, self.densityInner, interpolate = interpolate, accuracy = accuracy,
						max_r_interpolate = max_r_interpolate, max_r_integrate = max_r_integrate)

	###############################################################################################

	def surfaceDensityOuter(self, r, 
					interpolate = True,
					accuracy = defaults.HALO_PROFILE_SURFACE_DENSITY_ACCURACY, 
					max_r_interpolate = defaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTERPOLATE,
					max_r_integrate = defaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTEGRATE):
		"""
		The projected surface density at radius r due to the outer profile.
		
		This function checks whether there are explicit expressions for the surface density of 
		the outer profile terms available, and uses them if possible. Note that there are some
		outer terms whose surface density integrates to infinity, such as the mean density of the
		universe which is constant to infinitely large radii. 

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
		interpolate: bool
			Use an interpolation table for density during the integration. This should make the
			evaluation somewhat faster, depending on how large the radius array is. 
		accuracy: float
			The minimum accuracy of the integration.
		max_r_interpolate: float
			The maximum radius in physical kpc/h to which the density profile is integrated when 
			using interpolating density.
		max_r_integrate: float
			The maximum radius in physical kpc/h to which the density profile is integrated when 
			using exact densities.
			
		Returns
		-------------------------------------------------------------------------------------------
		Sigma: array_like
			The surface density at radius ``r``, in physical :math:`M_{\odot} h/{\\rm kpc}^2`; has the 
			same dimensions as ``r``.
		"""

		if utilities.isArray(r):	
			sigma_outer = np.zeros((len(r)), np.float)
		else:
			sigma_outer = 0.0
			
		for i in range(self.N_outer):
			if 'surfaceDensity' in self._outer_terms[i].__class__.__dict__:
				sigma_outer += self._outer_terms[i].surfaceDensity(r)
			else:
				sigma_outer += self._surfaceDensity(r, self._outer_terms[i].density, interpolate = interpolate, accuracy = accuracy,
						max_r_interpolate = max_r_interpolate, max_r_integrate = max_r_integrate)

		return sigma_outer

	###############################################################################################

	def _deltaSigma(self, r, surface_density_func, interpolate, interpolate_surface_density,
					accuracy, min_r_interpolate, max_r_interpolate, max_r_integrate):

		def integrand_interp(logr, interp):
			r2 = np.exp(logr)**2
			ret = r2 * np.exp(interp(logr))
			return ret

		def integrand_exact(logr, interp):
			r = np.exp(logr)
			ret = r**2 * surface_density_func(r, accuracy = accuracy, 
					interpolate = interpolate_surface_density, 
					max_r_interpolate = max_r_interpolate, max_r_integrate = max_r_integrate)
			return ret

		if np.max(r) > self.rmax:
			msg = 'Cannot evaluate DeltaSigma for radius %.2e, must be smaller than %.2e for this profile type.' \
				% (np.max(r), self.rmax)
			raise Exception(msg)

		log_min_r = np.log(min_r_interpolate)
		log_max_r = np.log(np.max(r) * 1.01)
		if interpolate:
			table_log_r = np.arange(log_min_r, log_max_r + 0.01, 0.1)
			table_log_Sigma = np.log(surface_density_func(np.exp(table_log_r), accuracy = accuracy, 
					interpolate = interpolate_surface_density, 
					max_r_interpolate = max_r_interpolate, max_r_integrate = max_r_integrate))
			interp = scipy.interpolate.InterpolatedUnivariateSpline(table_log_r, table_log_Sigma)
			integrand = integrand_interp
		else:
			interp = None
			integrand = integrand_exact

		r_use, is_array = utilities.getArray(r)
		deltaSigma = 0.0 * r_use
		for i in range(len(r_use)):
			deltaSigma[i], _ = scipy.integrate.quad(integrand, log_min_r, np.log(r_use[i]), 
										args = (interp), epsrel = accuracy, limit = 1000)
		
		if interpolate:
			Sigma = np.exp(interp(np.log(r_use)))
		else:
			Sigma = surface_density_func(r_use, accuracy = accuracy, 
					interpolate = interpolate_surface_density, max_r_interpolate = max_r_interpolate, 
					max_r_integrate = max_r_integrate)
			
		deltaSigma = deltaSigma * 2.0 / r_use**2 - Sigma
		
		if not is_array:
			deltaSigma = deltaSigma[0]

		return deltaSigma

	###############################################################################################

	def deltaSigma(self, r, 
					interpolate = True, interpolate_surface_density = True,
					accuracy = defaults.HALO_PROFILE_SURFACE_DENSITY_ACCURACY, 
					min_r_interpolate = defaults.HALO_PROFILE_DELTA_SIGMA_MIN_R_INTERPOLATE,
					max_r_interpolate = defaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTERPOLATE,
					max_r_integrate = defaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTEGRATE):
		"""
		The excess surface density at radius r.
		
		This quantity is useful in weak lensing studies, and is defined as 
		:math:`\\Delta\\Sigma(R) = \\Sigma(<R)-\\Sigma(R)` where :math:`\\Sigma(<R)` is the 
		averaged surface density within R weighted by area,
		
		.. math::
			\\Delta\\Sigma(R) =  \\frac{1}{\\pi R^2} \\int_0^{R} 2 \\pi r \\Sigma(r) dr - \\Sigma(R)
		
		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
		interpolate: bool
			Use an interpolation table for the surface density during the integration. This 
			can speed up the evaluation significantly, as the surface density can be expensive to
			evaluate.
		interpolate_surface_density: bool
			Use an interpolation table for density during the computation of the surface density.
			This should make the evaluation somewhat faster, but can fail for some density terms
			which are negative at particular radii. 
		accuracy: float
			The minimum accuracy of the integration (used both to compute the surface density and
			average it to get DeltaSigma).
		min_r_interpolate: float
			The minimum radius in physical kpc/h from which the surface density profile is 
			averaged.
		max_r_interpolate: float
			The maximum radius in physical kpc/h to which the density profile is integrated when 
			using interpolating density.
		max_r_integrate: float
			The maximum radius in physical kpc/h to which the density profile is integrated when 
			using exact densities.
			
		Returns
		-------------------------------------------------------------------------------------------
		DeltaSigma: array_like
			The excess surface density at radius ``r``, in physical 
			:math:`M_{\odot} h/{\\rm kpc}^2`; has the same dimensions as ``r``.
		"""
		
		deltaSigma = self.deltaSigmaInner(r, 
						interpolate = interpolate, interpolate_surface_density = interpolate_surface_density,
						accuracy = accuracy, min_r_interpolate = min_r_interpolate, 
						max_r_interpolate = max_r_interpolate, max_r_integrate = max_r_integrate)
		if self.N_outer > 0:
			deltaSigma += self.deltaSigmaOuter(r, 
						interpolate = interpolate, interpolate_surface_density = interpolate_surface_density,
						accuracy = accuracy, min_r_interpolate = min_r_interpolate, 
						max_r_interpolate = max_r_interpolate, max_r_integrate = max_r_integrate)
		
		return deltaSigma

	###############################################################################################

	def deltaSigmaInner(self, r, 
					interpolate = True, interpolate_surface_density = True,
					accuracy = defaults.HALO_PROFILE_SURFACE_DENSITY_ACCURACY, 
					min_r_interpolate = defaults.HALO_PROFILE_DELTA_SIGMA_MIN_R_INTERPOLATE,
					max_r_interpolate = defaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTERPOLATE,
					max_r_integrate = defaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTEGRATE):
		"""
		The excess surface density at radius r due to the inner profile.
		
		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
		interpolate: bool
			Use an interpolation table for the surface density during the integration. This 
			can speed up the evaluation significantly, as the surface density can be expensive to
			evaluate.
		interpolate_surface_density: bool
			Use an interpolation table for density during the computation of the surface density.
			This should make the evaluation somewhat faster, but can fail for some density terms
			which are negative at particular radii. 
		accuracy: float
			The minimum accuracy of the integration (used both to compute the surface density and
			average it to get DeltaSigma).
		min_r_interpolate: float
			The minimum radius in physical kpc/h from which the surface density profile is 
			averaged.
		max_r_interpolate: float
			The maximum radius in physical kpc/h to which the density profile is integrated when 
			using interpolating density.
		max_r_integrate: float
			The maximum radius in physical kpc/h to which the density profile is integrated when 
			using exact densities.
			
		Returns
		-------------------------------------------------------------------------------------------
		DeltaSigma: array_like
			The excess surface density at radius ``r``, in physical 
			:math:`M_{\odot} h/{\\rm kpc}^2`; has the same dimensions as ``r``.
		"""
				
		return self._deltaSigma(r, self.surfaceDensityInner, interpolate, 
							interpolate_surface_density, accuracy, min_r_interpolate,
							max_r_interpolate, max_r_integrate)	

	###############################################################################################

	def deltaSigmaOuter(self, r, 
					interpolate = True, interpolate_surface_density = True,
					accuracy = defaults.HALO_PROFILE_SURFACE_DENSITY_ACCURACY, 
					min_r_interpolate = defaults.HALO_PROFILE_DELTA_SIGMA_MIN_R_INTERPOLATE,
					max_r_interpolate = defaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTERPOLATE,
					max_r_integrate = defaults.HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTEGRATE):
		"""
		The excess surface density at radius r due to the outer profile.
		
		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
		interpolate: bool
			Use an interpolation table for the surface density during the integration. This 
			can speed up the evaluation significantly, as the surface density can be expensive to
			evaluate.
		interpolate_surface_density: bool
			Use an interpolation table for density during the computation of the surface density.
			This should make the evaluation somewhat faster, but can fail for some density terms
			which are negative at particular radii. 
		accuracy: float
			The minimum accuracy of the integration (used both to compute the surface density and
			average it to get DeltaSigma).
		min_r_interpolate: float
			The minimum radius in physical kpc/h from which the surface density profile is 
			averaged.
		max_r_interpolate: float
			The maximum radius in physical kpc/h to which the density profile is integrated when 
			using interpolating density.
		max_r_integrate: float
			The maximum radius in physical kpc/h to which the density profile is integrated when 
			using exact densities.
			
		Returns
		-------------------------------------------------------------------------------------------
		DeltaSigma: array_like
			The excess surface density at radius ``r``, in physical 
			:math:`M_{\odot} h/{\\rm kpc}^2`; has the same dimensions as ``r``.
		"""
					
		return self._deltaSigma(r, self.surfaceDensityOuter, interpolate,
							interpolate_surface_density, accuracy, min_r_interpolate,
							max_r_interpolate, max_r_integrate)	

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
			The circular velocity in km / s; has the same dimensions as ``r``.

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

	# Find a number by which the inner profile needs to be multiplied in order to give a particular
	# total enclosed mass at a particular radius.

	def _normalizeInner(self, R, M):
			
		Mr_inner = self.enclosedMassInner(R)
		Mr_outer = self.enclosedMassOuter(R)
		norm = (M - Mr_outer) / Mr_inner
		
		return norm

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
			See :doc:`halo_mass` for details.
			
		Returns
		-------------------------------------------------------------------------------------------
		R: float
			Spherical overdensity radius in physical kpc/h.

		See also
		-------------------------------------------------------------------------------------------
		MDelta: The spherical overdensity mass of a given mass definition.
		RMDelta: The spherical overdensity radius and mass of a given mass definition.
		"""		

		density_threshold = mass_so.densityThreshold(z, mdef)
		R = scipy.optimize.brentq(self._thresholdEquation, self.min_RDelta, self.max_RDelta, density_threshold)

		return R

	###############################################################################################

	def RMDelta(self, z, mdef):
		"""
		The spherical overdensity radius and mass of a given mass definition.
		
		This is a wrapper for the :func:`RDelta` and :func:`MDelta` functions which returns both 
		radius and mass.

		Parameters
		-------------------------------------------------------------------------------------------
		z: float
			Redshift
		mdef: str
			The mass definition for which the spherical overdensity mass is computed.
			See :doc:`halo_mass` for details.
			
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
		M = mass_so.R_to_M(R, z, mdef)
		
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
			See :doc:`halo_mass` for details.
			
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

	def _fitDiffFunction(self, x, r, q, f, df_inner, df_outer, Q, mask, N_par_fit, verbose):

		p = self._fitConvertParamsBack(x.copy(), mask)
		self.setParameterArray(p, mask = mask)
		
		# If very verbose, output the best-fit parameters at this iteration
		if verbose > 1:
			s = ''
			for i in range(N_par_fit):
				s += '%+10.2e' % (p[i])
			print(s)
		
		q_fit = f(r)
		q_diff = q_fit - q
		mf = np.dot(Q, q_diff)
		
		return mf

	###############################################################################################

	# Evaluate the derivative of the parameters, and multiply with the same matrix as in the diff
	# function. This function should only be called if fp is not None, i.e. if the analytical 
	# derivative is implemented.

	def _fitParamDerivHighlevel(self, x, r, q, f, df_inner, df_outer, Q, mask, N_par_fit, verbose):
		
		deriv = df_inner(self, r, mask, N_par_fit)

		# Add derivative of outer terms
		if self.N_outer > 0:
			for i in range(self.N_outer):
				if df_outer[i] is not None:
					f_outer = df_outer[i][0]
					pp = df_outer[i][1]
					mask_outer = df_outer[i][2]
					N_par_fit_outer = df_outer[i][3]
					deriv[pp, :] = f_outer(self._outer_terms[i], r, mask_outer, N_par_fit_outer)

		for j in range(N_par_fit):
			deriv[j] = np.dot(Q, deriv[j])
		
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

	def _fitMethodLeastsq(self, r, q, f, df_inner, df_outer, Q, mask, N_par_fit, verbose,
						tolerance, maxfev):
		
		# Prepare arguments
		if df_inner is None:
			deriv_func = None
		else:
			deriv_func = self._fitParamDerivHighlevel
		args = r, q, f, df_inner, df_outer, Q, mask, N_par_fit, verbose

		# Very verbose output
		if verbose > 1:
			s = ''
			for i in range(self.N_par):
				if mask[i]:			
					s += '%10s' % list(self.par.keys())[i]
			print(s)

		# Run the actual fit
		ini_guess = self._fitConvertParams(self.getParameterArray(mask = mask), mask)
		x_fit, cov, dict, fit_msg, err_code = scipy.optimize.leastsq(self._fitDiffFunction, 
							ini_guess, Dfun = deriv_func, col_deriv = 1, args = args, 
							full_output = 1, xtol = tolerance, maxfev = maxfev)
		
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
		tolerance = 1E-5, maxfev = 0,
		# Options specific to the MCMC initialization
		initial_step = 0.1, nwalkers = 100, random_seed = None,
		# Options specific to running the MCMC chain and its analysis
		convergence_step = 100, converged_GR = 0.01, best_fit = 'median', output_every_n = 100):
		"""
		Fit the density, mass, or surface density profile to a given set of data points.
		
		This function represents a general interface for finding the best-fit parameters of a 
		halo density profile given a set of data points. These points can represent a number of
		different physical quantities: ``quantity`` can either be density, enclosed mass, 
		surface density, or Delta Sigma (``rho``, ``M``, ``Sigma``, or ``DeltaSigma``). 
		
		The data points q at radii r can optionally have error bars, and the user can pass a full 
		covariance matrix. Please note that not passing any estimate of the uncertainty, i.e. 
		``q_err = None`` and ``q_cov = None``, can lead to very poor fit results: the fitter will 
		minimize the absolute difference between points, strongly favoring the high densities at 
		the center.
		
		There are two fundamental methods for performing the fit, a least-squares minimization 
		(``method = 'leastsq'``) and a Markov-Chain Monte Carlo (``method = 'mcmc'``). The MCMC 
		method has some specific options (see below). In either case, the current parameters of 
		the profile instance serve as an initial guess. Finally, the user can choose to vary only 
		a sub-set of the profile parameters through the ``mask`` parameter.
		
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
			The data to fit; can either be density in physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`, 
			enclosed mass in :math:`M_{\odot} /h`, or surface density in physical 
			:math:`M_{\odot} h/{\\rm kpc}^2`. Must have the same dimensions as r.
		quantity: str
			Indicates which quantity is given in as input in ``q``, can be ``rho``, ``M``, 
			``Sigma``, or ``DeltaSigma``.
		q_err: array_like
			Optional; the uncertainty on the values in ``q`` in the same units. If 
			``method == 'mcmc'``, either ``q_err`` or ``q_cov`` must be passed. If 
			``method == 'leastsq'`` and neither ``q_err`` nor ``q_cov`` are passed, the absolute 
			different between data points and fit is minimized. In this case, the returned 
			``chi2`` is in units of absolute difference, meaning its value will depend on the 
			units of ``q``.
		q_cov: array_like
			Optional; the covariance matrix of the elements in ``q``, as a 2-dimensional numpy 
			array. This array must have dimensions of q**2 and be in units of the square of the 
			units of ``q``. If ``q_cov`` is passed, ``q_err`` is ignored since the diagonal 
			elements of ``q_cov`` correspond to q_err**2.
		method: str
			The fitting method; can be ``leastsq`` for a least-squares minimization of ``mcmc``
			for a Markov-Chain Monte Carlo.
		mask: array_like
			Optional; a numpy array of booleans that has the same length as the variables vector
			of the density profile class. Only variables where ``mask == True`` are varied in the
			fit, all others are kept constant. Important: this has to be a numpy array rather than
			a list.
		verbose: bool / int
			If true, output information about the fitting process. The flag can also be set as a
			number, where 1 has the same effect as True, and 2 outputs large amounts of information
			such as the fit parameters at each iteration.
		tolerance: float
			Only active when ``method == 'leastsq'``. The accuracy to which the best-fit parameters
			are found.
		maxfev: int
			Only active when ``method == 'leastsq'``. The maximum number of function evaluations before
			the fit is aborted. If zero, the default value of the scipy leastsq function is used.
		initial_step: array_like
			Only active when ``method == 'mcmc'``. The MCMC samples ("walkers") are initially 
			distributed in a Gaussian around the initial guess. The width of the Gaussian is given
			by initial_step, either as an array of length ``N_par`` (giving the width of each 
			Gaussian) or as a float number, in which case the width is set to initial_step times the initial
			value of the parameter.
		nwalkers: int
			Only active when ``method == 'mcmc'``. The number of MCMC samplers that are run in parallel.
		random_seed: int
			Only active when ``method == 'mcmc'``. If random_seed is not None, it is used to initialize
			the random number generator. This can be useful for reproducing results.
		convergence_step: int
			Only active when ``method == 'mcmc'``. The convergence criteria are computed every
			convergence_step steps (and output is printed if ``verbose == True``). 
		converged_GR: float
			Only active when ``method == 'mcmc'``. The maximum difference between different chains, 
			according to the Gelman-Rubin criterion. Once the GR indicator is lower than this 
			number in all parameters, the chain is ended. Setting this number too low leads to
			very long runtimes, but setting it too high can lead to inaccurate results.
		best_fit: str
			Only active when ``method == 'mcmc'``. This parameter determines whether the ``mean`` or 
			``median`` value of the likelihood distribution is used as the output parameter set.
		output_every_n: int
			Only active when ``method == 'mcmc'``. This parameter determines how frequently the MCMC
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
				The fitted profile at the radii of the data points; has the same units as ``q``
				and the same dimensions as ``r``.
			``chi2``: float
				The chi^2 of the best-fit profile. If a covariance matrix was passed, the 
				covariances are taken into account. If no uncertainty was passed at all, ``chi2`` 
				is in units of absolute difference, meaning its value will depend on the units 
				of ``q``.
			``ndof``: int
				The number of degrees of freedom, i.e. the number of fitted data points minus 
				the number of free parameters.
			``chi2_ndof``: float
				The chi^2 per degree of freedom.
		
			If ``method == 'leastsq'``, the dictionary additionally contains the entries returned 
			by scipy.optimize.leastsq as well as the following:
			
			``nfev``: int
				The number of function calls used in the fit.
			``x_err``: array_like
				An array of dimensions ``[2, nparams]`` which contains an estimate of the lower and 
				upper uncertainties on the fitted parameters. These uncertainties are computed 
				from the covariance matrix estimated by the fitter. Please note that this estimate
				does not exactly correspond to a 68% likelihood. In order to get more statistically
				meaningful uncertainties, please use the MCMC samples instead of least-squares. In
				some cases, the fitter fails to return a covariance matrix, in which case 
				``x_err`` is ``None``.
				
			If ``method == 'mcmc'``, the dictionary contains the following entries:
			
			``x_initial``: array_like
				The initial positions of the walkers, in an array of dimensions 
				``[nwalkers, nparams]``.
			``chain_full``: array_like
				A numpy array of dimensions ``[n_independent_samples, nparams]`` with the parameters 
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
				The mean of the chain for each parameter; has length ``nparams``.
			``x_median``: array_like
				The median of the chain for each parameter; has length ``nparams``.
			``x_stddev``: array_like
				The standard deviation of the chain for each parameter; has length ``nparams``.
			``x_percentiles``: array_like
				The lower and upper values of each parameter that contain a certain percentile of 
				the probability; has dimensions ``[n_percentages, 2, nparams]`` where the second 
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
		
			# If an analytical parameter derivative is implemented for this class, use it. We need
			# to check both for the inner and outer terms.
			deriv_name = '_fitParamDeriv_%s' % (quantity)
			all_found = (deriv_name in self.__class__.__dict__)
			for i in range(self.N_outer):
				N_outer_par = np.count_nonzero(mask[self._outer_par_positions[i]])
				all_found = all_found and (N_outer_par == 0 or (deriv_name in self._outer_terms[i].__class__.__dict__))
			
			# If we have analytical derivatives for inner and outer terms, we pre-compute
			# function pointers and the indices of the outer parameters.
			if all_found:				
				df_inner = self.__class__.__dict__[deriv_name]
				df_outer = []
				
				for i in range(self.N_outer):
					N_outer_par = np.count_nonzero(mask[self._outer_par_positions[i]])
					par_pos = self._outer_par_positions[i][mask[self._outer_par_positions[i]]]
				
					if N_outer_par > 0:
						_df_outer = []
						_df_outer.append(self._outer_terms[i].__class__.__dict__[deriv_name])
						pp = np.zeros((N_outer_par), np.int)
						for j in range(N_outer_par):
							pp[j] = np.count_nonzero(mask[:par_pos[j]])
						_df_outer.append(pp)
						_df_outer.append(mask[self._outer_par_positions[i]])
						_df_outer.append(np.count_nonzero(mask[self._outer_par_positions[i]]))
						df_outer.append(_df_outer)
						
					else:
						df_outer.append(None)
						
				if verbose:
					print(('Found analytical derivative function for quantity %s.' % (quantity)))
			else:
				df_inner = None
				df_outer = None
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
				
			x, dict = self._fitMethodLeastsq(r, q, f, df_inner, df_outer,
									Q, mask, N_par_fit, verbose, tolerance, maxfev)
			
		else:
			msg = 'Unknown fitting method, %s.' % method
			raise Exception(msg)
		
		# Compute a few convenient outputs
		dict['x'] = x
		dict['q_fit'] = f(r)
		dict['chi2'] = self._fitChi2(r, q, f, covinv)
		dict['ndof'] = (len(r) - N_par_fit)
		dict['chi2_ndof'] = dict['chi2'] / dict['ndof']
		
		if verbose:
			print('chi2 / Ndof = %.1f / %d = %.2f' % (dict['chi2'], dict['ndof'], dict['chi2_ndof']))
			utilities.printLine()

		return dict
