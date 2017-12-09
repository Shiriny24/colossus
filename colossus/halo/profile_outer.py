###################################################################################################
#
# profile_outer.py           (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module implements terms that describe the outer halo density profile. Specific terms are 
derived from the :class:`OuterTerm` base class. The :doc:`tutorials` contain more detailed code
examples.

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

Let us create an NFW profile, but add a description of the outer profile using the matter-matter 
correlation function::
    
    from colossus.halo import profile_nfw
    from colossus.halo import profile_outer

    outer_term = profile_outer.OuterTermCorrelationFunction(z = 0.0, bias = 2.0)
    profile = profile_nfw.NFWProfile(M = 1E12, mdef = 'vir', z = 0.0, c = 10.0, outer_terms = [outer_term])

The ``outer_terms`` keyword can be used with any class derived from 
:class:`~halo.profile_base.HaloDensityProfile`.

---------------------------------------------------------------------------------------------------
Models for the outer term
---------------------------------------------------------------------------------------------------

.. table::
	:widths: auto

	======================================= =======================================================
	Class                                   Explanation
	======================================= =======================================================
	:class:`OuterTermMeanDensity`           The mean matter density of the universe  
	:class:`OuterTermCorrelationFunction`   A term based on the matter-matter correlation      
	:class:`OuterTermPowerLaw`              A power-law profile
	======================================= =======================================================

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
import scipy.misc
import abc
import collections
import six

from colossus.utils import utilities
from colossus import defaults
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.lss import bias as halo_bias

###################################################################################################
# ABSTRACT BASE CLASS FOR OUTER PROFILE TERMS
###################################################################################################

@six.add_metaclass(abc.ABCMeta)
class OuterTerm():
	"""
	Base class for outer profile terms.
	
	In Colossus, the density profile is assumed to consist of an inner term (i.e., a description
	of the 1-halo term, such as an NFW profile) as well as one or multiple outer terms which are 
	added to the inner term. 
	
	These outer terms must be derived from the OuterTerm base class, and must at least overwrite 
	the _density() routine. The derived outer terms must also, in their constructor, call the 
	constructor of this class with the parameters specified below. The user interface to such
	derived classes will, in general, be much simpler than the constructor of this super class.
	
	The user can then add one or multiple outer terms to a density profile by calling its 
	constructor and passing a list of OuterTerm objects in the ``outer_terms`` argument (see the 
	documentation of :class:`~halo.profile_base.HaloDensityProfile`). Once the profile has been 
	created, the outer terms themselves cannot be added or removed. Their parameters, however, can
	be modified in the same ways as the parameters of the inner profile.

	Parameters
	-----------------------------------------------------------------------------------------------
	par_array: list
		A list of parameter values for the outer term.
	opt_array: list
		A list of option values for the outer term.
	par_names: list
		A list of parameter names, corresponding to the values passed in par_array. If these names
		overlap with already existing parameters, the parameter is NOT added to the profile. 
		Instead, the value of the existing parameter will be used. This behavior can be useful when
		outer profile terms rely on parameters or options of the inner profile.
	opt_names:
		A list of option names, corresponding to the values passed in opt_array.
	"""
	
	def __init__(self, par_array, opt_array, par_names, opt_names):
		
		if len(par_array) != len(par_names):
			msg = 'Arrays with parameters and parameter names must have the same length (%d, %d).' % \
				(len(par_array), len(par_names))
			raise Exception(msg)
		
		if len(opt_array) != len(opt_names):
			msg = 'Arrays with options and option names must have the same length (%d, %d).' % \
				(len(opt_array), len(opt_names))
			raise Exception(msg)

		self.term_par_names = par_names
		self.term_opt_names = opt_names

		# The parameters of the profile are stored in a dictionary
		self.term_par = collections.OrderedDict()
		self.N_par = len(self.term_par_names)
		for i in range(self.N_par):
			self.term_par[self.term_par_names[i]] = par_array[i]

		# Additionally to the numerical parameters, there can be options
		self.term_opt = collections.OrderedDict()
		self.N_opt = len(self.term_opt_names)
		for i in range(self.N_opt):
			self.term_opt[self.term_opt_names[i]] = opt_array[i]

		return

	###############################################################################################

	# Return the density of at an array r

	@abc.abstractmethod
	def _density(self, r):
		"""
		The density due to the outer term as a function of radius.
		
		Abstract function which must be overwritten by child classes.
		
		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; guaranteed to be an array, even if of length 1.

		Returns
		-------------------------------------------------------------------------------------------
		density: array_like
			Density in physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`; has the same dimensions 
			as ``r``.
		"""

		return

	###############################################################################################

	def density(self, r):
		"""
		The density due to the outer term as a function of radius.
		
		This function provides a convenient wrapper around _density() by ensuring that the radius
		values passed are a numpy array. This function should generally not be overwritten by 
		child classes.
		
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
		rho = self._density(r_array)
		if not is_array:
			rho = rho[0]
		
		return rho

	###############################################################################################

	def densityDerivativeLin(self, r):
		"""
		The linear derivative of the density due to the outer term, :math:`d \\rho / dr`. 

		This function should be overwritten by child classes if there is an analytic, faster 
		expression for the derivative.

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
		density_der = 0.0 * r_use
		for i in range(len(r_use)):	
			density_der[i] = scipy.misc.derivative(self.density, r_use[i], dx = 0.001, n = 1, order = 3)
		if not is_array:
			density_der = density_der[0]
			
		return density_der

###################################################################################################
# OUTER TERM: MEAN DENSITY
###################################################################################################

class OuterTermMeanDensity(OuterTerm):
	"""
	An outer term that adds the mean matter density of the universe to a density profile.
	
	This is perhaps the simplest outer term one can imagine. The only parameter is the redshift at
	which the halo density profile is modeled. Note that this term is cosmology-dependent, meaning
	that a cosmology has to be set before the constructor is called.
	
	Furthermore, note that a constant term such as this one means that the surface density cannot
	be evaluated any more, since the integral over density will diverge. If the surface density
	is to be evaluated, one should always remove constant outer terms from the profile. This 
	class does overwrite the surface density function and issues a warning if it is called.
	
	In this implementation, the redshift is added to the profile options rather than parameters,
	meaning that it cannot be varied in a fit.

	Parameters
	-----------------------------------------------------------------------------------------------
	z: float
		The redshift at which the profile is modeled.
	z_name: str
		The internal name of the redshift option. If this name is set to an already existing
		profile option, the redshift is set to this other profile option, and thus not an
		independent option any more.
	"""

	def __init__(self, z = None, z_name = 'z'):
		
		if z is None:
			raise Exception('Redshift cannot be None.')
		
		OuterTerm.__init__(self, [], [z], [], [z_name])

		return

	###############################################################################################

	def _getParameters(self):

		z = self.opt[self.term_opt_names[0]]
		
		return z

	###############################################################################################

	def _density(self, r):
		
		z = self._getParameters()
		cosmo = cosmology.getCurrent()
		rho = np.ones((len(r)), np.float) * cosmo.rho_m(z)
		
		return rho

	###############################################################################################

	def surfaceDensity(self, r):
		"""
		The projected surface density at radius r due to the outer profile.

		This function is overwritten for the mean density outer profile because it is ill-defined:
		as the mean density is constant out to infinite radii, the line-of-sight integral 
		diverges. In principle, this function could just return zero in order to ignore this 
		spurious contribution, but that causes an inconsistency between the 3D (rho) and 2D 
		(Sigma) density profiles.

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
			
		Returns
		-------------------------------------------------------------------------------------------
		Sigma: array_like
			An array of zeros.
		"""
		
		print('WARNING: Ignoring surface density of mean-density outer profile. This term should be removed before evaluating the surface density.')
		
		return r * 0.0

###################################################################################################
# OUTER TERM: HALO-MATTER CORRELATION FUNCTION
###################################################################################################

class OuterTermCorrelationFunction(OuterTerm):
	"""
	An outer term that adds an estimate based on the matter-matter correlation function.

	On large scales, we can model the 2-halo term, i.e., the excess density due to neighboring
	halos, by assuming a linear bias. In that case, the halo-matter correlation function is a 
	multiple of the matter-matter correlation function, independent of radius:
	
	.. math::
		\\rho(r) = \\rho_{\\rm m} \\times b(\\nu) \\times \\xi_{\\rm mm}
		
	where :math:`b(\\nu)` is called the halo bias. Note that this implementation does not add the 
	constant due to the mean density of the universe which is sometimes included. If desired, this
	contribution can be added with the :class:`OuterTermMeanDensity` term.
	
	The bias should be initialized to a physically motivated value. This value can be calculated
	self-consistently, but this needs to be done iteratively because the bias depends on mass, 
	which circularly depends on the value of bias due to the inclusion of this outer term. Thus, 
	creating such a profile can be very slow. See the :mod:`~lss.bias` module for models of the
	bias as a function of halo mass.

	In this implementation, the redshift is added to the profile options rather than parameters,
	meaning it cannot be varied in a fit. The halo bias (i.e., the normalization of this outer 
	term) is a parameter though and can be varied in a fit. 
	
	Note that this outer term can be evaluated at radii outside the range where the correlation
	function is defined by the cosmology module without throwing an error or warning. In such 
	cases, the return value is the correlation function at the min/max radius. This behavior is
	convenient when initializing profiles etc, where the outer term may be insignificant at 
	some radii. However, when integrating this outer term (e.g., when evaluating the surface 
	density), care must be taken to set the correct integration limits. See the documentation of 
	the correlation function in the cosmology module for more information.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	z: float
		The redshift at which the profile is modeled.
	derive_bias_from: str or None
		If ``None``, the bias is passed through the bias parameter and added to the profile 
		parameters. If ``derive_bias_from`` is a string, it must correspond to a profile parameter 
		or option. Furthermore, this parameter or option must represent a valid spherical overdensity 
		mass or radius such as ``'R200m'`` or ``'Mvir'`` from which the bias can be computed. If so, 
		the bias is updated from that quantity every time the density is computed. 
	bias: float
		The halo bias.
	z_name: str
		The internal name of the redshift option. If this name is set to an already existing
		profile option, the redshift is set to this other profile option, and thus not an
		independent option any more.
	bias_name: str
		The internal name of the bias parameter. If this name is set to an already existing
		profile parameter, the bias is set to this other profile parameter, and thus not an
		independent parameter any more.
	"""

	def __init__(self, z = None, derive_bias_from = None, bias = None, z_name = 'z', 
				bias_name = 'bias'):
		
		if z is None:
			raise Exception('Redshift cannot be None.')
		
		par_array = []
		opt_array = [z]
		par_names = []
		opt_names = [z_name]
		
		if derive_bias_from is None:
			if bias is None:
				raise Exception('Bias cannot be None if derive_bias_from is None.')
			par_array.append(bias)
			par_names.append(bias_name)
			self._derive_bias = False
		else:
			if bias is None:
				bias = 0.0
			_rm, self._bias_from_mdef, _, _ = mass_so.parseRadiusMassDefinition(derive_bias_from)
			self._bias_from_radius = (_rm == 'R')
			self._derive_bias = True
			self._rm_bias_name = derive_bias_from
			
		OuterTerm.__init__(self, par_array, opt_array, par_names, opt_names)
		
		return

	###############################################################################################

	def _getParameters(self):

		z = self.opt[self.term_opt_names[0]]
		
		if self._derive_bias:
			if self._rm_bias_name in self.par:
				rm_bias = self.par[self._rm_bias_name]
			elif self._rm_bias_name in self.opt:
				rm_bias = self.opt[self._rm_bias_name]
			else:
				msg = 'Could not find the parameter or option "%s".' % (self._rm_bias_name)
				raise Exception(msg)

			if self._bias_from_radius:
				rm_bias = mass_so.R_to_M(rm_bias, z, self._bias_from_mdef)
			
			bias = halo_bias.haloBias(rm_bias, z, self._bias_from_mdef)
			
		else:
			bias = self.par[self.term_par_names[0]]
		
		return z, bias

	###############################################################################################

	# We have to be a little careful when evaluating the matter-matter correlation function, since
	# it may not be defined at very small or large radii.

	def _xi_mm(self, r, z):

		cosmo = cosmology.getCurrent()
		r_com = r / 1000.0 * (1 + z)
		
		R_min = cosmo.R_xi[0] * 1.00001
		R_max = cosmo.R_xi[-1] * 0.99999
		
		mask_not_small = (r_com > R_min)
		mask_not_large = (r_com < R_max)
		mask = (mask_not_small & mask_not_large)
		
		xi_mm = np.zeros((len(r)), np.float)

		if np.count_nonzero(mask) > 0:
			xi_mm[mask] = cosmo.correlationFunction(r_com[mask], z)

		xi_mm[np.logical_not(mask_not_small)] = cosmo.correlationFunction(R_min, z)
		xi_mm[np.logical_not(mask_not_large)] = cosmo.correlationFunction(R_max, z)
		
		return xi_mm

	###############################################################################################

	def _density(self, r):
		
		z, bias = self._getParameters()
		rho = cosmology.getCurrent().rho_m(z) * bias * self._xi_mm(r, z)

		return rho

###################################################################################################
# OUTER TERM: POWER LAW
###################################################################################################

class OuterTermPowerLaw(OuterTerm):
	"""
	An outer term that describes density as a power-law in radius. 
	
	This class implements a power-law outer profile with a free normalization and slope, and a 
	fixed or variable pivot radius,
	
	.. math::
		\\rho(r) = \\frac{a \\times \\rho_m(z)}{\\frac{1}{m} + \\left(\\frac{r}{r_{\\rm pivot}}\\right)^{b}}
		
	where a is the normalization in units of the mean density of the universe, b the slope, and m 
	the maximum contribution to the density this term can make. Without such a limit, sufficiently 
	steep power-law profiles can lead to a spurious density contribution at the halo center. Note 
	that the slope is inverted, i.e. that a more positive slope means a steeper profile.
	
	The user can also set the internal parameter names of the input variables. If these names are 
	matched with an existing profile variable, that variable is used instead, meaning the outer
	term variable is not independent any more.

	Parameters
	-----------------------------------------------------------------------------------------------
	norm: float
		The density normalization of the term, in units of the mean matter density of the universe.
	slope: float
		The slope of the power-law profile.
	pivot: str
		Can either be ``'fixed'``, in which case ``pivot_factor`` determines the pivot radius in 
		physical units, or the name of one of the profile parameters or options.
	pivot_factor: float
		There are fundamentally two ways to set the pivot radius. If ``pivot=='fixed'``, 
		``pivot_factor`` gives the pivot radius in physical kpc/h. Otherwise, ``pivot`` must 
		indicate the name of a profile parameter or option. In this case, the pivot radius is set to 
		``pivot_factor`` times the parameter or option in question. For example, for profiles based 
		on a scale radius, a pivot radius of :math:`2 r_s` can be set by passing ``pivot = 'rs'`` 
		and ``pivot_factor = 2.0``. 
	z: float
		Redshift.
	max_rho: float
		The maximum density in units of the normalization times the mean density of the universe.
		This limit prevents spurious density contributions at the very center of halos. If you are
		unsure what this parameter should be set to, it can be useful to plot the density 
		contribution of the outer profile term. It should flatten to max_rho times norm times the 
		mean density at a radius where the inner profile strongly dominates the density, i.e. 
		where the contribution from the outer term does not matter.
	norm_name: str
		The internal name of the normalization parameter. If this name is set to an already existing
		profile parameter, the normalization is set to this other profile parameter, and thus not an
		independent parameter any more.
	slope_name: str
		The internal name of the slope parameter. See ``norm_name``.
	pivot_name: str
		The internal name of the pivot parameter. See ``norm_name``.
	pivot_factor_name: str
		The internal name of the pivot_factor parameter. See ``norm_name``.
	z_name: str
		The internal name of the redshift parameter. See ``norm_name``.
	max_rho_name: str
		The internal name of the maximum density parameter. See ``norm_name``.
	"""
	
	def __init__(self, norm = None, slope = None, pivot = None, pivot_factor = None, 
				z = None, max_rho = defaults.HALO_PROFILE_OUTER_PL_MAXRHO, 
				norm_name = 'pl_norm', slope_name = 'pl_slope', 
				pivot_name = 'pivot', pivot_factor_name = 'pivot_factor', z_name = 'z', 
				max_rho_name = 'pl_max_rho'):

		if norm is None:
			raise Exception('Normalization of power law cannot be None.')
		if slope is None:
			raise Exception('Slope of power law cannot be None.')
		if pivot is None:
			raise Exception('Pivot of power law cannot be None.')
		if pivot_factor is None:
			raise Exception('Pivot factor of power law cannot be None.')
		if z is None:
			raise Exception('Redshift of power law cannot be None.')
		if max_rho is None:
			raise Exception('Maximum of power law cannot be None.')
		
		OuterTerm.__init__(self, [norm, slope], [pivot, pivot_factor, z, max_rho],
						[norm_name, slope_name], [pivot_name, pivot_factor_name, z_name, max_rho_name])

		return

	###############################################################################################

	def _getParameters(self):

		r_pivot_id = self.opt[self.term_opt_names[0]]
		if r_pivot_id == 'fixed':
			r_pivot = 1.0
		elif r_pivot_id in self.par:
			r_pivot = self.par[r_pivot_id]
		elif r_pivot_id in self.opt:
			r_pivot = self.opt[r_pivot_id]
		else:
			msg = 'Could not find the parameter or option "%s".' % (r_pivot_id)
			raise Exception(msg)

		norm = self.par[self.term_par_names[0]]
		slope = self.par[self.term_par_names[1]]
		r_pivot *= self.opt[self.term_opt_names[1]]
		z = self.opt[self.term_opt_names[2]]
		max_rho = self.opt[self.term_opt_names[3]]
		rho_m = cosmology.getCurrent().rho_m(z)
		
		return norm, slope, r_pivot, max_rho, rho_m

	###############################################################################################

	def _density(self, r):
		
		norm, slope, r_pivot, max_rho, rho_m = self._getParameters()
		rho = rho_m * norm / (1.0 / max_rho + (r / r_pivot)**slope)

		return rho

	###############################################################################################

	def densityDerivativeLin(self, r):
		"""
		The linear derivative of the density due to the outer term, :math:`d \\rho / dr`. 

		Parameters
		-------------------------------------------------------------------------------------------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		derivative: array_like
			The linear derivative in physical :math:`M_{\odot} h / {\\rm kpc}^2`; has the same 
			dimensions as r.
		"""

		norm, slope, r_pivot, max_rho, rho_m = self._getParameters()
		t1 = (r / r_pivot)**slope
		drho_dr = -rho_m * norm * slope * t1 / r / (1.0 / max_rho + t1)**2

		return drho_dr

	###############################################################################################

	def _fitParamDeriv_rho(self, r, mask, N_par_fit):
		
		deriv = np.zeros((N_par_fit, len(r)), np.float)
		norm, slope, r_pivot, max_rho, rho_m = self._getParameters()
		
		rro = r / r_pivot
		t1 = 1.0 / max_rho + rro**slope
		rho = rho_m * norm / t1
		
		counter = 0
		# norm
		if mask[0]:
			deriv[counter] = rho / norm
			counter += 1
		# slope
		if mask[1]:
			deriv[counter] = -rho * np.log(rro) / t1 * rro**slope
		
		return deriv

	###############################################################################################
