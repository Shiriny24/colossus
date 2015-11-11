###################################################################################################
#
# profile_outer.py           (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module implements terms that describe the outer halo density profile. Specific terms are 
derived from the :class:`OuterTerm` base class.
"""

###################################################################################################

import numpy as np
import scipy.misc
import abc
import collections
import six

from colossus.utils import utilities
from colossus.cosmology import cosmology
from colossus.halo import basics
from colossus.halo import bias

###################################################################################################
# ABSTRACT BASE CLASS FOR OUTER PROFILE TERMS
###################################################################################################

# The parameter and option names in this class are up to the user. This enables two kinds of 
# behavior: first, potentially conflicting standard names can be changed, and second, the user can
# "point" to an already existing parameter.

@six.add_metaclass(abc.ABCMeta)
class OuterTerm():
	"""
	Base class for outer profile terms.
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
		
		# Some other settings
		self.include_in_surface_density = True
		
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
# OUTER TERM: MEAN DENSITY
###################################################################################################

class OuterTermRhoMean(OuterTerm):
	
	def __init__(self, z):
		
		if z is None:
			raise Exception('Redshift cannot be None.')
		
		OuterTerm.__init__(self, [], [z], [], ['z'])
		
		self.z = z
		cosmo = cosmology.getCurrent()
		self.rho_m = cosmo.rho_m(z)
		
		# This term should not be included when computing the surface density, as it leads to 
		# a diverging integral.
		self.include_in_surface_density = False
		
		return

	###############################################################################################

	def _density(self, r):
		
		return np.ones((len(r)), np.float) * self.rho_m

###################################################################################################
# OUTER TERM: POWER LAW
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
		max_rho = self.opt[self.term_opt_names[2]]
		z = self.opt[self.term_opt_names[3]]
		rho_m = cosmology.getCurrent().rho_m(z)
		
		return norm, slope, r_pivot, max_rho, rho_m

	###############################################################################################

	def _density(self, r):
		
		norm, slope, r_pivot, max_rho, rho_m = self._getParameters()
		rho = rho_m * norm / (1.0 / max_rho + (r / r_pivot)**slope)
		#rho = rho_m * norm / ((r / r_pivot)**slope)

		return rho

	###############################################################################################

	def densityDerivativeLin(self, r):

		norm, slope, r_pivot, max_rho, rho_m = self._getParameters()
		t1 = 1.0 / r_pivot
		t2 = r * t1
		drho_dr = -rho_m * norm * slope * t1 * (1.0 / max_rho + t2**slope)**-2 * t2**(slope - 1.0)

		return drho_dr

	###############################################################################################

	def _fitParamDeriv_rho(self, r, mask, N_par_fit):
		
		deriv = np.zeros((N_par_fit, len(r)), np.float)
		norm, slope, r_pivot, max_rho, rho_m = self._getParameters()
		
		#print(norm, slope, r_pivot, max_rho, rho_m)
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
		
		#print(deriv)
		return deriv

	###############################################################################################
	
	# TODO correct this function; must include max_rho
	def changePivot(self, new_pivot):
		
		norm, slope, r_pivot, _, _ = self._getParameters()
		
		self.par[self.term_par_names[0]] = norm * (r_pivot / new_pivot)**slope

		return

###################################################################################################
# OUTER TERM: 2-HALO TERM BASED ON THE MATTER-MATTER CORRELATION FUNCTION, TIMES POWER LAW
###################################################################################################

class OuterTermXiMatterPowerLaw(OuterTerm):
	
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

		r_pivot_id = self.opt[self.term_opt_names[0]]
		if r_pivot_id in self.par:
			r_pivot = self.par[r_pivot_id]
		elif r_pivot_id in self.opt:
			r_pivot = self.opt[r_pivot_id]
		else:
			msg = 'Could not find the parameter or option %s.' % (r_pivot_id)
			raise Exception(msg)

		norm = self.par[self.term_par_names[0]]
		slope = self.par[self.term_par_names[1]]
		r_pivot *= self.opt[self.term_opt_names[1]]
		max_rho = self.opt[self.term_opt_names[2]]
		z = self.opt[self.term_opt_names[3]]
		rho_m = cosmology.getCurrent().rho_m(z)
		
		return norm, slope, r_pivot, max_rho, z, rho_m

	###############################################################################################

	# TODO don't use special knowledge about DK14 profile
	def _density(self, r):
		
		r_array, is_array = utilities.getArray(r)

		norm, slope, r_pivot, max_rho, z, rho_m = self._getParameters()
		
		cosmo = cosmology.getCurrent()
		r_Mpc = r_array / 1000.0
		mask = (r_Mpc > cosmo.R_xi[0]) & (r_Mpc < cosmo.R_xi[-1])

		rho = np.ones((len(r)), np.float) * norm * rho_m / max_rho
		
		#print(mask)
		if np.count_nonzero(mask) > 0:
			xi_mm = cosmo.correlationFunction(r_Mpc[mask], z)
	
			M200m = basics.R_to_M(self.owner.opt['R200m'], z, '200m')
			#M200m = self.owner.MDelta(z, '200m')
			b = bias.haloBias(M200m, z, '200m')
			
			rho[mask] = norm * rho_m * (1.0 / max_rho + xi_mm * b * (r_array[mask] / r_pivot)**slope)

		if not is_array:
			rho = rho[0]

		return rho

	###############################################################################################
	
	# TODO correct this function; must include max_rho
	def changePivot(self, new_pivot):
		
		#norm, slope, r_pivot, _, _ = self._getParameters()
		
		#self.par[self.term_par_names[0]] = norm * (r_pivot / new_pivot)**slope

		return
	