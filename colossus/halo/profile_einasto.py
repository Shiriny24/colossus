###################################################################################################
#
# profile_einasto.py        (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import numpy as np
import scipy.special

from colossus.cosmology import cosmology
from colossus.halo import basics
from colossus.halo import profile_base
from colossus.halo import profile_utils

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
				M = None, c = None, z = None, mdef = None, **kwargs):
	
		self.par_names = ['rhos', 'rs', 'alpha']
		self.opt_names = []
		profile_base.HaloDensityProfile.__init__(self, **kwargs)

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
		self._computeMassTerms()

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
				Mvir, _, _ = profile_utils.changeMassDefinition(M, c, z, mdef, 'vir')
			cosmo = cosmology.getCurrent()
			nu_vir = cosmo.peakHeight(Mvir, z)
			alpha = 0.155 + 0.0095 * nu_vir**2
		
		self.par['alpha'] = alpha
		self.par['rhos'] = 1.0
		self._computeMassTerms()
		M_unnorm = self.enclosedMass(R)
		self.par['rhos'] = M / M_unnorm
		
		return
	
	###############################################################################################

	def _computeMassTerms(self):
		
		self.mass_norm = np.pi * self.par['rhos'] * self.par['rs']**3 * 2.0**(2.0 - 3.0 / self.par['alpha']) \
			* self.par['alpha']**(-1.0 + 3.0 / self.par['alpha']) * np.exp(2.0 / self.par['alpha']) 
		self.gamma_3alpha = scipy.special.gamma(3.0 / self.par['alpha'])
		
		return
	
	###############################################################################################

	# The enclosed mass for the Einasto profile is semi-analytical, in that it can be expressed
	# in terms of Gamma functions. We pre-compute some factors to speed up the computation 
	# later.
	
	def update(self):
		
		profile_base.HaloDensityProfile.update(self)
		self._computeMassTerms()
		
		return
	
	###############################################################################################

	# We need to overwrite the setParameterArray function because the mass terms need to be 
	# updated when the user changes the parameters. Note that we do not call the update function
	# which triggers an update of the parent class.
	
	def setParameterArray(self, pars, mask = None):
		
		profile_base.HaloDensityProfile.setParameterArray(self, pars, mask = mask)
		self._computeMassTerms()
		
		return

	###############################################################################################

	def densityInner(self, r):
		
		rho = self.par['rhos'] * np.exp(-2.0 / self.par['alpha'] * \
										((r / self.par['rs'])**self.par['alpha'] - 1.0))
		
		return rho

	###############################################################################################
	
	def densityDerivativeLinInner(self, r):

		rho = self.density(r)
		drho_dr = rho * (-2.0 / self.par['rs']) * (r / self.par['rs'])**(self.par['alpha'] - 1.0)	
		
		return drho_dr

	###############################################################################################
	
	def densityDerivativeLogInner(self, r):

		der = -2.0 * (r / self.par['rs'])**self.par['alpha']
		
		return der

	###############################################################################################

	def enclosedMassInner(self, r):
		
		mass = self.mass_norm * self.gamma_3alpha * scipy.special.gammainc(3.0 / self.par['alpha'],
								2.0 / self.par['alpha'] * (r / self.par['rs'])**self.par['alpha'])
		
		return mass
	
	###############################################################################################

	# When fitting the Einasto profile, use log(rhos), log(rs) and log(alpha)

	def _fitConvertParams(self, p, mask):

		N_convert = np.count_nonzero(mask[:3])
		pp = p.copy()
		pp[:N_convert] = np.log(pp[:N_convert])
				
		return pp

	###############################################################################################
	
	def _fitConvertParamsBack(self, p, mask):
		
		N_convert = np.count_nonzero(mask[:3])
		pp = p.copy()
		pp[:N_convert] = np.exp(pp[:N_convert])
		
		return pp
	
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

		return deriv
