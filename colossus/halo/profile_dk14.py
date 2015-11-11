###################################################################################################
#
# profile_dk14.py           (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import numpy as np
import scipy.optimize

from colossus.utils import defaults
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.halo import profile_base
from colossus.halo import profile_outer
from colossus.halo import mass_defs

###################################################################################################
# DIEMER & KRAVTSOV 2014 PROFILE
###################################################################################################

class DK14Profile(profile_base.HaloDensityProfile):
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
	# CONSTRUCTOR
	###############################################################################################
	
	def __init__(self, rhos = None, rs = None, rt = None, alpha = None, beta = None, gamma = None, 
				R200m = None,
				M = None, c = None, z = None, mdef = None,
				selected_by = 'M', Gamma = None, 
				outer_term_names = ['mean', 'pl'], 
				be = defaults.HALO_PROFILE_DK14_BE, se = defaults.HALO_PROFILE_DK14_SE,
				power_law_max = 1000.0, acc_warn = 0.01, acc_err = 0.05):
	
		# Set the fundamental variables par_names and opt_names
		self.par_names = ['rhos', 'rs', 'rt', 'alpha', 'beta', 'gamma']
		self.opt_names = ['selected_by', 'Gamma', 'R200m']
		self.fit_log_mask = np.array([False, False, False, False, False, False])

		# Set outer terms
		outer_terms = []
		for i in range(len(outer_term_names)):
			
			if outer_term_names[i] == 'mean':
				if z is None:
					raise Exception('Redshift z must be set if a mean density outer term is chosen.')
				t = profile_outer.OuterTermRhoMean(z)
			
			elif outer_term_names[i] == 'pl':
				t = profile_outer.OuterTermPowerLaw(be, se, 'R200m', 5.0, power_law_max, z, norm_name = 'be', slope_name = 'se')
			
			elif outer_term_names[i] == 'ximm':
				t = profile_outer.OuterTermXiMatterPowerLaw(be, se, 'R200m', 5.0, power_law_max, z, norm_name = 'be', slope_name = 'se')
		
			else:
				msg = 'Unknown outer term, %s.' % (outer_terms[i])
				raise Exception(msg)
		
			outer_terms.append(t)
		
		# Run the constructor
		profile_base.HaloDensityProfile.__init__(self, outer_terms = outer_terms)
		
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
			M200m = mass_so.R_to_M(R200m, z, '200m')
			nu200m = cosmo.peakHeight(M200m, z)

			self.par['alpha'], self.par['beta'], self.par['gamma'], rt_R200m = \
				self.getFixedParameters(selected_by, nu200m = nu200m, z = z, Gamma = Gamma)
			self.par['rt'] = rt_R200m * R200m
			self.par['rhos'] *= self._normalizeInner(R200m, M200m)

			par2['RDelta'] = self._RDeltaLowlevel(par2['RDelta'], rho_target, guess_tolerance = GUESS_TOL)
			
			return par2['RDelta'] - R_target
		
		# -----------------------------------------------------------------------------------------
		
		# The user needs to set a cosmology before this function can be called
		cosmo = cosmology.getCurrent()
		R_target = mass_so.M_to_R(M, z, mdef)
		self.par['rs'] = R_target / c
		
		if mdef == '200m':
			
			# The user has supplied M200m, the parameters follow directly from the input
			M200m = M
			self.opt['R200m'] = mass_so.M_to_R(M200m, z, '200m')
			nu200m = cosmo.peakHeight(M200m, z)
			self.par['alpha'], self.par['beta'], self.par['gamma'], rt_R200m = \
				self.getFixedParameters(selected_by, nu200m = nu200m, z = z, Gamma = Gamma)
			self.par['rt'] = rt_R200m * self.opt['R200m']

			# Guess rhos = 1.0, then re-normalize			
			self.par['rhos'] = 1.0
			self.par['rhos'] *= self._normalizeInner(self.opt['R200m'], M200m)
			
		else:
			
			# The user has supplied some other mass definition, we need to iterate.
			_, R200m_guess, _ = mass_defs.changeMassDefinition(M, c, z, mdef, '200m')
			par2['RDelta'] = R_target
			self.par['rhos'] = 1.0

			# Iterate to find an M200m for which the desired mass is correct
			rho_target = mass_so.densityThreshold(z, mdef)
			args = par2, Gamma, rho_target, R_target
			self.opt['R200m'] = scipy.optimize.brentq(radius_diff, R200m_guess / 1.3, R200m_guess * 1.3,
								args = args, xtol = RTOL)

			# Check the accuracy of the result; M should be very close to MDelta now
			M_result = mass_so.R_to_M(par2['RDelta'], z, mdef)
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

	# The opt['R200m'] parameter is not guaranteed to stay in sync with the other parameters, for
	# example after a fit. This function remedies that, and also changes other parameters that 
	# depend on R200m, for example be and se in the case of an outer power law term.

	def update(self):
		
		profile_base.HaloDensityProfile.update(self)
		
		R200m_new = self.RDelta(self.opt['z'], '200m')
		
		# If the power law outer term relies on 
		for i in range(len(self._outer_terms)):
			if isinstance(self._outer_terms[i], profile_outer.OuterTermPowerLaw):
				if self._outer_terms[i].term_opt['r_pivot'] == 'R200m':
					self._outer_terms[i].changePivot(R200m_new)
				
		self.opt['R200m'] = R200m_new
		
		return

	###############################################################################################
	
	def densityInner(self, r):
		
		inner = self.par['rhos'] * np.exp(-2.0 / self.par['alpha'] * ((r / self.par['rs'])**self.par['alpha'] - 1.0))
		fT = (1.0 + (r / self.par['rt'])**self.par['beta'])**(-self.par['gamma'] / self.par['beta'])
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
	
		M200m = mass_so.R_to_M(self.opt['R200m'], z, mdef)
		_, R_guess, _ = mass_defs.changeMassDefinition(M200m, self.opt['R200m'] / self.par['rs'], z, '200m', mdef)
		density_threshold = mass_so.densityThreshold(z, mdef)
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
		
		R200m = self.opt['R200m']
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

	# When fitting the DK14 profile, use a mixture of linear and logarithmic parameters. Only 
	# conver the parameters for the inner profile though.

	def _fitConvertParams(self, p, mask):
		
		p_fit = p.copy()
		log_mask = self.fit_log_mask[mask[:6]]
		p_fit[log_mask] = np.log(p_fit[log_mask])

		return p_fit

	###############################################################################################
	
	def _fitConvertParamsBack(self, p, mask):
		
		p_def = p.copy()
		log_mask = self.fit_log_mask[mask[:6]]
		p_def[log_mask] = np.exp(p_def[log_mask])

		return p_def

	###############################################################################################
	
	def _fitParamDeriv_rho(self, r, mask, N_par_fit):

		x = self.getParameterArray()
		deriv = np.zeros((N_par_fit, len(r)), np.float)
		rho_inner = self.densityInner(r)

		rhos = x[0]
		rs = x[1]
		rt = x[2]
		alpha = x[3]
		beta = x[4]
		gamma = x[5]

		rrs = r / rs
		rrt = r / rt
		term1 = 1.0 + rrt**beta
		
		counter = 0
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

		# Correct for log parameters
		counter = 0
		for i in range(6):
			if self.fit_log_mask[i] and mask[i]:
				deriv[counter] *= x[i]
			if mask[i]:
				counter += 1
		
		return deriv
	