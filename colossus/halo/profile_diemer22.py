###################################################################################################
#
# profile_diemer22.py       (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This module implements the Diemer 2022b form of the density profile. Please see 
:doc:`halo_profile` for a general introduction to the Colossus density profile module.

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

The `Diemer (2022b) <http://adsabs.harvard.edu/abs/2014ApJ...789....1D>`_ profile corresponds to an 
Einasto profile at small radii but steepens around the truncation radius:

	.. math::
		\\rho(r) = \\rho_{\\rm s} \\exp \\left\\{ -\\frac{2}{\\alpha} \\left[ \\left( \\frac{r}{r_{\\rm s}} \\right)^\\alpha - 1 \\right] -\\frac{1}{\\beta} \\left[ \\left( \\frac{r}{r_{\\rm t}} \\right)^\\beta - \\left( \\frac{r_{\\rm s}}{r_{\\rm t}} \\right)^\\beta \\right] \\right\\}

The meaning of this functional form is easiest to understand by considering its logarithmic slope:

	.. math::
		\\gamma(r) \\equiv \\frac{{\\rm d} \\ln \\rho}{{\\rm d} \\ln r} = -2 \\left( \\frac{r}{r_{\\rm s}} \\right)^\\alpha - \\left( \\frac{r}{r_{\\rm t}} \\right)^\\beta

The profile form was designed to fit the orbiting component of dark matter halos even at radii 
where the infalling component comes to dominate. The idea is thus to combine this profile form
with an infalling profile. The formula has 5 free parameters with well-defined physical 
interpretations:

.. table::
	:widths: auto
	
	======= ==================== ===================================================================================
	Param.  Symbol               Explanation	
	======= ==================== ===================================================================================
	rhos	:math:`\\rho_s`       Density at the scale radius, in physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`
	rs      :math:`r_{\\rm s}`     The scale radius in physical kpc/h
	alpha   :math:`\\alpha`       Determines how quickly the slope of the inner Einasto profile steepens
	rt      :math:`r_{\\rm t}`     The radius where the profile steepens beyond the Einasto profile, in physical kpc/h
	beta    :math:`\\beta`        Sharpness of the truncation
	======= ==================== ===================================================================================

There are two ways to initialize a D22 profile. First, the user can pass the fundamental
parameters of the profile listed above. Second, the user can pass a spherical overdensity mass 
and concentration, the conversion to the native parameters then relies on the calibrations 
in Diemer 22c. In this case, the user can give additional information about the profile that can be 
used in setting the fundamental parameters. 

In particular, the fitting function was calibrated for the median and mean profiles of two 
types of halo samples, namely samples selected by mass, and samples selected by both mass and 
mass accretion rate. The user can choose between those by setting ``selected_by = 'M'`` or 
``selected_by = 'Gamma'``. The latter option results in a more accurate representation
of the density profile, but the mass accretion rate must be known. 

See the :func:`~halo.profile_diemer22.D22Profile.deriveParameters` function for more details.

.. note::
	The D22 profile makes sense only if some description of the infalling profile is added. 
	
Adding outer terms is easy using the wrapper function :func:`getD22ProfileWithInfalling`::

	from colossus.cosmology import cosmology
	from colossus.halo import profile_diemer22

	cosmology.setCosmology('planck15')
	p = profile_diemer22.getD22ProfileWithInfalling(M = 1E12, c = 10.0, z = 0.0, mdef = 'vir')
	
This line will return a D22 profile object with a power-law outer profile and the mean density of
the universe added by default. Alternatively, the user can pass a list of OuterTerm objects 
(see documentation of the :class:`~halo.profile_base.HaloDensityProfile` parent class). The user
can pass additional parameters to the outer profile terms::

	p = profile_diemer22.getD22ProfileWithInfalling(M = 1E12, c = 10.0, z = 0.0, mdef = 'vir',
			power_law_slope = 1.2)

or change the outer terms altogether::

	p = profile_diemer22.getD22ProfileWithInfalling(M = 1E12, c = 10.0, z = 0.0, mdef = 'vir', 
			outer_term_names = ['mean', 'cf'], derive_bias_from = None, bias = 1.2)

Some of the outer term parameterizations (namely the 2-halo term) rely, in turn, on properties of 
the total profile such as the mass. In those cases, the constructor determines the mass iteratively,
taking the changing contribution of the outer term into account. This procedure can make the 
constructor slow. Thus, it is generally preferred to initialize the outer terms with fixed 
parameters (e.g., pivot radius or bias). Please see the :doc:`tutorials` for more code examples.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

import numpy as np
import scipy.optimize

from colossus import defaults
from colossus.utils import utilities
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.halo import mass_so
from colossus.halo import profile_base
from colossus.halo import mass_defs

###################################################################################################
# DIEMER & KRAVTSOV 2014 PROFILE
###################################################################################################

class D22Profile(profile_base.HaloDensityProfile):
	"""
	The Diemer 2022 density profile.
	
	The redshift must always be passed to this constructor, regardless of whether the 
	fundamental parameters or a mass and concentration are given.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	rhos: float
		The central scale density, in physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`.
	rs: float
		The scale radius in physical kpc/h.
	rt: float
		The radius where the profile steepens, in physical kpc/h.
	alpha: float
		Determines how quickly the slope of the inner Einasto profile steepens.
	beta: float
		Sharpness of the steepening.
	M: float
		Halo mass in :math:`M_{\odot}/h`.
	c: float
		Concentration in the same mass definition as ``M``.
	mdef: str
		The mass definition to which ``M`` corresponds. See :doc:`halo_mass` for details.
	z: float
		Redshift
	selected_by: str
		The halo sample to which this profile refers can be selected mass ``M`` or by accretion
		rate ``Gamma``. This parameter influences how some of the fixed parameters in the 
		profile are set, in particular those that describe the steepening term.
	Gamma: float
		The mass accretion rate over the past dynamical time, which is defined as the crossing 
		time (see func:`~halo.mass_so.dynamicalTime` or Diemer 2017 for details). The definition 
		in the DK14 profile is slightly different, but the definitions are close enough that they
		can be used interchangeably without great loss of accuracy. The Gamma parameter only needs 
		to be passed if ``selected_by == 'Gamma'``.
	acc_warn: float
		If the function achieves a relative accuracy in matching ``M`` less than this value, a 
		warning is printed.
	acc_err: float
		If the function achieves a relative accuracy in matching ``M`` less than this value, an 
		exception is raised.
	"""
	
	###############################################################################################
	# CONSTRUCTOR
	###############################################################################################
	
	def __init__(self, z = None, selected_by = defaults.HALO_PROFILE_SELECTED_BY, Gamma = None, 
				**kwargs):

		# Set the fundamental variables par_names and opt_names
		self.par_names = ['rhos', 'rs', 'rt', 'alpha', 'beta']
		self.opt_names = ['selected_by', 'Gamma', 'R200m', 'z']

		if z is None:
			raise Exception('Need the redshift z to construct a Diemer22 profile.')

		self.opt['selected_by'] = selected_by
		self.opt['Gamma'] = Gamma
		self.opt['z'] = z
		self.opt['R200m'] = None
		
		# Run the constructor
		profile_base.HaloDensityProfile.__init__(self, **kwargs)
	
		# Sanity checks
		if self.par['rhos'] < 0.0 or self.par['rs'] < 0.0 or self.par['rt'] < 0.0:
			raise Exception('The radius parameters cannot be negative, something went wrong (%s).' % (str(self.par)))

		# We need to guess a radius when computing vmax
		self.r_guess = self.par['rs']

		return

	###############################################################################################
	# STATIC METHODS
	###############################################################################################

	@staticmethod
	def deriveParameters(selected_by, nu200m = None, z = None, Gamma = None):
		"""
		Calibration of the parameters :math:`\\alpha`, :math:`\\beta`, and :math:`r_{\\rm t}`.

		This function determines the values of those parameters in the Diemer22 profile that can be 
		calibrated based on mass, and potentially mass accretion rate. The latter is the stronger
		determinant of the profile shape, but may not always be available (e.g., for mass-selected
		samples).
		
		We set :math:`\\alpha = 0.18` and :math:`\\beta = 3`, which are the default parameters for 
		individual halo profiles. However, they are not necessarily optimal for any type of 
		averaged sample, where the optimal values vary. We do not calibrate :math:`\\alpha` with 
		mass as suggested by
		`Gao et al. 2008 <http://adsabs.harvard.edu/abs/2008MNRAS.387..536G>`_ because we do 
		not reproduce this relation in our data in Diemer 2022c.
		
		The truncation ratius :math:`r_{\\rm t}` is calibrated as suggested by DK14.
		If ``selected_by = 'M'``, we use Equation 6 in DK14. If ``selected_by = 'Gamma'``, 
		:math:`r_{\\rm t}` is calibrated from ``Gamma`` and ``z``. The DK14 calibrations are based
		on slightly different definitions of peak height (:math:`\\nu = \\nu_{\\rm vir}`), 
		accretion rate, and for a different fitting function. However, the resulting :math:`r_{\\rm t}`
		values are very similar to the forthcoming analysis in Diemer 2022c. 

		Parameters
		-------------------------------------------------------------------------------------------
		selected_by: str
			The halo sample to which this profile refers can be selected mass ``M`` or by accretion
			rate ``Gamma``.
		nu200m: float
			The peak height of the halo for which the parameters are to be calibrated. This 
			parameter only needs to be passed if ``selected_by == 'M'``.
		z: float
			Redshift
		Gamma: float
			The mass accretion rate over the past dynamical time, which is defined as the crossing 
			time (see func:`~halo.mass_so.dynamicalTime` or Diemer 2017 for details). The definition 
			in the DK14 profile is slightly different, but the definitions are close enough that they
			can be used interchangeably without great loss of accuracy. The Gamma parameter only needs 
			to be passed if ``selected_by == 'Gamma'``.

		Returns
		-------------------------------------------------------------------------------------------
		alpha: float
			The Einasto steepening parameter.
		beta: float
			The steepening of the truncation term.
		rt_R200m: float
			The truncation radius in units of R200m.
		"""

		alpha = 0.18
		beta = 3.0

		if selected_by == 'M':
			if (nu200m is not None):
				rt_R200m = 1.9 - 0.18 * nu200m
			else:
				raise Exception('Need nu200m to compute rt.')				
			
		elif selected_by == 'Gamma':
			if (Gamma is not None) and (z is not None):
				cosmo = cosmology.getCurrent()
				rt_R200m =  0.43 * (1.0 + 0.92 * cosmo.Om(z)) * (1.0 + 2.18 * np.exp(-Gamma / 1.91))
			else:
				raise Exception('Need Gamma and z to compute rt.')

		else:
			msg = "Unknown sample selection, %s." % (selected_by)
			raise Exception(msg)

		return alpha, beta, rt_R200m

	###############################################################################################
	# METHODS BOUND TO THE CLASS
	###############################################################################################

	def setNativeParameters(self, M, c, z, mdef, selected_by, Gamma = None, 
							acc_warn = 0.01, acc_err = 0.05):

		# Declare shared variables; these parameters are advanced during the iterations
		par2 = {}
		par2['RDelta'] = 0.0
		
		RTOL = 0.01
		MTOL = 0.01
		GUESS_TOL = 2.5

		# -----------------------------------------------------------------------------------------

		# Try a radius R200m, compute the resulting RDelta using the old RDelta as a starting guess
		
		def radius_diff(R200m, par2, Gamma, rho_target, R_target):
			
			self.opt['R200m'] = R200m
			M200m = mass_so.R_to_M(R200m, z, '200m')
			nu200m = peaks.peakHeight(M200m, z)

			self.par['alpha'], self.par['beta'], rt_R200m = \
				self.deriveParameters(selected_by, nu200m = nu200m, z = z, Gamma = Gamma)
			self.par['rt'] = rt_R200m * R200m
			self.par['rhos'] *= self._normalizeInner(R200m, M200m)

			par2['RDelta'] = self._RDeltaLowlevel(par2['RDelta'], rho_target, 
												guess_tolerance = GUESS_TOL)
			
			return par2['RDelta'] - R_target
		
		# -----------------------------------------------------------------------------------------
		
		# The user needs to set a cosmology before this function can be called
		R_target = mass_so.M_to_R(M, z, mdef)
		self.par['rs'] = R_target / c
		
		if mdef == '200m':
			
			# The user has supplied M200m, the parameters follow directly from the input
			M200m = M
			self.opt['R200m'] = mass_so.M_to_R(M200m, z, '200m')
			nu200m = peaks.peakHeight(M200m, z)
			self.par['alpha'], self.par['beta'], rt_R200m = \
				self.deriveParameters(selected_by, nu200m = nu200m, z = z, Gamma = Gamma)
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
				msg = 'WARNING: D22 profile parameters converged to an accuracy of %.1f percent.' % (abs(err) * 100.0)
				print(msg)
			
			if abs(err) > acc_err:
				msg = 'D22 profile parameters not converged (%.1f percent error).' % (abs(err) * 100.0)
				raise Exception(msg)
		
		return

	###############################################################################################
	
	def densityInner(self, r):
		"""
		Density of the inner profile as a function of radius.
		
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
		
		rs = self.par['rs']
		rt = self.par['rt']
		alpha = self.par['alpha']
		beta = self.par['beta']
		
		S = -2.0 / alpha * ((r / rs)**alpha - 1.0) - 1.0 / beta * ((r / rt)**beta - (rs / rt)**beta)
		rho = self.par['rhos'] * utilities.safeExp(S)

		return rho

	###############################################################################################
	
	def densityDerivativeLinInner(self, r):
		"""
		The linear derivative of the inner density, :math:`d \\rho_{\\rm inner} / dr`. 
		
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
		
		d_lnrho_d_lnr = self.densityDerivativeLogInner(r)
		rho = self.density(r)
		der = d_lnrho_d_lnr * rho / r
		
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
		
		der = -2.0 * (r / self.par['rs'])**self.par['alpha'] - (r / self.par['rt'])**self.par['beta']

		return der
		
	###############################################################################################

	# This function returns the spherical overdensity radius (in kpc / h) given a mass definition
	# and redshift. We know R200m and thus M200m for a DK14 profile, and use those parameters to
	# compute what R would be for an NFW profile and use this radius as an initial guess.
	
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
	
		M200m = mass_so.R_to_M(self.opt['R200m'], z, mdef)
		_, R_guess, _ = mass_defs.changeMassDefinition(M200m, self.opt['R200m'] / self.par['rs'], z, '200m', mdef)
		density_threshold = mass_so.densityThreshold(z, mdef)
		R = self._RDeltaLowlevel(R_guess, density_threshold)
	
		return R

	###############################################################################################

	def Rsp(self, search_range = 5.0):
		"""
		The splashback radius, :math:`R_{\\rm sp}`.
		
		See the :doc:`halo_splashback` section for a detailed description of the splashback radius.
		Here, we define :math:`R_{\\rm sp}` as the radius where the profile reaches its steepest 
		logarithmic slope.
		
		Parameters
		-------------------------------------------------------------------------------------------
		search_range: float
			When searching for the radius of steepest slope, search within this factor of 
			:math:`R_{\\rm 200m}` (optional).
			
		Returns
		-------------------------------------------------------------------------------------------
		Rsp: float
			The splashback radius, :math:`R_{\\rm sp}`, in physical kpc/h.
			
		See also
		-------------------------------------------------------------------------------------------
		RMsp: The splashback radius and mass within, :math:`R_{\\rm sp}` and :math:`M_{\\rm sp}`.
		Msp: The mass enclosed within :math:`R_{\\rm sp}`, :math:`M_{\\rm sp}`.
		"""
		
		R200m = self.opt['R200m']
		rc = scipy.optimize.fminbound(self.densityDerivativeLog, R200m / search_range, R200m * search_range)

		return rc
	
	###############################################################################################

	def RMsp(self, search_range = 5.0):
		"""
		The splashback radius and mass within, :math:`R_{\\rm sp}` and :math:`M_{\\rm sp}`.
		
		See the :doc:`halo_splashback` section for a detailed description of the splashback radius.
		Here, we define :math:`R_{\\rm sp}` as the radius where the profile reaches its steepest 
		logarithmic slope.
		
		Parameters
		-------------------------------------------------------------------------------------------
		search_range: float
			When searching for the radius of steepest slope, search within this factor of 
			:math:`R_{\\rm 200m}` (optional).
			
		Returns
		-------------------------------------------------------------------------------------------
		Rsp: float
			The splashback radius, :math:`R_{\\rm sp}`, in physical kpc/h.
		Msp: float
			The mass enclosed within the splashback radius, :math:`M_{\\rm sp}`, in :math:`M_{\odot} / h`.
			
		See also
		-------------------------------------------------------------------------------------------
		Rsp: The splashback radius, :math:`R_{\\rm sp}`.
		Msp: The mass enclosed within :math:`R_{\\rm sp}`, :math:`M_{\\rm sp}`.
		"""
		
		Rsp = self.Rsp(search_range = search_range)
		Msp = self.enclosedMass(Rsp)

		return Rsp, Msp
	
	###############################################################################################

	def Msp(self, search_range = 5.0):
		"""
		The mass enclosed within :math:`R_{\\rm sp}`, :math:`M_{\\rm sp}`.
		
		See the :doc:`halo_splashback` section for a detailed description of the splashback radius.
		Here, we define :math:`R_{\\rm sp}` as the radius where the profile reaches its steepest 
		logarithmic slope.
		
		Parameters
		-------------------------------------------------------------------------------------------
		search_range: float
			When searching for the radius of steepest slope, search within this factor of 
			:math:`R_{\\rm 200m}` (optional).
			
		Returns
		-------------------------------------------------------------------------------------------
		Msp: float
			The mass enclosed within the splashback radius, :math:`M_{\\rm sp}`, in :math:`M_{\odot} / h`.
			
		See also
		-------------------------------------------------------------------------------------------
		Rsp: The splashback radius, :math:`R_{\\rm sp}`.
		RMsp: The splashback radius and mass within, :math:`R_{\\rm sp}` and :math:`M_{\\rm sp}`.
		"""
		
		_, Msp = self.RMsp(search_range = search_range)

		return Msp
	
	###############################################################################################

	# We fit all parameters in log space

	def _fitConvertParams(self, p, mask):

		return np.log(p)

	###############################################################################################
	
	def _fitConvertParamsBack(self, p, mask):
	
		return np.exp(p)

	###############################################################################################
	
	def _fitParamDeriv_rho(self, r, mask, N_par_fit):

		x = self.getParameterArray()
		deriv = np.zeros((N_par_fit, len(r)), float)

		rhos = x[0]
		rs = x[1]
		rt = x[2]
		alpha = x[3]
		beta = x[4]
		
		rrs = r / rs
		rrt = r / rt
		rrsa = rrs**alpha
		rrtb = rrt**beta
		rsrt = rs / rt
		rsrtb = rsrt**beta

		s = -2.0 / alpha * (rrsa - 1.0) - 1.0 / beta * (rrtb - rsrtb)
		rho = rhos * utilities.safeExp(s)
		
		counter = 0
		# rhos
		if mask[0]:
			deriv[counter][:] = 1.0
			counter += 1
		# rs
		if mask[1]:
			deriv[counter] = 2.0 * rrsa + rsrtb
			counter += 1
		# rt
		if mask[2]:
			deriv[counter] = rrtb - rsrtb
			counter += 1
		# alpha
		if mask[3]:
			deriv[counter] = 2.0 / alpha * (rrsa * (1.0 - alpha * np.log(rrs)) - 1.0)
			counter += 1
		# beta
		if mask[4]:
			deriv[counter] = rrtb * (1.0 / beta - np.log(rrt)) - rsrtb * (1.0 / beta - np.log(rsrt))

		deriv[:, :] *= rho[None, :]

		return deriv
