###################################################################################################
#
# profile_spline.py         (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import numpy as np
import scipy.integrate
import scipy.interpolate

from colossus.halo import profile_base

###################################################################################################
# SPLINE DEFINED PROFILE
###################################################################################################

class SplineProfile(profile_base.HaloDensityProfile):
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
		profile_base.HaloDensityProfile.__init__(self)
		
		self.rmin = np.min(r)
		self.rmax = np.max(r)
		self.r_guess = np.sqrt(self.rmin * self.rmax)
		self.min_RDelta = self.rmin
		self.max_RDelta = self.rmax

		if rho is None and M is None:
			msg = 'Either mass or density must be specified.'
			raise Exception(msg)
		
		self.rho_spline = None
		self.M_spline = None
		logr = np.log(r)
		
		if M is not None:
			logM = np.log(M)
			self.M_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logM)

		if rho is not None:
			logrho = np.log(rho)
			self.rho_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logrho)

		# Construct M(r) from density. For some reason, the spline integrator fails on the 
		# innermost bin, and the quad integrator fails on the outermost bin. 
		if self.M_spline is None:
			integrand = 4.0 * np.pi * r**2 * rho
			integrand_spline = scipy.interpolate.InterpolatedUnivariateSpline(r, integrand)
			logM = 0.0 * r
			for i in range(len(logM) - 1):
				logM[i], _ = scipy.integrate.quad(integrand_spline, 0.0, r[i])
			logM[-1] = integrand_spline.integral(0.0, r[-1])
			logM = np.log(logM)
			self.M_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logM)

		if self.rho_spline is None:
			deriv = self.M_spline(np.log(r), nu = 1) * M / r
			logrho = np.log(deriv / 4.0 / np.pi / r**2)
			self.rho_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logrho)

		return

	###############################################################################################
	# METHODS BOUND TO THE CLASS
	###############################################################################################

	def densityInner(self, r):
		
		return np.exp(self.rho_spline(np.log(r)))

	###############################################################################################
	
	def densityDerivativeLinInner(self, r):

		log_deriv = self.rho_spline(np.log(r), nu = 1)
		deriv = log_deriv * self.density(r) / r
		
		return deriv

	###############################################################################################

	def densityDerivativeLogInner(self, r):
	
		return self.rho_spline(np.log(r), nu = 1)
	
	###############################################################################################

	def enclosedMass(self, r, accuracy = 1E-6):

		return self.enclosedMassInner(r) + self.enclosedMassOuter(r, accuracy)

	###############################################################################################

	def enclosedMassInner(self, r):

		return np.exp(self.M_spline(np.log(r)))
