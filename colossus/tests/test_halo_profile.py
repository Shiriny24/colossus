###################################################################################################
#
# test_halo_profile.py  (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import numpy as np
import unittest

from colossus.tests import test_colossus
from colossus.utils import utilities
from colossus.cosmology import cosmology
from colossus.halo import profile_outer
from colossus.halo import profile_nfw
from colossus.halo import profile_einasto
from colossus.halo import profile_dk14
from colossus.halo import profile_base
from colossus.halo import profile_spline

###################################################################################################

TEST_N_DIGITS = test_colossus.TEST_N_DIGITS

###################################################################################################
# TEST CASE: SPHERICAL OVERDENSITY
###################################################################################################

# This test case compares three different implementations of the NFW density profile: 
# - the exact, analytic form
# - the generic implementation of the HaloDensityProfile base class, where only the density is 
#   computed analytically, but all other functions numerically ('Numerical')
# - a discrete profile where density and/or mass are given as arrays. Three cases are tested, with
#   only rho, only M, and both ('ArrayRho', 'ArrayM', and 'ArrayRhoM').

class TCBase(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('WMAP9')
		self.MAX_DIFF_RHO = 1E-8
		self.MAX_DIFF_M = 1E-8
		self.MAX_DIFF_DER = 1E-2
		self.MAX_DIFF_SIGMA = 1E-5
		self.MAX_DIFF_VCIRC = 1E-8
		self.MAX_DIFF_RMAX = 1E-3
		self.MAX_DIFF_VMAX = 1E-8
		self.MAX_DIFF_SO_R = 1E-8
		self.MAX_DIFF_SO_M = 1E-8
	
	def test_profileAccuracy(self, verbose = False):

		class TestProfile(profile_base.HaloDensityProfile):
			
			def __init__(self, rhos, rs):
				
				self.par_names = ['rhos', 'rs']
				self.opt_names = []
				profile_base.HaloDensityProfile.__init__(self)
				
				self.par['rhos'] = rhos
				self.par['rs'] = rs
				
				return
			
			def densityInner(self, r):
			
				x = r / self.par['rs']
				density = self.par['rhos'] / x / (1.0 + x)**2
				
				return density
		
		# Properties of the test halo
		M = 1E12
		c = 10.0
		mdef = 'vir'
		z = 0.0
		
		# Radii and reshifts where to test
		r_test = np.array([0.011, 1.13, 10.12, 102.3, 505.0])
		z_test = 1.0
		mdef_test = '200c'
	
		# Parameters for the finite-resolution NFW profile; here we want to test whether this method
		# converges to the correct solution, so the resolution is high.
		r_min = 1E-2
		r_max = 1E4
		N = 1000
	
		# PROFILE 1: Analytical NFW profile
		prof1 = profile_nfw.NFWProfile(M = M, c = c, z = z, mdef = mdef)
		rs = prof1.par['rs']
		rhos = prof1.par['rhos']
	
		# PROFILE 2: Only the density is analytical, the rest numerical
		prof2 = TestProfile(rhos = rhos, rs = rs)
		
		# PROFILES 3/4/5: User-defined NFW with finite resolution
		log_min = np.log10(r_min)
		log_max = np.log10(r_max)
		bin_width = (log_max - log_min) / N
		r_ = 10**np.arange(log_min, log_max + bin_width, bin_width)
		rho_ = prof1.density(r_)
		M_ = prof1.enclosedMass(r_)
		prof3 = profile_spline.SplineProfile(r = r_, rho = rho_)
		prof4 = profile_spline.SplineProfile(r = r_, M = M_)
		prof5 = profile_spline.SplineProfile(r = r_, rho = rho_, M = M_)
	
		# Test for all profiles
		profs = [prof1, prof2, prof3, prof4, prof5]
		prof_names = ['Reference', 'Numerical', 'ArrayRho', 'ArrayM', 'ArrayRhoM']
	
		if verbose:
			utilities.printLine()
			print(("Profile properties as a function of radius"))
			utilities.printLine()
			print(("Density"))
		
		for i in range(len(profs)):
			res = profs[i].density(r_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_RHO, 'Difference in density too large.')
				if verbose:
					print('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff))
							
		if verbose:
			utilities.printLine()
			print(("Density Linear Derivative"))
		
		for i in range(len(profs)):
			res = profs[i].densityDerivativeLin(r_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_DER, 'Difference in density derivative too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))
	
		if verbose:
			utilities.printLine()
			print(("Density Logarithmic Derivative"))
		
		for i in range(len(profs)):
			res = profs[i].densityDerivativeLog(r_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_DER, 'Difference in density log derivative too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))
		
		if verbose:
			utilities.printLine()
			print(("Enclosed mass"))
		
		for i in range(len(profs)):
			res = profs[i].enclosedMass(r_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_M, 'Difference in enclosed mass too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))
	
		if verbose:
			utilities.printLine()
			print(("Surface density"))
		
		for i in range(len(profs)):
			res = profs[i].surfaceDensity(r_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_SIGMA, 'Difference in surface density too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))
	
		if verbose:
			utilities.printLine()
			print(("Circular velocity"))
		
		for i in range(len(profs)):
			res = profs[i].circularVelocity(r_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_VCIRC, 'Difference in circular velocity too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))
		
		if verbose:
			utilities.printLine()
			print(("Rmax"))
		
		for i in range(len(profs)):
			_, res = profs[i].Vmax()
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_RMAX, 'Difference in Rmax too large.')
				if verbose:
					print('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff))
		
		if verbose:
			utilities.printLine()
			print(("Vmax"))
		
		for i in range(len(profs)):
			res, _ = profs[i].Vmax()
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_VMAX, 'Difference in Vmax too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))
	
		if verbose:
			utilities.printLine()
			print(("Spherical overdensity radii and masses"))
			utilities.printLine()
			print(("Spherical overdensity radius"))
		
		for i in range(len(profs)):
			res = profs[i].RDelta(z_test, mdef_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_SO_R, 'Difference in SO radius too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))
	
		if verbose:
			utilities.printLine()
			print(("Spherical overdensity mass"))
		
		for i in range(len(profs)):
			res = profs[i].MDelta(z_test, mdef_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_SO_M, 'Difference in SO mass too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
