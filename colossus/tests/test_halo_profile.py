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
from colossus.halo import mass_so
from colossus.halo import profile_outer
from colossus.halo import profile_nfw
from colossus.halo import profile_einasto
from colossus.halo import profile_dk14
from colossus.halo import profile_base
from colossus.halo import profile_spline
from colossus.halo import concentration

###################################################################################################
# CONSTANTS
###################################################################################################

# For some test cases in the profile unit, we cannot expect the results to agree to very high 
# precision because numerical approximations are made.

TEST_N_DIGITS_LOW = 4

###################################################################################################
# TEST CASE: BASE CLASS
###################################################################################################

# This test case compares three different implementations of the NFW density profile: 
# - the exact, analytic form
# - the generic implementation of the HaloDensityProfile base class, where only the density is 
#   computed analytically, but all other functions numerically ('Numerical')
# - a discrete profile where density and/or mass are given as arrays. Three cases are tested, with
#   only rho, only M, and both ('ArrayRho', 'ArrayM', and 'ArrayRhoM').

class TCBase(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('WMAP9', {'persistence': ''})
		self.MAX_DIFF_RHO = 1E-8
		self.MAX_DIFF_M = 1E-8
		self.MAX_DIFF_DER = 1E-2
		self.MAX_DIFF_SIGMA = 1E-5
		self.MAX_DIFF_VCIRC = 1E-8
		self.MAX_DIFF_RMAX = 1E-3
		self.MAX_DIFF_VMAX = 1E-8
		self.MAX_DIFF_SO_R = 1E-7
		self.MAX_DIFF_SO_M = 1E-7
	
	def test_base_nfw(self, verbose = False):

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
# TEST CASE: PROFILE VALUES FOR INNER PROFILES
###################################################################################################

class TCInner(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('WMAP9', {'persistence': ''})
		
		M = 4E14
		c = 5.7
		mdef = '200c'
		z = 0.2
		
		self.p = []
		self.p.append(profile_nfw.NFWProfile(M = M, c = c, mdef = mdef, z = z))
		self.p.append(profile_einasto.EinastoProfile(M = M, c = c, mdef = mdef, z = z))
		self.p.append(profile_dk14.DK14Profile(M = M, c = c, mdef = mdef, z = z))
	
	def test_inner(self, verbose = False):
		
		r = 576.2

		correct_rho = [87808.770952624793, 88484.228470315546, 89374.936828127422]
		correct_Menc = [236306365549189.75, 242969347272094.19, 247265891283410.44]
		correct_Sigma = [1.146313109242e+08, 1.071274995675e+08, 1.007159590602e+08]
		correct_DeltaSigma = [1.857504107214e+08, 1.923576934645e+08, 1.964754638958e+08]
		correct_derLin = [-379.41823828559359, -394.55464355743817, -404.29227795786403]
		correct_derLog = [-2.489737488958943, -2.5692983885152381, -2.6064713310768792]
		correct_vcirc = [1328.2265364218915, 1346.8219502791576, 1358.6780165060245]
		correct_vmax = [1339.0321613005094, 1360.357622100061, 1373.1897650926066]
		correct_rdelta = [1010.9220128231149, 1014.0087194079377, 1016.4548249128767]

		for i in range(len(self.p)):
			
			q = self.p[i].density(r)
			self.assertAlmostEqual(q, correct_rho[i], places = TEST_N_DIGITS_LOW)
			
			q = self.p[i].enclosedMass(r)
			self.assertAlmostEqual(q, correct_Menc[i], places = TEST_N_DIGITS_LOW)

			q = self.p[i].surfaceDensity(r)
			self.assertAlmostEqual(q, correct_Sigma[i], places = TEST_N_DIGITS_LOW)

			q = self.p[i].deltaSigma(r)
			self.assertAlmostEqual(q, correct_DeltaSigma[i], places = TEST_N_DIGITS_LOW)

			q = self.p[i].densityDerivativeLin(r)
			self.assertAlmostEqual(q, correct_derLin[i], places = TEST_N_DIGITS_LOW)

			q = self.p[i].densityDerivativeLog(r)
			self.assertAlmostEqual(q, correct_derLog[i], places = TEST_N_DIGITS_LOW)

			q = self.p[i].circularVelocity(r)
			self.assertAlmostEqual(q, correct_vcirc[i], places = TEST_N_DIGITS_LOW)

			q, _ = self.p[i].Vmax()
			self.assertAlmostEqual(q, correct_vmax[i], places = TEST_N_DIGITS_LOW)

			q = self.p[i].RDelta(0.7, mdef = 'vir')
			self.assertAlmostEqual(q, correct_rdelta[i], places = TEST_N_DIGITS_LOW)

###################################################################################################
# TEST CASE: OUTER PROFILES
###################################################################################################

class TCOuter(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('WMAP9', {'persistence': ''})

		z = 0.2
		M = 4E12
		c = 5.7
		mdef = '200c'

		self.t = []
		self.t.append(profile_outer.OuterTermMeanDensity(z = z))
		self.t.append(profile_outer.OuterTermCorrelationFunction(z = z, bias = 2.2))
		self.t.append(profile_outer.OuterTermPowerLaw(z = z, norm = 2.0, slope = 1.4, 
										max_rho = 1200.0, pivot = 'fixed', pivot_factor = 257.0))
		
		self.p = []
		for i in range(len(self.t)):
			self.p.append(profile_nfw.NFWProfile(M = M, c = c, mdef = mdef, z = z, outer_terms = [self.t[i]]))
	
	def test_outer(self, verbose = False):
		
		r = 980.2

		correct_rho = [4.326743100631e+02, 1.422357724157e+03, 3.374673592803e+02]
		correct_der = [-8.787573448793e-01, -1.879908075296e+00, -9.389703075811e-01]

		for i in range(len(self.p)):
			
			q = self.p[i].density(r)
			self.assertAlmostEqual(q, correct_rho[i])

			q = self.p[i].densityDerivativeLin(r)
			self.assertAlmostEqual(q, correct_der[i])

###################################################################################################
# TEST CASE: FITTING
###################################################################################################

class TCFitting(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('WMAP9', {'persistence': ''})
		M = 1E12
		c = 6.0
		mdef = 'vir'
		z = 0.0
		self.p = profile_nfw.NFWProfile(M = M, c = c, z = z, mdef = mdef)
	
	def test_leastsq(self, verbose = False):

		scatter = 0.001
		r = 10**np.arange(0.1, 3.6, 0.1)
		mask = np.array([True, True])
		q_true = self.p.density(r)
		scatter_sigma = scatter * 0.3
		np.random.seed(157)
		q_err = np.abs(np.random.normal(scatter, scatter_sigma, (len(r)))) * q_true
		q = q_true.copy()
		for i in range(len(r)):
			q[i] += np.random.normal(0.0, q_err[i])
		x_true = self.p.getParameterArray(mask)
		ini_guess = x_true * 1.5
		self.p.setParameterArray(ini_guess, mask = mask)
		dict = self.p.fit(r, q, 'rho', q_err = q_err, verbose = False, mask = mask, tolerance = 1E-6)
		x = self.p.getParameterArray(mask = mask)
		acc = abs(x / x_true - 1.0)
		
		self.assertLess(acc[0], 1E-2)
		self.assertLess(acc[1], 1E-2)

###################################################################################################
# TEST CASE: NFW SPECIAL FUNCTIONS
###################################################################################################
	
class TCNFW(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('WMAP9', {'persistence': ''})
				
	def test_pdf(self):
		
		M = 10**np.arange(9.0, 15.5, 0.2)
		mdef = 'vir'
		z = 0.0
		c = concentration.concentration(M, mdef, z)
		N = len(M)
		p = np.random.uniform(0.0, 1.0, (N))
		r1 = profile_nfw.radiusFromPdf(M, c, z, mdef, p, interpolate = False)
		r2 = profile_nfw.radiusFromPdf(M, c, z, mdef, p, interpolate = True)
		R = mass_so.M_to_R(M, z, mdef)
		rs = R / c
		p1 = profile_nfw.NFWProfile.mu(r1 / rs) / profile_nfw.NFWProfile.mu(c)
		p2 = profile_nfw.NFWProfile.mu(r2 / rs) / profile_nfw.NFWProfile.mu(c)
		diff1 = np.max(np.abs(p1 / p - 1.0))
		diff2 = np.max(np.abs(p2 / p - 1.0))
		
		self.assertLess(diff1, 1E-8)
		self.assertLess(diff2, 1E-2)

###################################################################################################
# TEST CASE: DK14 SPECIAL FUNCTIONS
###################################################################################################

# This test case checks whether the iterative setting of R200m works in the DK14 profile

class TCDK14(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15', {'persistence': ''})
		self.M = 1E14
		self.c = 7.0
		self.z = 1.0
		self.mdef = '200c'
	
	# Test 1: No outer terms	
	def test_DK14ConstructorInner(self):
		
		p1 = profile_dk14.DK14Profile(M = self.M, c = self.c, z = self.z, mdef = self.mdef)
		p2 = profile_dk14.DK14Profile(z = self.z, 
					rhos = p1.par['rhos'], rs = p1.par['rs'], rt = p1.par['rt'], 
					alpha = p1.par['alpha'], beta = p1.par['beta'], gamma = p1.par['gamma'])
		self.assertAlmostEqual(p1.opt['R200m'], p2.opt['R200m'], places = 3)

	# Test 2: With outer terms
	def test_DK14ConstructorOuter(self):

		power_law_norm = 3.0
		ot_pl = profile_outer.OuterTermPowerLaw(norm = power_law_norm, 
								slope = 1.5, pivot = 'R200m', pivot_factor = 5.0, z = self.z,
								max_rho = 1000.0)
		p1 = profile_dk14.DK14Profile(M = self.M, c = self.c, z = self.z, mdef = self.mdef, 
									outer_terms = [ot_pl])
		p2 = profile_dk14.DK14Profile(z = self.z, 
					rhos = p1.par['rhos'], rs = p1.par['rs'], rt = p1.par['rt'], 
					alpha = p1.par['alpha'], beta = p1.par['beta'], gamma = p1.par['gamma'], 
					outer_terms = [ot_pl])
		self.assertAlmostEqual(p1.opt['R200m'], p2.opt['R200m'], places = 3)

	# Test 3: Using wrapper function
	def test_DK14ConstructorWrapper(self):

		power_law_norm = 3.0
		p1 = profile_dk14.getDK14ProfileWithOuterTerms(M = self.M, c = self.c, z = self.z, mdef = self.mdef, 
					outer_term_names = ['pl'], power_law_norm = power_law_norm)
		p2 = profile_dk14.getDK14ProfileWithOuterTerms(z = self.z, 
					rhos = p1.par['rhos'], rs = p1.par['rs'], rt = p1.par['rt'], 
					alpha = p1.par['alpha'], beta = p1.par['beta'], gamma = p1.par['gamma'], 
					outer_term_names = ['pl'], power_law_norm = power_law_norm)
		self.assertAlmostEqual(p1.opt['R200m'], p2.opt['R200m'], places = 3)

	# Test 4: Using the correlation function. This test is skipped because it is very slow
	
# 	def test_DK14ConstructorCorrelationFunction(self):
# 		
# 		ot_cf = profile_outer.OuterTermCorrelationFunction(derive_bias_from = 'R200m', z = self.z)
# 		p1 = profile_dk14.DK14Profile(M = self.M, c = self.c, z = self.z, mdef = self.mdef, 
# 									outer_terms = [ot_cf])
# 		p2 = profile_dk14.DK14Profile(z = self.z, 
# 					rhos = p1.par['rhos'], rs = p1.par['rs'], rt = p1.par['rt'], 
# 					alpha = p1.par['alpha'], beta = p1.par['beta'], gamma = p1.par['gamma'], 
# 					outer_terms = [ot_cf])
# 		self.assertAlmostEqual(p1.opt['R200m'], p2.opt['R200m'], places = 3)

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
