###################################################################################################
#
# test_lss_peaks.py     (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest

from colossus.tests import test_colossus
from colossus.cosmology import cosmology
from colossus.lss import peaks

###################################################################################################
# TEST PARAMETERS
###################################################################################################

TEST_Z2 = 5.4
TEST_M = 3E12
TEST_R = 1.245
TEST_NU = 0.89
TEST_SLOPE_SCALE = 2.2

###################################################################################################
# TEST CASE 1: STANDARD PEAK FUNCTIONS
###################################################################################################

class TCPeaks(test_colossus.ColosssusTestCase):

	def setUp(self):
		self.cosmo_name = 'planck15'
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 
															'persistence': ''})
		pass
	
	def test_lagrangianR(self):
		self.assertAlmostEqual(peaks.lagrangianR(TEST_M), 2.029225828233e+00)

	def test_lagrangianM(self):
		self.assertAlmostEqual(peaks.lagrangianM(TEST_R), 6.928483143783e+11)

	def test_peakHeight(self):
		self.assertAlmostEqual(peaks.peakHeight(TEST_M, 0.0), 9.434114186555e-01)
		self.assertAlmostEqual(peaks.peakHeight(TEST_M, TEST_Z2), 4.741932629254e+00)

	def test_peakCurvature(self):
		correct = [[1.402997282813e+00, 6.309120539451e-01, 2.286757743003e+00, 1.401589845625e+00, -6.602831299662e-02], 
				[7.051980146273e+00, 6.309120539451e-01, 4.862963021325e+00, 4.137837428600e-01, 6.618287713862e+00]]
		for j in range(2):
			z = [0.0, TEST_Z2][j]
			res = peaks.peakCurvature(TEST_M, z, exact = False)
			for i in range(5):
				self.assertAlmostEqual(res[i], correct[j][i])

###################################################################################################
# TEST CASE 2: PEAK FUNCTIONS THAT NEED COSMOLOGICAL INTERPOLATION
###################################################################################################

class TCPeaksInterp(test_colossus.ColosssusTestCase):

	def setUp(self):
		self.cosmo_name = 'planck15'
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': True, 
															'persistence': ''})
		pass

	def test_massFromPeakHeight(self):
		self.assertAlmostEqual(peaks.massFromPeakHeight(TEST_NU, 0.0), 2.072290437038e+12)
		self.assertAlmostEqual(peaks.massFromPeakHeight(TEST_NU, TEST_Z2), 5.957743632443e+04)

	def test_nonLinearMass(self):
		self.assertAlmostEqual(peaks.nonLinearMass(1.1), 9.833246074319e+10)

	def test_powerSpectrumSlope(self):
		self.assertAlmostEqual(peaks.powerSpectrumSlope(TEST_NU, 0.0, slope_type = 'P', 
								scale = 1.0), -2.492990825082e+00)
		self.assertAlmostEqual(peaks.powerSpectrumSlope(TEST_NU, 0.0, slope_type = 'sigma', 
								scale = 1.0), -2.069673766363e+00)
		self.assertAlmostEqual(peaks.powerSpectrumSlope(TEST_NU, 0.0, slope_type = 'P', 
								scale = TEST_SLOPE_SCALE), -2.581375393787e+00)
		self.assertAlmostEqual(peaks.powerSpectrumSlope(TEST_NU, 0.0, slope_type = 'sigma', 
								scale = TEST_SLOPE_SCALE), -1.873258111430e+00)

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
