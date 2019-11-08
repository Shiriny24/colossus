###################################################################################################
#
# test_lss_peaks.py     (c) Benedikt Diemer
#     				    	diemer@umd.edu
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
		self.assertAlmostEqual(peaks.lagrangianR(TEST_M), 2.029075203793e+00)

	def test_lagrangianM(self):
		self.assertAlmostEqual(peaks.lagrangianM(TEST_R), 6.930026225629e+11)

	def test_peakHeight(self):
		self.assertAlmostEqual(peaks.peakHeight(TEST_M, 0.0), 9.433778391351e-01)
		self.assertAlmostEqual(peaks.peakHeight(TEST_M, TEST_Z2), 4.741763836029e+00)

	def test_peakCurvature(self):
		correct = [[1.402937353994e+00, 6.309081004982e-01, 2.286742816988e+00, 
					1.401618275862e+00, -6.609666016081e-02], 
				[7.051678906812e+00, 6.309081004982e-01, 4.862772322171e+00, 
					4.138109777506e-01, 6.617964241261e+00]]
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
		self.assertAlmostEqual(peaks.massFromPeakHeight(TEST_NU, 0.0), 2.072751968615e+12)
		self.assertAlmostEqual(peaks.massFromPeakHeight(TEST_NU, TEST_Z2), 5.959070789523e+04)

	def test_nonLinearMass(self):
		self.assertAlmostEqual(peaks.nonLinearMass(1.1), 9.835436092511e+10)

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
