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
		self.assertAlmostEqual(peaks.lagrangianR(TEST_M), 2.0292015228231484)

	def test_lagrangianM(self):
		self.assertAlmostEqual(peaks.lagrangianM(TEST_R), 692873211113.4847)

	def test_peakHeight(self):
		self.assertAlmostEqual(peaks.peakHeight(TEST_M, 0.0), 9.434060001705e-01)
		self.assertAlmostEqual(peaks.peakHeight(TEST_M, TEST_Z2), 4.741905393957e+00)

	def test_peakCurvature(self):
		correct = [[1.728777200283e+00, 6.498395946425e-01, 2.377044934137e+00, 1.253617059077e+00, 3.186369656037e-01], 
				[8.689469782353e+00, 6.498395946425e-01, 5.948308808583e+00, 3.015472875606e-01, 8.350272130432e+00]]
		for j in range(2):
			z = [0.0, TEST_Z2][j]
			res = peaks.peakCurvature(TEST_M, z)
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
		self.assertAlmostEqual(peaks.massFromPeakHeight(TEST_NU, 0.0), 2.072364902494e+12)
		self.assertAlmostEqual(peaks.massFromPeakHeight(TEST_NU, TEST_Z2), 5.932394446473e+04)

	def test_nonLinearMass(self):
		self.assertAlmostEqual(peaks.nonLinearMass(1.1), 9.830911710403e+10)

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
