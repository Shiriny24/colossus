###################################################################################################
#
# test_lss_lss.py       (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest

from colossus.tests import test_colossus
from colossus.cosmology import cosmology
from colossus.lss import lss

###################################################################################################
# TEST PARAMETERS
###################################################################################################

TEST_Z2 = 5.4
TEST_M = 3E12
TEST_R = 1.245
TEST_NU = 0.89

###################################################################################################
# TEST CASE 1: STANDARD LSS FUNCTIONS
###################################################################################################

class TCLss(test_colossus.ColosssusTestCase):

	def setUp(self):
		self.cosmo_name = 'planck15'
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 'storage': ''})
		pass
	
	def test_lagrangianR(self):
		self.assertAlmostEqual(lss.lagrangianR(TEST_M), 2.0292015228231484)

	def test_lagrangianM(self):
		self.assertAlmostEqual(lss.lagrangianM(TEST_R), 692873211113.4847)

	def test_peakHeight(self):
		self.assertAlmostEqual(lss.peakHeight(TEST_M, 0.0), 0.94312293214221243)
		self.assertAlmostEqual(lss.peakHeight(TEST_M, TEST_Z2), 4.7404825899781677)

	def test_peakCurvature(self):
		correct = [[1.7282851398751131, 0.64951660047324522, 2.3770980172992502, 1.2545481285991393, 0.31882292066585705], 
				[8.6869965058389198, 0.64951660047324511, 5.9443788502296497, 0.30203041143419568, 8.3476707802933614]]
		for j in range(2):
			z = [0.0, TEST_Z2][j]
			res = lss.peakCurvature(TEST_M, z)
			for i in range(5):
				self.assertAlmostEqual(res[i], correct[j][i])

###################################################################################################
# TEST CASE 2: LSS FUNCTIONS THAT NEED COSMOLOGICAL INTERPOLATION
###################################################################################################

class TCLssInterp(test_colossus.ColosssusTestCase):

	def setUp(self):
		self.cosmo_name = 'planck15'
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': True, 'storage': ''})
		pass

	def test_massFromPeakHeight(self):
		self.assertAlmostEqual(lss.massFromPeakHeight(TEST_NU, 0.0), 2.077136472813e+12)
		self.assertAlmostEqual(lss.massFromPeakHeight(TEST_NU, TEST_Z2), 59607.184484321471)

	def test_nonLinearMass(self):
		self.assertAlmostEqual(lss.nonLinearMass(1.1), 98435044937.058929)				

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
