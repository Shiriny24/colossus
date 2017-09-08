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
		self.assertAlmostEqual(lss.peakHeight(TEST_M, 0.0), 9.431430836525e-01)
		self.assertAlmostEqual(lss.peakHeight(TEST_M, TEST_Z2), 4.740583878878e+00)

	def test_peakCurvature(self):
		correct = [[1.728295409748e+00, 6.498395946425e-01, 2.376881414525e+00, 1.253766626032e+00, 3.179869335955e-01], 
				[8.687048125995e+00, 6.498395946425e-01, 5.946828630668e+00, 3.016407978311e-01, 8.347745288368e+00]]
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
		self.assertAlmostEqual(lss.massFromPeakHeight(TEST_NU, 0.0), 2.076094075382e+12)
		self.assertAlmostEqual(lss.massFromPeakHeight(TEST_NU, TEST_Z2), 5.961576593638e+04)

	def test_nonLinearMass(self):
		self.assertAlmostEqual(lss.nonLinearMass(1.1), 9.853011711832e+10)

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
