###################################################################################################
#
# test_lss_bias.py      (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest
import numpy as np

from colossus.tests import test_colossus
from colossus.cosmology import cosmology
from colossus.lss import bias

###################################################################################################
# TEST CASES
###################################################################################################

class TCBias(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15')
		pass
	
	def test_haloBiasFromNu(self):
		self.assertAlmostEqual(bias.haloBiasFromNu(3.0, 1.0, '200c'), 5.290627688108e+00)
		self.assertAlmostEqual(bias.haloBiasFromNu(3.0, 1.0, '200c', model = 'cole89'), 5.743636115674e+00)
		self.assertAlmostEqual(bias.haloBiasFromNu(3.0, 1.0, '200c', model = 'sheth01'), 4.720384850956e+00)
	
	def test_haloBias(self):
		self.assertAlmostEqual(bias.haloBias(2.3E12, 1.0, '200c'), 1.552991796791e+00)
		
	def test_twoHaloTerm(self):
		r = np.array([1.2, 10.8, 101.0])
		correct = np.array([4.573596528967e+04, 2.073245325281e+04, 6.739324203655e+03])
		self.assertAlmostEqualArray(bias.twoHaloTerm(r, 2.3E12, 1.0, '200c'), correct)

	def test_modelCole89(self):
		self.assertAlmostEqual(bias.modelCole89(3.0), 5.743636115674e+00)

	def test_modelSheth01(self):
		self.assertAlmostEqual(bias.modelSheth01(3.0), 4.720384850956e+00)
		
	def test_modelTinker10(self):
		self.assertAlmostEqual(bias.modelTinker10(3.0, 1.0, '200c'), 5.290627688108e+00)
		
###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
