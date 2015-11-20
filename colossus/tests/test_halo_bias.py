###################################################################################################
#
# test_halo_bias.py     (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest
import numpy as np

from colossus.tests import test_colossus
from colossus.cosmology import cosmology
from colossus.halo import bias

###################################################################################################
# TEST CASES
###################################################################################################

class TCBias(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15')
		pass
	
	def test_haloBiasFromNu(self):
		self.assertAlmostEqual(bias.haloBiasFromNu(3.0, 1.0, '200c'), 5.2906151991178554)
	
	def test_haloBias(self):
		self.assertAlmostEqual(bias.haloBias(2.3E12, 1.0, '200c'), 1.5522875685982045)
		
	def test_twoHaloTerm(self):
		r = np.array([1.2, 10.8, 101.0])
		correct = np.array([45715.732274087895, 20723.281476352931, 6736.342811423111])
		self.assertAlmostEqualArray(bias.twoHaloTerm(r, 2.3E12, 1.0, '200c'), correct)
		
	def test_modelTinker10(self):
		self.assertAlmostEqual(bias.modelTinker10(3.0, 1.0, '200c'), 5.2906151991178554)
		
###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
