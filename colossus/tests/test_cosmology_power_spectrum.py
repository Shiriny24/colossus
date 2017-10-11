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
from colossus.cosmology import power_spectrum

###################################################################################################
# TEST PARAMETERS
###################################################################################################

TEST_K = np.array([1.2E-3, 1.1E3])

###################################################################################################
# TEST CASES
###################################################################################################

class TCTransferFunction(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15')
		pass
	
	def test_transferFunctionEisenstein98(self):
		models = power_spectrum.models
		cosmo = cosmology.getCurrent()
		for i in range(len(models)):
			msg = 'Failure in model = %s' % (models[i])
			if models[i] == 'eisenstein98':
				correct = [9.892256929454e-01, 1.490479340442e-08]
			elif models[i] == 'eisenstein98_zb':
				correct = [9.890484789741e-01, 1.471045424612e-08]
			else:
				msg = 'Unknown model, %s.' % models[i]
				raise Exception(msg)
			T = power_spectrum.transferFunction(TEST_K, cosmo.h, cosmo.Om0, cosmo.Ob0, cosmo.Tcmb0, 
										model = models[i])
			self.assertAlmostEqualArray(T, correct, msg = msg)

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
