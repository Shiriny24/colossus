###################################################################################################
#
# test_halo_profile.py  (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest

from colossus.tests import test_colossus
from colossus.cosmology import cosmology

###################################################################################################

TEST_N_DIGITS = test_colossus.TEST_N_DIGITS

###################################################################################################
# TEST CASE: SPHERICAL OVERDENSITY
###################################################################################################

class TCProfileBase(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15')
		pass

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
