###################################################################################################
#
# test_colossus.py      (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest

###################################################################################################
# CONSTANTS FOR ALL TESTS
###################################################################################################

TEST_N_DIGITS = 8

###################################################################################################
# UNIT TEST CLASS FOR ALL COLOSSUS TESTS
###################################################################################################

class ColosssusTestCase(unittest.TestCase):
	
	def assertAlmostEqualArray(self, first, second, places = None, msg = None, delta = None):
		N1 = len(first)
		if N1 != len(second):
			raise Exception('Length of arrays must be the same.')
		for i in range(N1):
			msg = 'Array element %d/%d' % (i + 1, N1)
			self.assertAlmostEqual(first[i], second[i], places = places, msg = msg, delta = delta)

###################################################################################################
