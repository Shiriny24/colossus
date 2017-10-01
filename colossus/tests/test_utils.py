###################################################################################################
#
# test_utils.py         (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest

from colossus.tests import test_colossus
from colossus.utils import storage

###################################################################################################
# TEST CASES
###################################################################################################

class TCGen(test_colossus.ColosssusTestCase):

	def setUp(self):
		pass
	
	def test_home_dir(self):
		self.assertNotEqual(storage.getCacheDir(), None)
		
###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
