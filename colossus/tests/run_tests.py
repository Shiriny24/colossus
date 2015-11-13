###################################################################################################
#
# run_tests.py          (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest

from colossus.tests import test_cosmology
from colossus.tests import test_utils

###################################################################################################

suite_cosmo1 = unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCComp)
suite_cosmo2 = unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCInterp)

suite_utils = unittest.TestLoader().loadTestsFromTestCase(test_utils.TCGen)

suite = unittest.TestSuite([suite_cosmo1, suite_cosmo2, suite_utils])
unittest.TextTestRunner(verbosity = 2).run(suite)
