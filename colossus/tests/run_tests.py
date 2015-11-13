###################################################################################################
#
# run_tests.py          (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest

from colossus.tests import test_cosmology
from colossus.tests import test_utils
from colossus.tests import test_halo_bias

###################################################################################################

suite_cosmo1 = unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCComp)
suite_cosmo2 = unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCInterp)

suite_utils = unittest.TestLoader().loadTestsFromTestCase(test_utils.TCGen)

suite_halo_bias = unittest.TestLoader().loadTestsFromTestCase(test_halo_bias.TCBias)

suite = unittest.TestSuite([suite_cosmo1, suite_cosmo2, suite_utils, suite_halo_bias])
unittest.TextTestRunner(verbosity = 2).run(suite)
