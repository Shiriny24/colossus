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
from colossus.tests import test_halo_concentration
from colossus.tests import test_halo_mass

###################################################################################################

suites = []

suites.append(unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCComp))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCInterp))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCNotFlat))

suites.append(unittest.TestLoader().loadTestsFromTestCase(test_utils.TCGen))

suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_bias.TCBias))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_concentration.TCConcentration))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_mass.TCMassSO))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_mass.TCMassDefs))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_mass.TCMassAdv))

suite = unittest.TestSuite(suites)
unittest.TextTestRunner(verbosity = 2).run(suite)
