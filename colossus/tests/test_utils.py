###################################################################################################
#
# test_utils.py         (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest
import numpy as np

from colossus.tests import test_colossus
from colossus.utils import storage
from colossus.utils import utilities
from colossus.utils import constants

###################################################################################################
# TEST CASES
###################################################################################################

class TCGen(test_colossus.ColosssusTestCase):

	def setUp(self):
		pass
	
	def test_home_dir(self):
		self.assertNotEqual(storage.getCacheDir(), None)
		
class TCVersions(test_colossus.ColosssusTestCase):

	def setUp(self):
		pass
	
	def test_versions(self):
		self.assertEqual(utilities.versionIsOlder('1.0.1', '1.0.0') , True)
		self.assertEqual(utilities.versionIsOlder('1.1.0', '1.0.0') , True)
		self.assertEqual(utilities.versionIsOlder('2.0.0', '1.0.0') , True)
		self.assertEqual(utilities.versionIsOlder('1.0.0', '1.0.0') , False)
		self.assertEqual(utilities.versionIsOlder('1.0.0', '2.0.0') , False)
		self.assertEqual(utilities.versionIsOlder('1.0.0', '1.1.0') , False)
		self.assertEqual(utilities.versionIsOlder('1.0.0', '1.0.1') , False)

class TCConstants(test_colossus.ColosssusTestCase):

	def setUp(self):
		pass
	
	def test_gravitational_constant(self):
		G_const = constants.G / 1000.0
		G_deriv = constants.G_CGS / constants.MPC * constants.MSUN / 1E10
		self.assertAlmostEqual(G_const, G_deriv, places = 9)

		G_const = constants.G
		G_deriv = constants.G_CGS / constants.KPC * constants.MSUN / 1E10
		self.assertAlmostEqual(G_const, G_deriv, places = 9)

	def test_critical_density(self):
		G_Mpc = constants.G / 1000.0
		rhoc_deriv = 3 * 100.0**2 / (8 * np.pi * G_Mpc)
		rhoc_const = constants.RHO_CRIT_0_MPC3
		self.assertAlmostEqual(rhoc_const, rhoc_deriv, places = 9)

		H0_kpc = 100.0 / 1000.0
		rhoc_deriv = 3 * H0_kpc**2 / (8 * np.pi * constants.G)
		rhoc_const = constants.RHO_CRIT_0_KPC3
		self.assertAlmostEqual(rhoc_const, rhoc_deriv, places = 9)

	def test_deltac(self):
		deltac_deriv = 3.0 / 5.0 * (3.0 * np.pi / 2.0)**(2.0 / 3.0)
		deltac_const = constants.DELTA_COLLAPSE
		self.assertAlmostEqual(deltac_const, deltac_deriv, places = 5)

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
