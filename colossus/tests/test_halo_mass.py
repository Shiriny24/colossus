###################################################################################################
#
# test_halo_mass.py     (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest

from colossus.tests import test_colossus
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.halo import mass_defs
from colossus.halo import mass_adv

###################################################################################################

TEST_N_DIGITS = test_colossus.TEST_N_DIGITS

###################################################################################################
# TEST CASE: SPHERICAL OVERDENSITY
###################################################################################################

class TCMassSO(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15')
		pass

	def test_parseMassDefinition(self):
		t, d = mass_so.parseMassDefinition('200m')
		self.assertEqual(t, 'm')
		self.assertEqual(d, 200)
		t, d = mass_so.parseMassDefinition('500c')
		self.assertEqual(t, 'c')
		self.assertEqual(d, 500)
		t, d = mass_so.parseMassDefinition('vir')
		self.assertEqual(t, 'vir')
		self.assertEqual(d, None)
		with self.assertRaises(Exception):
			mass_so.parseMassDefinition('100r')
			mass_so.parseMassDefinition('e')
			mass_so.parseMassDefinition('79.6c')

	def test_parseRadiusMassDefinition(self):
		rm, _, t, d = mass_so.parseRadiusMassDefinition('R200m')
		self.assertEqual(rm, 'R')
		self.assertEqual(t, 'm')
		self.assertEqual(d, 200)
		rm, _, t, d = mass_so.parseRadiusMassDefinition('r200m')
		self.assertEqual(rm, 'R')
		self.assertEqual(t, 'm')
		self.assertEqual(d, 200)
		rm, _, t, d = mass_so.parseRadiusMassDefinition('M500c')
		self.assertEqual(rm, 'M')
		self.assertEqual(t, 'c')
		self.assertEqual(d, 500)
		rm, _, t, d = mass_so.parseRadiusMassDefinition('Mvir')
		self.assertEqual(rm, 'M')
		self.assertEqual(t, 'vir')
		self.assertEqual(d, None)
		with self.assertRaises(Exception):
			mass_so.parseRadiusMassDefinition('e500c')
			mass_so.parseRadiusMassDefinition('e')
			mass_so.parseRadiusMassDefinition('79.6c')

	def test_densityThreshold(self):
		self.assertAlmostEqual(mass_so.densityThreshold(0.7, '200m'), 84223.612767872, places = TEST_N_DIGITS)
		self.assertAlmostEqual(mass_so.densityThreshold(6.1, '400c'), 12373756.401747715, places = TEST_N_DIGITS)
		self.assertAlmostEqual(mass_so.densityThreshold(1.2, 'vir'), 179234.67533064212, places = TEST_N_DIGITS)
		with self.assertRaises(Exception):
			mass_so.densityThreshold('100t')

	def test_deltaVir(self):
		self.assertAlmostEqual(mass_so.deltaVir(0.7), 148.15504207273736, places = TEST_N_DIGITS)
	
	def test_M_to_R(self):
		self.assertAlmostEqual(mass_so.M_to_R(1.1E12, 0.7, '200m'), 146.09098023845536, places = TEST_N_DIGITS)
		self.assertAlmostEqual(mass_so.M_to_R(1.1E12, 0.7, 'vir'), 142.45956950993343, places = TEST_N_DIGITS)

	def test_R_to_M(self):
		self.assertAlmostEqual(mass_so.R_to_M(212.0, 0.7, '200m'), 3361476338653.47, places = TEST_N_DIGITS)
		self.assertAlmostEqual(mass_so.R_to_M(150.0, 0.7, 'vir'), 1284078514739.949, places = TEST_N_DIGITS)

###################################################################################################
# TEST CASE: DEFINITIONS
###################################################################################################

class TCMassDefs(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15')
		pass
	
	def test_pseudoEvolve(self):
		z1 = 0.68
		z2 = 3.1
		M1 = [1.5E8, 1.1E15]
		c1 = 4.6
		correct_M = [44584660.05778446, 326954173757086.38]
		correct_R = [2.1518177857985319, 418.06065970441745]
		correct_c = [1.3008705755189411, 1.300870575518942]
		for i in range(len(M1)):
			M, R, c = mass_defs.pseudoEvolve(M1[i], c1, z1, '200m', z2, 'vir')
			self.assertAlmostEqual(M, correct_M[i], places = TEST_N_DIGITS)
			self.assertAlmostEqual(R, correct_R[i], places = TEST_N_DIGITS)
			self.assertAlmostEqual(c, correct_c[i], places = TEST_N_DIGITS)

		return

	def test_changeMassDefinition(self):
		z1 = 0.98
		M1 = [1.5E8, 1.1E15]
		c1 = 4.6
		correct_M = [118946488.07233413, 872274245863784.38]
		correct_R = [4.7970184134005516, 931.97699905443392]
		correct_c = [3.4334724175833529, 3.433472417583356]
		for i in range(len(M1)):
			M, R, c = mass_defs.changeMassDefinition(M1[i], c1, z1, 'vir', '300c')
			self.assertAlmostEqual(M, correct_M[i], places = TEST_N_DIGITS)
			self.assertAlmostEqual(R, correct_R[i], places = TEST_N_DIGITS)
			self.assertAlmostEqual(c, correct_c[i], places = TEST_N_DIGITS)

		return
	
###################################################################################################
# TEST CASE: ADVANCED
###################################################################################################

class TCMassAdv(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15')
		pass

	def test_changeMassDefinitionCModel(self):
		z1 = 0.98
		M1 = [1.5E8, 1.1E15]
		correct_M = [129838642.85775913, 879043735681505.62]
		correct_R = [4.9391873517684282, 934.38173127832636]
		correct_c = [9.0766320379479737, 3.6654767598601383]
		for i in range(len(M1)):
			M, R, c = mass_adv.changeMassDefinitionCModel(M1[i], z1, 'vir', '300c')
			self.assertAlmostEqual(M, correct_M[i], places = TEST_N_DIGITS)
			self.assertAlmostEqual(R, correct_R[i], places = TEST_N_DIGITS)
			self.assertAlmostEqual(c, correct_c[i], places = TEST_N_DIGITS)

		return
	
	def test_M4rs(self):
		self.assertAlmostEqual(mass_adv.M4rs(1E12, 0.7, '500c', 3.8), 1041815679897.7153, places = TEST_N_DIGITS)

	def test_RspOverR200m(self):
		self.assertAlmostEqual(mass_adv.RspOverR200m(nu200m = 2.4, z = None, Gamma = None), 1.1038203372651374, places = TEST_N_DIGITS)
		self.assertAlmostEqual(mass_adv.RspOverR200m(nu200m = 2.4, z = 1.4, Gamma = 0.8), 1.6080082344281803, places = TEST_N_DIGITS)
				
	def test_MspOverM200m(self):
		self.assertAlmostEqual(mass_adv.MspOverM200m(nu200m = 2.4, z = None, Gamma = None), 1.0812429190621984, places = TEST_N_DIGITS)
		self.assertAlmostEqual(mass_adv.MspOverM200m(nu200m = 2.4, z = 1.4, Gamma = 0.8), 1.359753402031187, places = TEST_N_DIGITS)

	def test_Rsp(self):
		self.assertAlmostEqual(mass_adv.Rsp(450.2, 0.4, '200m'), 549.64492801853442, places = TEST_N_DIGITS)

	def test_Msp(self):
		self.assertAlmostEqual(mass_adv.Msp(1.1E12, 0.4, '200m'), 1330106019812.7119, places = TEST_N_DIGITS)
	
###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
