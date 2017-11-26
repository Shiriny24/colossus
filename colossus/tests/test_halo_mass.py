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
from colossus.halo import profile_dk14
from colossus.halo import profile_outer

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
		self.assertAlmostEqual(mass_so.densityThreshold(0.7, '200m'), 84223.612767872)
		self.assertAlmostEqual(mass_so.densityThreshold(6.1, '400c'), 12373756.401747715)
		self.assertAlmostEqual(mass_so.densityThreshold(1.2, 'vir'), 179234.67533064212)
		with self.assertRaises(Exception):
			mass_so.densityThreshold('100t')

	def test_deltaVir(self):
		self.assertAlmostEqual(mass_so.deltaVir(0.7), 148.15504207273736)
	
	def test_M_to_R(self):
		self.assertAlmostEqual(mass_so.M_to_R(1.1E12, 0.7, '200m'), 146.09098023845536)
		self.assertAlmostEqual(mass_so.M_to_R(1.1E12, 0.7, 'vir'), 142.45956950993343)

	def test_R_to_M(self):
		self.assertAlmostEqual(mass_so.R_to_M(212.0, 0.7, '200m'), 3361476338653.47)
		self.assertAlmostEqual(mass_so.R_to_M(150.0, 0.7, 'vir'), 1284078514739.949)

###################################################################################################
# TEST CASE: DEFINITIONS
###################################################################################################

class TCMassDefs(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15')
	
	def test_pseudoEvolve(self):
		z1 = 0.68
		z2 = 3.1
		M1 = [1.5E8, 1.1E15]
		c1 = 4.6
		correct_M = [44584660.05778446, 326954173757086.38]
		correct_R = [2.1518177857985319, 418.06065970441745]
		correct_c = [1.3008705755189411, 1.300870575518942]
		for i in range(len(M1)):
			M, R, c = mass_defs.evolveSO(M1[i], c1, z1, '200m', z2, 'vir')
			self.assertAlmostEqual(M, correct_M[i])
			self.assertAlmostEqual(R, correct_R[i])
			self.assertAlmostEqual(c, correct_c[i])

	def test_pseudoEvolveWithDk14(self):
		z1 = 0.68
		z2 = 3.1
		M1 = [1.5E8, 1.1E15]
		c1 = 4.6
		correct_M = [4.111592480994e+07, 4.333529117746e+14]
		correct_R = [2.094499980659e+00, 4.592236498407e+02]
		correct_c = [1.266219385882e+00, 1.428956587502e+00]
		for i in range(len(M1)):
			t = profile_outer.OuterTermPowerLaw(norm = 1.0, slope = 1.5, pivot = 'R200m', 
											pivot_factor = 5.0, z = 0.0)
			M, R, c = mass_defs.evolveSO(M1[i], c1, z1, '200m', z2, 'vir',
						profile = profile_dk14.DK14Profile, profile_args = {'outer_terms': [t]})
			self.assertAlmostEqual(M, correct_M[i])
			self.assertAlmostEqual(R, correct_R[i])
			self.assertAlmostEqual(c, correct_c[i])

	def test_changeMassDefinition(self):
		z1 = 0.98
		M1 = [1.5E8, 1.1E15]
		c1 = 4.6
		correct_M = [118946488.07233413, 872274245863784.38]
		correct_R = [4.7970184134005516, 931.97699905443392]
		correct_c = [3.4334724175833529, 3.433472417583356]
		for i in range(len(M1)):
			M, R, c = mass_defs.changeMassDefinition(M1[i], c1, z1, 'vir', '300c')
			self.assertAlmostEqual(M, correct_M[i])
			self.assertAlmostEqual(R, correct_R[i])
			self.assertAlmostEqual(c, correct_c[i])
	
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
		correct_M = [1.298368729361e+08, 8.790756509777e+14]
		correct_R = [4.939164908490e+00, 9.343930392894e+02]
		correct_c = [9.074728024515e+00, 3.666625693650e+00]
		for i in range(len(M1)):
			M, R, c = mass_adv.changeMassDefinitionCModel(M1[i], z1, 'vir', '300c')
			self.assertAlmostEqual(M, correct_M[i])
			self.assertAlmostEqual(R, correct_R[i])
			self.assertAlmostEqual(c, correct_c[i])

		return
	
	def test_M4rs(self):
		self.assertAlmostEqual(mass_adv.M4rs(1E12, 0.7, '500c', 3.8), 1041815679897.7153)
	
###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
