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
		cosmology.setCosmology('planck15', {'persistence': ''})
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
		self.assertAlmostEqual(mass_so.densityThreshold(0.7, '200m'), 8.422058639458e+04)
		self.assertAlmostEqual(mass_so.densityThreshold(6.1, '400c'), 1.237331178052e+07)
		self.assertAlmostEqual(mass_so.densityThreshold(1.2, 'vir'), 1.792282349630e+05)
		with self.assertRaises(Exception):
			mass_so.densityThreshold('100t')

	def test_deltaVir(self):
		self.assertAlmostEqual(mass_so.deltaVir(0.7), 148.15504207273736)
	
	def test_M_to_R(self):
		self.assertAlmostEqual(mass_so.M_to_R(1.1E12, 0.7, '200m'), 1.460927300899e+02)
		self.assertAlmostEqual(mass_so.M_to_R(1.1E12, 0.7, 'vir'), 1.424612758650e+02)

	def test_R_to_M(self):
		self.assertAlmostEqual(mass_so.R_to_M(212.0, 0.7, '200m'), 3.361355552073e+12)
		self.assertAlmostEqual(mass_so.R_to_M(150.0, 0.7, 'vir'), 1.284032374462e+12)

###################################################################################################
# TEST CASE: DEFINITIONS
###################################################################################################

class TCMassDefs(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15', {'persistence': ''})
	
	def test_pseudoEvolve(self):
		z1 = 0.68
		z2 = 3.1
		M1 = [1.5E8, 1.1E15]
		c1 = 4.6
		correct_M = [4.458465867957e+07, 3.269541636502e+14]
		correct_R = [2.151843537712e+00, 4.180656628519e+02]
		correct_c = [1.300870562115e+00, 1.300870562115e+00]
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
		correct_M = [4.111590726837e+07, 4.333401137851e+14]
		correct_R = [2.094524770335e+00, 4.592246295680e+02]
		correct_c = [1.266219205810e+00, 1.428942520483e+00]
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
		correct_M = [1.189464767031e+08, 8.722741624894e+14]
		correct_R = [4.797075718384e+00, 9.319881324131e+02]
		correct_c = [3.433472308190e+00, 3.433472308190e+00]
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
		cosmology.setCosmology('planck15', {'persistence': ''})
		pass

	def test_changeMassDefinitionCModel(self):
		z1 = 0.98
		M1 = [1.5E8, 1.1E15]
		correct_M = [1.299198986900e+08, 8.814339499036e+14]
		correct_R = [4.940276660047e+00, 9.352390615503e+02]
		correct_c = [9.164742104731e+00, 3.753048751828e+00]
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
