###################################################################################################
#
# test_halo_concentration.py (c) Benedikt Diemer
#     				    	     benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest
import numpy as np

from colossus.tests import test_colossus
from colossus.cosmology import cosmology
from colossus.halo import concentration

###################################################################################################
# TEST CASES
###################################################################################################

class TCConcentration(test_colossus.ColosssusTestCase):

	def setUp(self):

		pass

	###############################################################################################

	def test_model_returns(self):

		cosmology.setCosmology('bolshoi')
		M_one = 1E12
		M_one_array = np.array([1E12])
		M_many = np.array([1E10, 1E12, 1E15])
		N_array = len(M_many)
		mdefs = ['200c', 'vir', '200m', '345m']
		z = 0.0
		models = concentration.MODELS
		
		for i in range(len(models)):
			for j in range(len(mdefs)):

				c, mask = concentration.concentration(M_one, mdefs[j], z = z, model = models[i], range_return = True, range_warning = False)
				self.assertNotIsInstance(c, np.ndarray, 'Concentration should be scalar float.')
				self.assertNotIsInstance(mask, np.ndarray, 'Mask should be scalar bool.')
			
				c, mask = concentration.concentration(M_one_array, mdefs[j], z = z, model = models[i], range_return = True, range_warning = False)
				self.assertIsInstance(c, np.ndarray, 'Concentration should be an array with one element.')
				self.assertIsInstance(mask, np.ndarray, 'Mask should be an array with one element.')
				self.assertEqual(len(c), 1, 'Concentration should be an array with one element.')
				self.assertEqual(len(mask), 1, 'Mask should be an array with one element.')

				c, mask = concentration.concentration(M_many, mdefs[j], z = z, model = models[i], range_return = True, range_warning = False)
				self.assertIsInstance(c, np.ndarray, 'Concentration should be an array with multiple elements.')
				self.assertIsInstance(mask, np.ndarray, 'Mask should be an array with multiple elements.')
				self.assertEqual(len(c), N_array, 'Concentration should be an array with multiple elements.')
				self.assertEqual(len(mask), N_array, 'Mask should be an array with multiple elements.')

				c = concentration.concentration(M_one, mdefs[j], z = z, model = models[i], range_return = False, range_warning = False)
				self.assertNotIsInstance(c, np.ndarray, 'Concentration should be scalar float.')
			
				c = concentration.concentration(M_one_array, mdefs[j], z = z, model = models[i], range_return = False, range_warning = False)
				self.assertIsInstance(c, np.ndarray, 'Concentration should be an array with one element.')
				self.assertEqual(len(c), 1, 'Concentration should be an array with one element.')
				
				c = concentration.concentration(M_many, mdefs[j], z = z, model = models[i], range_return = False, range_warning = False)
				self.assertIsInstance(c, np.ndarray, 'Concentration should be an array with multiple elements.')
				self.assertEqual(len(c), N_array, 'Concentration should be an array with multiple elements.')

	###############################################################################################

	def test_model_values(self):
		cosmology.setCosmology('bolshoi')
		M = 1E12
		z = 0.5
		mdef = '257m'
		models = concentration.MODELS
		for i in range(len(models)):
			msg = 'Failure in model = %s' % (models[i])
			c = concentration.concentration(M, mdef, z = z, model = models[i], range_return = False, range_warning = False)
			if models[i] == 'diemer15':
				self.assertAlmostEqual(c, 6.6521444198414388, msg = msg)
			elif models[i] == 'klypin15_nu':
				self.assertAlmostEqual(c, 6.4588364940434486, msg = msg)
			elif models[i] == 'klypin15_m':
				self.assertAlmostEqual(c, 6.2107920072554768, msg = msg)
			elif models[i] == 'dutton14':
				self.assertAlmostEqual(c, 7.5907186889384706, msg = msg)
			elif models[i] == 'bhattacharya13':
				self.assertAlmostEqual(c, 5.8630201567079006, msg = msg)
			elif models[i] == 'prada12':
				self.assertAlmostEqual(c, 7.5547918752086218, msg = msg)
			elif models[i] == 'klypin11':
				self.assertAlmostEqual(c, 9.3289793383381223, msg = msg)
			elif models[i] == 'duffy08':
				self.assertAlmostEqual(c, 5.8441337918319354, msg = msg)
			elif models[i] == 'bullock01':
				self.assertAlmostEqual(c, 6.8400945591304065, msg = msg)
			else:
				msg = 'Unknown model, %s.' % models[i]
				raise Exception(msg)

	###############################################################################################
	
	def test_hard_fail(self):
		c = concentration.concentration(1E16, 'vir', z = 0.0, model = 'bullock01', range_return = False, range_warning = False)
		self.assertEqual(c, concentration.INVALID_CONCENTRATION)

	###############################################################################################

	# If interpolation = False, the slope is computed "manually" in the concentration routine. This 
	# function tests how different the result is from the derivative function in the Cosmology module.

	def test_PkSlopeComputation(self):
		
		M = 1E1
		z = 30.0
		cosmo = cosmology.setCosmology('bolshoi')
		k_R = concentration._diemer15_k_R(M)
		cosmo.interpolation = True
		n1 = concentration._diemer15_n(k_R)
		c1 = concentration.modelDiemer15fromM(M, z, statistic = 'median')
		cosmo.interpolation = False
		n2 = concentration._diemer15_n(k_R)
		c2 = concentration.modelDiemer15fromM(M, z, statistic = 'median')
		err1 = abs(n2 / n1 - 1.0)
		err2 = abs(c2 / c1 - 1.0)
		self.assertLess(err1, 1E-3)
		self.assertLess(err2, 1E-3)
			
###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
