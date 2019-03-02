###################################################################################################
#
# test_halo_splashback.py  (c) Benedikt Diemer
#     				    	   benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest
import numpy as np

from colossus.tests import test_colossus
from colossus.cosmology import cosmology
from colossus.halo import splashback

###################################################################################################
# TEST CASE: SPLASHBACK MODELS
###################################################################################################

class TCSplashbackModel(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15', {'persistence': ''})
		pass

	def test_modelGamma(self):
		rsp, mask = splashback.splashbackModel('RspR200m', Gamma = 1.2, z = 0.1, model = 'more15')
		self.assertEqual(mask, True)
		self.assertAlmostEqual(rsp, 1.239462644843)
	
	def test_modelNu(self):
		rsp, mask = splashback.splashbackModel('RspR200m', nu200m = 0.6, z = 0.1, model = 'more15')
		self.assertEqual(mask, True)
		self.assertAlmostEqual(rsp, 1.424416723584)

	def test_modelGammaArray(self):
		Gamma = np.array([0.5, 4.1])
		z = 0.1
		mdef = '200m'
		R200m_test = 900.0
		R200m = np.ones_like(Gamma)* R200m_test
		models = splashback.models
		for k in models.keys():
			msg = 'Failure in model = %s' % (k)
			Rsp, _, mask = splashback.splashbackRadius(z, mdef, Gamma = Gamma, R = R200m, model = k)
			RspR200m = Rsp / R200m_test
			
			if k == 'adhikari14':
				correct_rsp = [1.269417774113e+00, 8.167315805978e-01]
			elif k == 'more15':
				correct_rsp = [1.392934317089e+00, 8.750736226357e-01]
			elif k == 'shi16':
				correct_rsp = [1.334955458442e+00, 6.672045990854e-01]
			elif k == 'mansfield17':
				correct_rsp = [1.386075126745e+00, 1.138968512092e+00]
			elif k == 'diemer17':
				correct_rsp = [1.232502327747e+00, 7.998382581962e-01]
			
			for i in range(len(Gamma)):
				self.assertEqual(mask[i], True, msg = msg)
				self.assertAlmostEqual(RspR200m[i], correct_rsp[i], msg = msg)
	
###################################################################################################
# TEST CASE: SPLASHBACK RADIUS
###################################################################################################

class TCSplashbackRadius(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15', {'persistence': ''})
	
	def test_rspR200m(self):
		R = np.array([900.0, 1100.0])
		z = 0.1
		mdef = '200m'
		Rsp, Msp, mask = splashback.splashbackRadius(z, mdef, R = R, model = 'more15')
		correct_rsp = [1.072344532566e+03, 1.270993853254e+03]
		correct_msp = [7.894529830841e+13, 1.414895737699e+14]
		for i in range(len(R)):
			self.assertEqual(mask[i], True)
			self.assertAlmostEqual(Rsp[i], correct_rsp[i])
			self.assertAlmostEqual(Msp[i], correct_msp[i])

	def test_rspRvir(self):
		R = np.array([900.0, 1100.0])
		z = 0.1
		mdef = 'vir'
		Rsp, Msp, mask = splashback.splashbackRadius(z, mdef, R = R, 
									model = 'more15', c_model = 'diemer15')
		correct_rsp = [1.238620949024e+03, 1.464941206737e+03]
		correct_msp = [1.294130812678e+14, 2.322119675566e+14]
		for i in range(len(R)):
			self.assertEqual(mask[i], True)
			self.assertAlmostEqual(Rsp[i], correct_rsp[i])
			self.assertAlmostEqual(Msp[i], correct_msp[i])

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
