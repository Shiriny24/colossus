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
		cosmology.setCosmology('planck15')
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
		Gamma = np.array([0.2, 1.2, 4.1])
		correct_rsp = [1.470326774861, 1.239462644843, 8.750736226357e-01]
		rsp, mask = splashback.splashbackModel('RspR200m', Gamma = Gamma, z = 0.1, model = 'more15')
		for i in range(len(Gamma)):
			self.assertEqual(mask[i], True)
			self.assertAlmostEqual(rsp[i], correct_rsp[i])
	
###################################################################################################
# TEST CASE: SPLASHBACK RADIUS
###################################################################################################

class TCSplashbackRadius(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15')
	
	def test_rspR200m(self):
		R = np.array([900.0, 1100.0])
		z = 0.1
		mdef = '200m'
		Rsp, Msp, mask = splashback.splashbackRadius(z, mdef, R = R, model = 'more15')
		correct_rsp = [1.072356135890e+03, 1.271008459317e+03]
		correct_msp = [7.894864613531e+13, 1.414956486665e+14]
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
		correct_rsp = [1.238830778939e+03, 1.465134309853e+03]
		correct_msp = [1.294892613098e+14, 2.323198518163e+14]
		for i in range(len(R)):
			self.assertEqual(mask[i], True)
			self.assertAlmostEqual(Rsp[i], correct_rsp[i])
			self.assertAlmostEqual(Msp[i], correct_msp[i])

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
