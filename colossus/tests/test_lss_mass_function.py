###################################################################################################
#
# test_lss_mass_function.py (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

import unittest
import numpy as np

from colossus.tests import test_colossus
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from colossus.lss import peaks

###################################################################################################
# TEST CASES
###################################################################################################

class TCMassFunction(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15', {'persistence': ''})
		pass
		
	def test_hmfInput(self):
		
		M = 1E12
		z = 1.0
		nu = peaks.peakHeight(M, z)
		delta_c = peaks.collapseOverdensity()
		sigma = delta_c / nu
		
		correct = 4.432081012627e-01
		
		mf = mass_function.massFunction(M, z, q_in = 'M', mdef = 'fof', model = 'press74')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity M.')			

		mf = mass_function.massFunction(sigma, z, q_in = 'sigma', mdef = 'fof', model = 'press74')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity sigma.')			

		mf = mass_function.massFunction(nu, z, q_in = 'nu', mdef = 'fof', model = 'press74')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity nu.')			

	def test_hmfConvert(self):
		
		M = 1E13
		z = 0.2
		
		correct = 4.496509540103e-01
		mf = mass_function.massFunction(M, z, q_in = 'M', mdef = 'fof', model = 'press74', q_out = 'f')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity f.')			

		correct = 6.782011823365e-04
		mf = mass_function.massFunction(M, z, q_in = 'M', mdef = 'fof', model = 'press74', q_out = 'dndlnM')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity dndlnM.')			

		correct = 7.910798600386e-02
		mf = mass_function.massFunction(M, z, q_in = 'M', mdef = 'fof', model = 'press74', q_out = 'M2dndM')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity M2dndM.')			
				
	def test_hmfModelsFOF(self):
		models = mass_function.models
		for k in models.keys():
			msg = 'Failure in model = %s.' % (k)
			
			if not 'fof' in models[k].mdefs:
				continue
			
			if k == 'press74':
				correct = [2.236817414379e-01, 1.792404500225e-02]
			elif k == 'sheth99':
				correct = [2.037009972300e-01, 3.218302373538e-02]
			elif k == 'jenkins01':
				correct = [6.026069725012e-02, 3.439425663994e-02]
			elif k == 'reed03':
				correct = [2.037009972300e-01, 2.876252283306e-02]
			elif k == 'warren06':
				correct = [2.176065144322e-01, 3.381465783767e-02]
			elif k == 'reed07':
				correct = [1.912774404547e-01, 3.725141648998e-02]
			elif k == 'crocce10':
				correct = [2.196760269744e-01, 4.196271782970e-02]
			elif k == 'bhattacharya11':
				correct = [2.241120148148e-01, 4.066855813171e-02]
			elif k == 'courtin11':
				correct = [1.519159471219e-01, 4.490343243803e-02]
			elif k == 'angulo12':
				correct = [2.283404301823e-01, 3.771150749193e-02]
			elif k == 'watson13':
				correct = [2.847700292451e-01, 3.805146849248e-02]
			else:
				msg = 'Unknown model, %s.' % k
				raise Exception(msg)
			
			self.assertAlmostEqualArray(mass_function.massFunction(np.array([1E8, 1E15]), 0.0, 
								q_in = 'M', mdef = 'fof', model = k), correct, msg = msg)

	def test_hmfModelsSO_200m(self):
		models = mass_function.models
		for k in models.keys():
			msg = 'Failure in model = %s.' % (k)
			
			mdef = '200m'
			z = 1.0
			
			if not (('*' in models[k].mdefs) or (mdef in models[k].mdefs)):
				continue
			
			if k == 'tinker08':
				correct = [2.510097130127e-01, 4.616673996075e-05]
			elif k == 'watson13':
				correct = [1.621407762267e-01, 4.432897408699e-05]
			elif k == 'bocquet16':
				correct = [2.836176934812e-01, 3.836934411575e-05]
			elif k == 'despali16':
				correct = [2.566857998226e-01, 6.649213465912e-05]
			else:
				msg = 'Unknown model, %s.' % k
				raise Exception(msg)
			
			self.assertAlmostEqualArray(mass_function.massFunction(np.array([1E8, 1E15]), z, 
								q_in = 'M', mdef = mdef, model = k), correct, msg = msg)

	def test_hmfModelsSO_vir(self):
		models = mass_function.models
		for k in models.keys():
			msg = 'Failure in model = %s.' % (k)
			
			mdef = 'vir'
			z = 1.0
			
			if not (('*' in models[k].mdefs) or (mdef in models[k].mdefs)):
				continue
			
			if k == 'tinker08':
				correct = [2.509240630699e-01, 4.545587447828e-05]
			elif k == 'watson13':
				correct = [1.613521597080e-01, 4.371829786360e-05]
			elif k == 'despali16':
				correct = [2.566082087345e-01, 6.545617453508e-05]
			elif k == 'comparat17':
				correct = [2.449535870384e-01, 2.345606191508e-05]
			else:
				msg = 'Unknown model, %s.' % k
				raise Exception(msg)
			
			self.assertAlmostEqualArray(mass_function.massFunction(np.array([1E8, 1E15]), z, 
								q_in = 'M', mdef = mdef, model = k), correct, msg = msg)

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
