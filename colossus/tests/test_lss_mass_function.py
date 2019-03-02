###################################################################################################
#
# test_lss_mass_function.py (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
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
		
		correct = 4.431977473927e-01
		
		mf = mass_function.massFunction(M, z, q_in = 'M', mdef = 'fof', model = 'press74')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity M.')			

		mf = mass_function.massFunction(sigma, z, q_in = 'sigma', mdef = 'fof', model = 'press74')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity sigma.')			

		mf = mass_function.massFunction(nu, z, q_in = 'nu', mdef = 'fof', model = 'press74')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity nu.')			

	def test_hmfConvert(self):
		
		M = 1E13
		z = 0.2
		
		correct = 4.496396025508e-01
		mf = mass_function.massFunction(M, z, q_in = 'M', mdef = 'fof', model = 'press74', q_out = 'f')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity f.')			

		correct = 6.780453997694e-04
		mf = mass_function.massFunction(M, z, q_in = 'M', mdef = 'fof', model = 'press74', q_out = 'dndlnM')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity dndlnM.')			

		correct = 7.910742947003e-02
		mf = mass_function.massFunction(M, z, q_in = 'M', mdef = 'fof', model = 'press74', q_out = 'M2dndM')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity M2dndM.')			
				
	def test_hmfModelsFOF(self):
		models = mass_function.models
		for k in models.keys():
			msg = 'Failure in model = %s.' % (k)
			
			if not 'fof' in models[k].mdefs:
				continue
			
			if k == 'press74':
				correct = [2.236854253750e-01, 1.791450365948e-02]
			elif k == 'sheth99':
				correct = [2.037029058919e-01, 3.217106126232e-02]
			elif k == 'jenkins01':
				correct = [6.026667291380e-02, 3.438018384696e-02]
			elif k == 'reed03':
				correct = [2.037029058919e-01, 2.875081667688e-02]
			elif k == 'warren06':
				correct = [2.176074174337e-01, 3.380079804646e-02]
			elif k == 'reed07':
				correct = [1.912793816150e-01, 3.723712580773e-02]
			elif k == 'crocce10':
				correct = [2.196770391218e-01, 4.194766615518e-02]
			elif k == 'bhattacharya11':
				correct = [2.241131373076e-01, 4.065332158031e-02]
			elif k == 'courtin11':
				correct = [1.519182109604e-01, 4.488768972968e-02]
			elif k == 'angulo12':
				correct = [2.283411904950e-01, 3.769659265894e-02]
			elif k == 'watson13':
				correct = [2.847701564867e-01, 3.803667902439e-02]
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
				correct = [2.510113220688e-01, 4.610872463607e-05]
			elif k == 'watson13':
				correct = [1.621416397547e-01, 4.426907168247e-05]
			elif k == 'bocquet16':
				correct = [2.836178696344e-01, 3.831952877856e-05]
			elif k == 'despali16':
				correct = [2.566882204044e-01, 6.641439189756e-05]
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
				correct = [2.509256654054e-01, 4.539867288497e-05]
			elif k == 'watson13':
				correct = [1.613530192304e-01, 4.365917063394e-05]
			elif k == 'despali16':
				correct = [2.566106202740e-01, 6.537953819021e-05]
			elif k == 'comparat17':
				correct = [2.449556775109e-01, 2.342447382920e-05]
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
