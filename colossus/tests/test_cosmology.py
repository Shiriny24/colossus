###################################################################################################
#
# test_cosmology.p      (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest
import numpy

from colossus.tests import test_colossus
from colossus.utils import defaults
from colossus.cosmology import cosmology

###################################################################################################
# TEST PARAMETERS
###################################################################################################

TEST_N_DIGITS = 10
TEST_Z = numpy.array([0.0, 1.283, 20.0])
TEST_M = 3E12
TEST_R = 1.245

###################################################################################################
# UNIT TEST CLASS
###################################################################################################

class CosmologyTestCase(test_colossus.ColosssusTestCase):

	def setUp(self):
		self.cosmo_name = 'planck15'
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 'storage': False})

	def _testRedshiftArray(self, f, correct):
		self.assertAlmostEqualArray(f(TEST_Z), correct, places = TEST_N_DIGITS)		

	###############################################################################################
	# BASICS
	###############################################################################################
	
	def test_init(self):
		c_dict = cosmology.cosmologies[self.cosmo_name]
		self.assertEqual(self.cosmo.name, self.cosmo_name)
		self.assertAlmostEqual(self.cosmo.Om0, c_dict['Om0'], places = TEST_N_DIGITS)
		self.assertAlmostEqual(self.cosmo.Ob0, c_dict['Ob0'], places = TEST_N_DIGITS)
		self.assertAlmostEqual(self.cosmo.sigma8, c_dict['sigma8'], places = TEST_N_DIGITS)
		self.assertAlmostEqual(self.cosmo.ns, c_dict['ns'], places = TEST_N_DIGITS)
		if 'Tcmb0' in c_dict:
			self.assertAlmostEqual(self.cosmo.Tcmb0, c_dict['Tcmb0'], places = TEST_N_DIGITS)
		else:
			self.assertAlmostEqual(self.cosmo.Tcmb0, defaults.COSMOLOGY_TCMB0, places = TEST_N_DIGITS)
		if 'Neff' in c_dict:
			self.assertAlmostEqual(self.cosmo.Neff, c_dict['Neff'], places = TEST_N_DIGITS)
		else:
			self.assertAlmostEqual(self.cosmo.Neff, defaults.COSMOLOGY_NEFF, places = TEST_N_DIGITS)
		self.assertAlmostEqual(self.cosmo.Ogamma0, 5.3888999e-05, places = TEST_N_DIGITS)
		self.assertAlmostEqual(self.cosmo.Onu0, 3.7278733e-05, places = TEST_N_DIGITS)
		self.assertAlmostEqual(self.cosmo.Or0, 9.1167732e-05, places = TEST_N_DIGITS)
	
	def test_initNoRel(self):
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 'storage': False, 'relspecies': False})
		c_dict = cosmology.cosmologies[self.cosmo_name]
		self.assertAlmostEqual(self.cosmo.Om0, c_dict['Om0'], places = TEST_N_DIGITS)
		self.assertAlmostEqual(self.cosmo.OL0, 1.0 - c_dict['Om0'], places = TEST_N_DIGITS)
		self.assertAlmostEqual(self.cosmo.Ob0, c_dict['Ob0'], places = TEST_N_DIGITS)
		self.assertAlmostEqual(self.cosmo.sigma8, c_dict['sigma8'], places = TEST_N_DIGITS)
		self.assertAlmostEqual(self.cosmo.ns, c_dict['ns'], places = TEST_N_DIGITS)
		self.assertAlmostEqual(self.cosmo.Tcmb0, defaults.COSMOLOGY_TCMB0, places = TEST_N_DIGITS)
		self.assertAlmostEqual(self.cosmo.Neff, defaults.COSMOLOGY_NEFF, places = TEST_N_DIGITS)
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 'storage': False})

	# TODO
	#def test_initNonFlat(self):
		

	###############################################################################################
	# Basic cosmology calculations
	###############################################################################################
	
	def test_Ez(self):
		correct = [1.0, 2.090250729474342, 53.657658359973375]
		self._testRedshiftArray(self.cosmo.Ez, correct)
		
	def test_Hz(self):
		correct = [67.74, 141.5935844145919, 3634.7697773045961]
		self._testRedshiftArray(self.cosmo.Hz, correct)

	###############################################################################################
	# Times & distances
	###############################################################################################

	def test_hubbleTime(self):
		correct = [14.434808845686167, 6.9057786427928889, 0.26901674964731592]
		self._testRedshiftArray(self.cosmo.hubbleTime, correct)
	
	def test_lookbackTime(self):
		correct = [0.0, 8.9280198746525148, 13.619006640208726]
		self._testRedshiftArray(self.cosmo.lookbackTime, correct)

	def test_age(self):
		correct = [13.797415621282903, 4.8693957466303877, 0.17840898107417968]
		self._testRedshiftArray(self.cosmo.age, correct)
	
	def test_comovingDistance(self):
		correct = [0.0, 2740.5127865862187, 7432.2116524758285]
		self.assertAlmostEqualArray(self.cosmo.comovingDistance(z_max = TEST_Z), correct, places = TEST_N_DIGITS)		

	def test_luminosityDistance(self):
		correct = [0.0, 6256.5906917763368, 156076.44470199241]
		self._testRedshiftArray(self.cosmo.luminosityDistance, correct)

	def test_angularDiameterDistance(self):
		correct = [0.0, 1200.399818916434, 353.91484059408708]
		self._testRedshiftArray(self.cosmo.angularDiameterDistance, correct)

	def test_distanceModulus(self):
		correct = [44.827462759550897, 51.81246085652802]
		self.assertAlmostEqualArray(self.cosmo.distanceModulus(TEST_Z[1:]), correct, places = TEST_N_DIGITS)		
	
	def test_soundHorizon(self):
		self.assertAlmostEqual(self.cosmo.soundHorizon(), 150.21442991795007, places = TEST_N_DIGITS)

	###############################################################################################
	# Densities and overdensities
	###############################################################################################

	def test_rho_c(self):
		correct = [277.48480000000001, 1212.3721900475718, 798918.78044411447]
		self._testRedshiftArray(self.cosmo.rho_c, correct)
	
	def test_rho_m(self):
		correct = [85.715054719999998, 1019.9405094378844, 793807.12176191993]
		self._testRedshiftArray(self.cosmo.rho_m, correct)
	
	def test_rho_L(self):
		self.assertAlmostEqual(self.cosmo.rho_L(), 191.7444476198966, places = TEST_N_DIGITS)
	
	def test_rho_gamma(self):
		correct = [0.014953378241588128, 0.40622155544932192, 2908.1479538023009]
		self._testRedshiftArray(self.cosmo.rho_gamma, correct)
	
	def test_rho_nu(self):
		correct = [0.010344281861837975, 0.28101143434165871, 2011.766280772111]
		self._testRedshiftArray(self.cosmo.rho_nu, correct)
	
	def test_rho_r(self):
		correct = [0.025297660103426101, 0.68723298979098058, 4919.9142345744112]
		self._testRedshiftArray(self.cosmo.rho_r, correct)
	
	def test_Om(self):
		correct = [0.3089, 0.84127672822803978, 0.99360177929557136]
		self._testRedshiftArray(self.cosmo.Om, correct)
	
	def test_OL(self):
		correct = [0.69100883226719656, 0.15815642192549204, 0.00024000493205743255]
		self._testRedshiftArray(self.cosmo.OL, correct)
	
	def test_Ok(self):
		correct = [0.0, 0.0, 0.0]
		self._testRedshiftArray(self.cosmo.Ok, correct)
	
	def test_Ogamma(self):
		correct = [5.3888999475243789e-05, 0.00033506340609263101, 0.003640104632645733]
		self._testRedshiftArray(self.cosmo.Ogamma, correct)
	
	def test_Onu(self):
		correct = [3.7278733328232663e-05, 0.0002317864403757333, 0.0025181111397253441]
		self._testRedshiftArray(self.cosmo.Onu, correct)
	
	def test_Or(self):
		correct = [9.1167732803476445e-05, 0.00056684984646836417, 0.0061582157723710767]
		self._testRedshiftArray(self.cosmo.Or, correct)

	###############################################################################################
	# Structure growth, power spectrum etc.
	###############################################################################################

	def test_lagrangianR(self):
		self.assertAlmostEqual(self.cosmo.lagrangianR(TEST_M), 2.0292015228231484, places = TEST_N_DIGITS)

	def test_lagrangianM(self):
		self.assertAlmostEqual(self.cosmo.lagrangianM(TEST_R), 692873211113.4847, places = TEST_N_DIGITS)

	###############################################################################################
	# Interpolation, derivatives, and inverses
	###############################################################################################

	# This test is general for all derivative functions, since they are based on the same routine
	#def testZDerivative(self):
	#	correct = [0.0, 6257.341863900032, 156183.01636351796]
	#	self.assertAlmostEqualArray(self.cosmo.age(TEST_Z, derivative = 1), correct, places = TEST_N_DIGITS)		



if __name__ == '__main__':
	unittest.main()
