###################################################################################################
#
# test_cosmology.py     (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import unittest
import numpy as np

from colossus.tests import test_colossus
from colossus import defaults
from colossus.cosmology import cosmology

###################################################################################################
# TEST PARAMETERS
###################################################################################################

TEST_Z = np.array([0.0, 1.283, 20.0])
TEST_Z2 = 5.4
TEST_K = np.array([1.2E-3, 1.1E3])
TEST_RR = np.array([1.2E-3, 1.4, 1.1E3])
TEST_AGE = np.array([13.7, 0.1])

###################################################################################################
# GENERAL CLASS FOR COSMOLOGY TEST CASES
###################################################################################################

class CosmologyTestCase(test_colossus.ColosssusTestCase):

	def _testRedshiftArray(self, f, correct):
		self.assertAlmostEqualArray(f(TEST_Z), correct)		

	def _testKArray(self, f, correct):
		self.assertAlmostEqualArray(f(TEST_K), correct)		

	def _testRZArray(self, f, z, correct):
		self.assertAlmostEqualArray(f(TEST_RR, z), correct)		

###################################################################################################
# TEST CASE 1: COMPUTATIONS WITHOUT INTERPOLATION
###################################################################################################

class TCComp(CosmologyTestCase):

	def setUp(self):
		self.cosmo_name = 'planck15'
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 'storage': ''})

	###############################################################################################
	# BASICS
	###############################################################################################
	
	def test_init(self):
		c_dict = cosmology.cosmologies[self.cosmo_name]
		self.assertEqual(self.cosmo.name, self.cosmo_name)
		self.assertAlmostEqual(self.cosmo.Om0, c_dict['Om0'])
		self.assertAlmostEqual(self.cosmo.Ob0, c_dict['Ob0'])
		self.assertAlmostEqual(self.cosmo.sigma8, c_dict['sigma8'])
		self.assertAlmostEqual(self.cosmo.ns, c_dict['ns'])
		if 'Tcmb0' in c_dict:
			self.assertAlmostEqual(self.cosmo.Tcmb0, c_dict['Tcmb0'])
		else:
			self.assertAlmostEqual(self.cosmo.Tcmb0, defaults.COSMOLOGY_TCMB0)
		if 'Neff' in c_dict:
			self.assertAlmostEqual(self.cosmo.Neff, c_dict['Neff'])
		else:
			self.assertAlmostEqual(self.cosmo.Neff, defaults.COSMOLOGY_NEFF)
		self.assertAlmostEqual(self.cosmo.Ogamma0, 5.388899947524e-05)
		self.assertAlmostEqual(self.cosmo.Onu0, 3.727873332823e-05)
		self.assertAlmostEqual(self.cosmo.Or0, 9.116773280347645e-05)
	
	def test_initNoRel(self):
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 'storage': '', 'relspecies': False})
		c_dict = cosmology.cosmologies[self.cosmo_name]
		self.assertAlmostEqual(self.cosmo.Om0, c_dict['Om0'])
		self.assertAlmostEqual(self.cosmo.OL0, 1.0 - c_dict['Om0'])
		self.assertAlmostEqual(self.cosmo.Ob0, c_dict['Ob0'])
		self.assertAlmostEqual(self.cosmo.sigma8, c_dict['sigma8'])
		self.assertAlmostEqual(self.cosmo.ns, c_dict['ns'])
		self.assertAlmostEqual(self.cosmo.Tcmb0, defaults.COSMOLOGY_TCMB0)
		self.assertAlmostEqual(self.cosmo.Neff, defaults.COSMOLOGY_NEFF)
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 'storage': ''})

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
		self.assertAlmostEqualArray(self.cosmo.comovingDistance(z_max = TEST_Z), correct)		

	def test_luminosityDistance(self):
		correct = [0.0, 6256.5906917763368, 156076.44470199241]
		self._testRedshiftArray(self.cosmo.luminosityDistance, correct)

	def test_angularDiameterDistance(self):
		correct = [0.0, 1200.399818916434, 353.91484059408708]
		self._testRedshiftArray(self.cosmo.angularDiameterDistance, correct)

	def test_distanceModulus(self):
		correct = [44.827462759550897, 51.81246085652802]
		self.assertAlmostEqualArray(self.cosmo.distanceModulus(TEST_Z[1:]), correct)		
	
	def test_soundHorizon(self):
		self.assertAlmostEqual(self.cosmo.soundHorizon(), 150.21442991795007)

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
		self.assertAlmostEqual(self.cosmo.rho_L(), 191.7444476198966)
	
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

	def test_growthFactor(self):
		correct = [1.0, 0.54093225419799251, 0.060968602011373191]
		self._testRedshiftArray(self.cosmo.growthFactor, correct)

	def test_transferFunctionEH98(self):
		correct = [0.98922569294539697, 1.4904793404415855e-08]
		self._testKArray(self.cosmo.transferFunctionEH98, correct)

	def test_transferFunctionEH98Smooth(self):
		correct = [0.98904847897413184, 1.4710454246122299e-08]
		self._testKArray(self.cosmo.transferFunctionEH98Smooth, correct)

	def test_matterPowerSpectrum(self):
		correct = [4503.7076619825366, 5.9333659722974083e-07]
		self._testKArray(self.cosmo.matterPowerSpectrum, correct)

	def test_sigma(self):
		correct = [12.071398454730772, 2.119458716015322, 0.0012803881954575054]
		self._testRZArray(self.cosmo.sigma, 0.0, correct)
		correct = [2.4016147068552129, 0.4216680645613225, 0.00025473429049883295]
		self._testRZArray(self.cosmo.sigma, TEST_Z2, correct)

	def test_correlationFunction(self):
		correct = [142.63237915313539, 3.9989807960025816, -2.7947065951546593e-07]
		self._testRZArray(self.cosmo.correlationFunction, 0.0, correct)
		correct = [5.6455937600879116, 0.15828538486611424, -1.1061848795175795e-08]
		self._testRZArray(self.cosmo.correlationFunction, TEST_Z2, correct)

###################################################################################################
# TEST CASE 2: INTERPOLATION, DERIVATIVES, INVERSES
###################################################################################################

class TCInterp(CosmologyTestCase):

	def setUp(self):
		self.cosmo_name = 'planck15'
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': True, 'storage': 'rw'})

	###############################################################################################
	# Function tests
	###############################################################################################

	def test_sigma(self):
		self.assertAlmostEqual(self.cosmo.sigma(12.5, 0.0), 5.892734771880e-01)		

	def test_ZDerivative(self):
		correct = [-14.431423683052429, -3.0331864799122887, -0.012861392030709832]
		self.assertAlmostEqualArray(self.cosmo.age(TEST_Z, derivative = 1), correct)		

	def test_ZDerivative2(self):
		correct = [20.668766775933239, 3.0310718810786343, 0.0015163247225108648]
		self.assertAlmostEqualArray(self.cosmo.age(TEST_Z, derivative = 2), correct)		

	def test_ZInverse(self):
		correct = [0.0067749101503343997, 29.812799507392906]
		self.assertAlmostEqualArray(self.cosmo.age(TEST_AGE, inverse = True), correct)		

	def test_ZInverseDerivative(self):
		correct = [-0.069866754435913142, -204.97494464862859]
		self.assertAlmostEqualArray(self.cosmo.age(TEST_AGE, inverse = True, derivative = 1), correct)		

###################################################################################################
# TEST CASE 3: NON-FLAT COSMOLOGY WITH POSITIVE CURVATURE
###################################################################################################

class TCNotFlat1(CosmologyTestCase):

	def setUp(self):
		c = {'flat': False, 'H0': 70.00, 'Om0': 0.2700, 'OL0': 0.7, 'Ob0': 0.0469, 'sigma8': 0.8200, 'ns': 0.9500, 'relspecies': True}
		cosmology.addCosmology('myCosmo', c)
		self.assertTrue('myCosmo' in cosmology.cosmologies)
		cosmology.setCosmology('myCosmo')
		self.cosmo = cosmology.getCurrent()

	def test_nonFlat(self):
		self.assertAlmostEqual(self.cosmo.Ok0, 0.02991462406767552)
		self.assertAlmostEqual(self.cosmo.Ok(4.5), 0.019417039615692584)

	def test_distanceNonFlat(self):
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 1.0, transverse = True), 2.340299035494e+03)
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 10.0, transverse = True), 6.959070874045e+03)
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 1.0, transverse = False), 2.333246162862e+03)
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 10.0, transverse = False), 6.784500417672e+03)

###################################################################################################
# TEST CASE 4: NON-FLAT COSMOLOGY WITH NEGATIVE CURVATURE
###################################################################################################

class TCNotFlat2(CosmologyTestCase):

	def setUp(self):
		c = {'flat': False, 'H0': 70.00, 'Om0': 0.2700, 'OL0': 0.8, 'Ob0': 0.0469, 'sigma8': 0.8200, 'ns': 0.9500, 'relspecies': True}
		cosmology.addCosmology('myCosmo', c)
		self.assertTrue('myCosmo' in cosmology.cosmologies)
		cosmology.setCosmology('myCosmo')
		self.cosmo = cosmology.getCurrent()

	def test_nonFlat(self):
		self.assertAlmostEqual(self.cosmo.Ok0, -7.008537593232e-02)
		self.assertAlmostEqual(self.cosmo.Ok(4.5), -4.853747713897e-02)

	def test_distanceNonFlat(self):
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 1.0, transverse = True), 2.391423796721e+03)
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 10.0, transverse = True), 6.597189702919e+03)
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 1.0, transverse = False), 2.409565059911e+03)
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 10.0, transverse = False), 7.042437425006e+03)

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
