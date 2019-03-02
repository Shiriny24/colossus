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
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 
															'persistence': ''})

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
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 
												'persistence': '', 'relspecies': False})
		c_dict = cosmology.cosmologies[self.cosmo_name]
		self.assertAlmostEqual(self.cosmo.Om0, c_dict['Om0'])
		self.assertAlmostEqual(self.cosmo.Ode0, 1.0 - c_dict['Om0'])
		self.assertAlmostEqual(self.cosmo.Ob0, c_dict['Ob0'])
		self.assertAlmostEqual(self.cosmo.sigma8, c_dict['sigma8'])
		self.assertAlmostEqual(self.cosmo.ns, c_dict['ns'])
		self.assertAlmostEqual(self.cosmo.Tcmb0, defaults.COSMOLOGY_TCMB0)
		self.assertAlmostEqual(self.cosmo.Neff, defaults.COSMOLOGY_NEFF)
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 
															'persistence': ''})

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
		self.assertAlmostEqual(self.cosmo.soundHorizon(), 1.017552548264e+02)

	###############################################################################################
	# Densities and overdensities
	###############################################################################################

	def test_rho_c(self):
		correct = [2.774748292500e+02, 1.212328626364e+03, 7.988900732161e+05]
		self._testRedshiftArray(self.cosmo.rho_c, correct)
	
	def test_rho_m(self):
		correct = [8.571197475533e+01, 1.019903860325e+03, 7.937785982091e+05]
		self._testRedshiftArray(self.cosmo.rho_m, correct)
	
	def test_rho_de(self):
		self.assertAlmostEqual(self.cosmo.rho_de(0.0), 1.917375577436e+02)
	
	def test_rho_gamma(self):
		correct = [1.495284092785e-02, 4.062069588531e-01, 2.908043456489e+03]
		self._testRedshiftArray(self.cosmo.rho_gamma, correct)
	
	def test_rho_nu(self):
		correct = [1.034391016491e-02, 2.810013368705e-01, 2.011693992781e+03]
		self._testRedshiftArray(self.cosmo.rho_nu, correct)
	
	def test_rho_r(self):
		correct = [2.529675109275e-02, 6.872082957237e-01, 4.919737449270e+03]
		self._testRedshiftArray(self.cosmo.rho_r, correct)
	
	def test_Om(self):
		correct = [0.3089, 0.84127672822803978, 0.99360177929557136]
		self._testRedshiftArray(self.cosmo.Om, correct)
	
	def test_Ob(self):
		correct = [4.860000000000e-02, 1.323601456519e-01, 1.563258221876e-01]
		self._testRedshiftArray(self.cosmo.Ob, correct)
	
	def test_Ode(self):
		correct = [0.69100883226719656, 0.15815642192549204, 0.00024000493205743255]
		self._testRedshiftArray(self.cosmo.Ode, correct)
	
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

	def test_matterPowerSpectrum(self):
		correct = [4.503657747484e+03, 5.933300212925e-07]
		self._testKArray(self.cosmo.matterPowerSpectrum, correct)

	def test_matterPowerSpectrumZ(self):
		correct = [1.824043465475e+03, 2.403068369957e-07]
		Pk = self.cosmo.matterPowerSpectrum(TEST_K, z = 0.9)
		self.assertAlmostEqualArray(Pk, correct)

	def test_sigma(self):
		correct = [1.207145625229e+01, 2.119444226232e+00, 1.280494909616e-03]
		self._testRZArray(self.cosmo.sigma, 0.0, correct)
		correct = [2.401626205727e+00, 4.216651818070e-01, 2.547555213689e-04]
		self._testRZArray(self.cosmo.sigma, TEST_Z2, correct)

	def test_correlationFunction(self):
		correct = [1.426307983614e+02, 3.998936475381e+00, -2.794675621480e-07]
		self._testRZArray(self.cosmo.correlationFunction, 0.0, correct)
		correct = [5.645531190089e+00, 1.582836305925e-01, -1.106172619694e-08]
		self._testRZArray(self.cosmo.correlationFunction, TEST_Z2, correct)

###################################################################################################
# TEST CASE 2: INTERPOLATION, DERIVATIVES, INVERSES
###################################################################################################

class TCInterp(CosmologyTestCase):

	def setUp(self):
		self.cosmo_name = 'planck15'
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': True, 
															'persistence': ''})

	###############################################################################################
	# Function tests
	###############################################################################################

	def test_sigma(self):
		self.assertAlmostEqual(self.cosmo.sigma(12.5, 0.0), 5.892448500561e-01)

	def test_ZDerivative(self):
		correct = [-1.443561688659e+01, -3.025536156424e+00, -1.281102065439e-02]
		self.assertAlmostEqualArray(self.cosmo.age(TEST_Z, derivative = 1), correct)		

	def test_ZDerivative2(self):
		correct = [2.112577677848e+01, 2.994339523099e+00, 1.532142754901e-03]
		self.assertAlmostEqualArray(self.cosmo.age(TEST_Z, derivative = 2), correct)		

	def test_ZInverse(self):
		correct = [6.738530459850e-03, 2.981581730191e+01]
		self.assertAlmostEqualArray(self.cosmo.age(TEST_AGE, inverse = True), correct)		

	def test_ZInverseDerivative(self):
		correct = [-6.998050778719e-02, -2.036515876830e+02]
		self.assertAlmostEqualArray(self.cosmo.age(TEST_AGE, inverse = True, derivative = 1), correct)		

###################################################################################################
# TEST CASE 3: NON-FLAT COSMOLOGY WITH POSITIVE CURVATURE
###################################################################################################

class TCNotFlat1(CosmologyTestCase):

	def setUp(self):
		c = {'flat': False, 'H0': 70.00, 'Om0': 0.2700, 'Ode0': 0.7, 'Ob0': 0.0469, 'sigma8': 0.8200, 
				'ns': 0.9500, 'relspecies': True, 'persistence': ''}
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
		c = {'flat': False, 'H0': 70.00, 'Om0': 0.2700, 'Ode0': 0.8, 'Ob0': 0.0469, 'sigma8': 0.8200, 
			'ns': 0.9500, 'relspecies': True, 'persistence': ''}
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
# TEST CASE 5: VARYING DARK ENERGY EQUATION OF STATE 1
###################################################################################################

class TCDarkEnergy1(CosmologyTestCase):

	def setUp(self):
		c = {'flat': True, 'H0': 70.00, 'Om0': 0.2700, 'Ob0': 0.0469, 'sigma8': 0.8200, 
			'ns': 0.9500, 'relspecies': True, 'de_model': 'w0wa', 'w0': -0.7, 'wa': 0.2, 'persistence': ''}
		cosmology.addCosmology('myCosmo', c)
		self.assertTrue('myCosmo' in cosmology.cosmologies)
		cosmology.setCosmology('myCosmo')
		self.cosmo = cosmology.getCurrent()

	def test_Ez(self):
		self.assertAlmostEqual(self.cosmo.wz(0.5), -6.333333333333e-01)
		self.assertAlmostEqual(self.cosmo.Ez(1.2), 2.143355324420e+00)

###################################################################################################
# TEST CASE 6: VARYING DARK ENERGY EQUATION OF STATE 2
###################################################################################################

class TCDarkEnergy2(CosmologyTestCase):

	def setUp(self):
		c = {'flat': True, 'H0': 70.00, 'Om0': 0.2700, 'Ob0': 0.0469, 'sigma8': 0.8200, 
			'ns': 0.9500, 'relspecies': True, 'de_model': 'w0', 'w0': -0.7, 'persistence': ''}
		cosmology.addCosmology('myCosmo', c)
		self.assertTrue('myCosmo' in cosmology.cosmologies)
		cosmology.setCosmology('myCosmo')
		self.cosmo = cosmology.getCurrent()

	def test_Ez(self):
		self.assertAlmostEqual(self.cosmo.wz(0.5), -0.7)
		self.assertAlmostEqual(self.cosmo.Ez(1.2), 2.088306361926e+00)

###################################################################################################
# TEST CASE 7: VARYING DARK ENERGY EQUATION OF STATE 2
###################################################################################################

# Dark energy equation of state test function
def wz_func(z):
	return -0.7 + 0.2 * (1.0 - 1.0 / (1.0 + z))

class TCDarkEnergy3(CosmologyTestCase):

	def setUp(self):
		c = {'flat': True, 'H0': 70.00, 'Om0': 0.2700, 'Ob0': 0.0469, 'sigma8': 0.8200, 
			'ns': 0.9500, 'relspecies': True, 'de_model': 'user', 'wz_function': wz_func,
			'persistence': ''}
		cosmology.addCosmology('myCosmo', c)
		self.assertTrue('myCosmo' in cosmology.cosmologies)
		cosmology.setCosmology('myCosmo')
		self.cosmo = cosmology.getCurrent()

	def test_Ez(self):
		self.assertAlmostEqual(self.cosmo.wz(0.5), -6.333333333333e-01)
		self.assertAlmostEqual(self.cosmo.Ez(1.2), 2.143355324420e+00)

###################################################################################################
# TEST CASE 8: GROWTH FACTOR IN w0CDM
###################################################################################################

class TCDarkEnergyGrowthFactor(CosmologyTestCase):

	def setUp(self):
		pass
	
	def test_growthFactorFromODE(self):
		z = np.array([1.0, 0.5, 2.0, -0.9, 3.0, 120.0, 0.0])
		for k in range(2):
			interpolation = (k == 1)
			my_cosmo_1 = {'flat': True, 'H0': 100 * 0.693, 'Om0': 0.287, 'Ob0': 0.043, 'sigma8': 0.820, 'ns': 1, 
						'persistence': '', 'interpolation': interpolation}
			my_cosmo_2 = {'flat': True, 'H0': 100 * 0.693, 'Om0': 0.287, 'Ob0': 0.043, 'sigma8': 0.820, 'ns': 1, 
						'persistence': '', 'interpolation': interpolation, "de_model": "w0", "w0": -1.0}
			cosmo1 = cosmology.setCosmology('test_1', my_cosmo_1)
			D1 = cosmo1.growthFactor(z)
			cosmo2 = cosmology.setCosmology('test_2', my_cosmo_2)
			D2 = cosmo2.growthFactor(z)
			self.assertAlmostEqualArray(D1, D2, places = 4)

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
