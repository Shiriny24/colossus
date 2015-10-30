###################################################################################################
#
# demo_cosmology.py         (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################
#
# Sample code demonstrating the usage of the cosmology.py module. Uncomment the functions listed
# in main() to explore various aspects of the cosmology module.
#
###################################################################################################

import numpy as np

from colossus.utils import utilities
from colossus.cosmology import cosmology

###################################################################################################

def main():
	
	# ---------------------------------------------------------------------------------------------
	# Basic operations: setting, getting and changing the cosmology
	# ---------------------------------------------------------------------------------------------

	demonstrateSettingAndGetting()
	#demonstrateAdding()
	#demonstrateChanging()
	
	# ---------------------------------------------------------------------------------------------
	# Basic computations; nothing will be persistently stored between runs.
	# ---------------------------------------------------------------------------------------------

	#compute()

	# ---------------------------------------------------------------------------------------------
	# Advanced computations; these can take a second the first time they are executed, but the 
	# results are stored in lookup tables. The second execution should be lightning fast.
	# ---------------------------------------------------------------------------------------------

	#computeAdvanced()

	return

###################################################################################################

def printCosmologyName():
	
	cosmo = cosmology.getCurrent()
	print((cosmo.name))
	
	return

###################################################################################################

def demonstrateSettingAndGetting():
	
	print("Let's set a cosmology and print it's name.")
	cosmo = cosmology.setCosmology('WMAP9')
	print((cosmo.name))
	utilities.printLine()
	print("Now we do the same but in a function, using the global cosmology variable.")
	printCosmologyName()
	utilities.printLine()
	print("Now let's temporarily switch cosmology without destroying the cosmology objects.")
	old_cosmo = cosmo
	cosmo = cosmology.setCosmology('planck13')
	printCosmologyName()
	cosmology.setCurrent(old_cosmo)
	print((cosmology.getCurrent().name))
	print("We can also change individual parameters when setting a cosmology.")
	cosmo = cosmology.setCosmology('WMAP7', {"interpolation": False})
	print((cosmo.name))
	print((cosmo.interpolation))
	print("We should be careful changing cosmological parameters this way, since the name does not match the cosmology any more.")

	return

###################################################################################################

def setMyCosmo():
	
	cosmo = cosmology.setCosmology('my_cosmo')
	
	return cosmo

###################################################################################################

def demonstrateAdding():

	my_cosmo = {'flat': True, 'H0': 72.0, 'Om0': 0.25, 'Ob0': 0.043, 'sigma8': 0.8, 'ns': 0.97}
	
	print("Let's set a non-standard cosmology")
	cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
	print(("We are now in " + cosmo.name + ", H0 = %.1f" % (cosmo.H0)))
	utilities.printLine()
	print("We can also add this cosmology to the library of cosmologies, which allos us to set it from any function.")
	cosmology.addCosmology('my_cosmo', my_cosmo)
	cosmo = setMyCosmo()
	print(("We are once again in " + cosmo.name + ", H0 = %.1f" % (cosmo.H0)))
	
	return

###################################################################################################

def demonstrateChanging():
	
	cosmo = cosmology.setCosmology('planck13')
	print(("We are in the " + cosmo.name + " cosmology"))
	print(("Omega_m = %.2f, Omega_L = %.2f" % (cosmo.Om0, cosmo.OL0)))
	utilities.printLine()
	print("Let's do something bad and change a parameter without telling the cosmology class...")
	cosmo.Om0 = 0.27
	print("Now the universe is not flat any more:")
	print(("Omega_m = %.2f, Omega_L = %.2f" % (cosmo.Om0, cosmo.OL0)))
	utilities.printLine()
	print("Now let's do it correctly and call checkForChangedcosmology():")
	cosmo.checkForChangedcosmology()
	print(("Omega_m = %.2f, Omega_L = %.2f" % (cosmo.Om0, cosmo.OL0)))
	print("It is OK to change parameters at any time, but you MUST call checkForChangedcosmology() immediately afterwards!")
	
	return

###################################################################################################

def compute():
	
	cosmo = cosmology.setCosmology('WMAP9')
	z = np.array([0.0, 1.0, 10.0])
	
	print("All cosmology functions can be called with numbers or numpy arrays:")
	print(("z                 = " + str(z)))
	utilities.printLine()
	print("Times are output in Gyr, for example:")
	print(("Age               = " + str(cosmo.age(z))))
	utilities.printLine()
	print("Distances are output in Mpc/h, for example:")
	print(("Comoving distance = " + str(cosmo.comovingDistance(z_max = z))))
	utilities.printLine()
	print("Densities are output in astronomical units, Msun h^2 / kpc^3, for example:")
	print(("Critical density  = " + str(cosmo.rho_c(z))))

	return

###################################################################################################

def computeAdvanced():

	cosmo = cosmology.setCosmology('WMAP9')
	z = 0.0
	M = np.array([1E9, 1E12, 1E15])
	
	print("We are now executing a function that needs sigma(R).")
	utilities.printLine()
	nu = cosmo.peakHeight(M, z)
	print(("Peak height = " + str(nu)))
	utilities.printLine()
	print("Now, a lookup table for sigma(R) should be stored in a binary file in the Data/ directory.")
	print("If you call this function again, it should execute faster!")
	utilities.printLine()
	print("If we want to turn this behavior off, set 'storage = False'.")
	print("If we do not want to use any lookup tables at all, set 'interpolation = False'.")
	print("This setting is almost never recommended, though.")
		
	return

###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
	main()
