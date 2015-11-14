###################################################################################################
#
# demo_halo_mass.py         (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################
#
# Sample code demonstrating the usage of the halo.profile module. 
#
###################################################################################################

from __future__ import division

from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.halo import mass_defs

###################################################################################################

def main():

	#demonstrateMassDefinitions()

	return

###################################################################################################

# Convert one mass definition to another, assuming an NFW profile

def demonstrateMassDefinitions():
	
	Mvir = 1E12
	cvir = 10.0
	z = 0.0
	cosmology.setCosmology('WMAP9')

	Rvir = mass_so.M_to_R(Mvir, z, 'vir')

	print(("We start with the following halo, defined using the virial mass definition:"))	
	print(("Mvir:   %.2e Msun / h" % Mvir))
	print(("Rvir:   %.2e kpc / h" % Rvir))
	print(("cvir:   %.2f" % cvir))
	
	M200c, R200c, c200c = mass_defs.changeMassDefinition(Mvir, cvir, z, 'vir', '200c')
	
	print(("Now, let's convert the halo data to the 200c mass definition, assuming an NFW profile:"))	
	print(("M200c:  %.2e Msun / h" % M200c))
	print(("R200c:  %.2e kpc / h" % R200c))
	print(("c200c:  %.2f" % c200c))
	
	return

###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
	main()
