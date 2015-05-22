###################################################################################################
#
# ConcentrationDemo.py 		(c) Benedikt Diemer
#							University of Chicago
#     				    	bdiemer@oddjob.uchicago.edu
#
###################################################################################################
#
# Sample code demonstrating the usage of the HaloDensityProfile.py module. 
#
###################################################################################################

import numpy
import matplotlib.pyplot as plt

import Cosmology
import Halo
import HaloDensityProfile

###################################################################################################

def main():

	demonstrateProfiles()
	#demonstrateMassDefinitions()

	return

###################################################################################################

# Compare the Diemer & Kravtsov 2014 and NFW profiles of a massive cluster halo

def demonstrateProfiles():
	
	M = 1E15
	mdef = 'vir'
	z = 0.0
	c = 5.0
	cosmo = Cosmology.setCosmology('WMAP9')
	
	r = 10**numpy.arange(-2.0, 4.5, 0.1)
	rho_m = cosmo.rho_m(z)
	
	prof_dk14 = HaloDensityProfile.DK14Profile(M = M, c = c, z = z, mdef = mdef, be = 1.0, se = 1.5)
	prof_nfw = HaloDensityProfile.NFWProfile(M = M, c = c, z = z, mdef = mdef)
	prof_ein = HaloDensityProfile.EinastoProfile(M = M, c = c, z = z, mdef = mdef)
	rho_dk14 = prof_dk14.density(r)
	rho_nfw = prof_nfw.density(r)
	rho_ein = prof_ein.density(r)
	
	plt.figure()
	plt.loglog()
	plt.xlabel(r'$r (kpc/h)$')
	plt.ylabel(r'$\rho / \rho_m$')
	plt.xlim(1E-2, 2E4)
	plt.ylim(1E-1, 1E8)
	plt.plot(r, rho_nfw / rho_m, '--', color = 'deepskyblue', label = 'NFW')
	plt.plot(r, rho_dk14 / rho_m, '-', color = 'darkblue', label = 'DK14')
	plt.plot(r, rho_ein / rho_m, '--', color = 'firebrick', label = 'Einasto')
	plt.legend()
	plt.show()
	
	return

###################################################################################################

# Convert one mass definition to another, assuming an NFW profile

def demonstrateMassDefinitions():
	
	Mvir = 1E12
	cvir = 10.0
	z = 0.0
	Cosmology.setCosmology('WMAP9')

	Rvir = Halo.M_to_R(Mvir, z, 'vir')

	print(("We start with the following halo, defined using the virial mass definition:"))	
	print(("Mvir:   %.2e Msun / h" % Mvir))
	print(("Rvir:   %.2e kpc / h" % Rvir))
	print(("cvir:   %.2f" % cvir))
	
	M200c, R200c, c200c = HaloDensityProfile.changeMassDefinition(Mvir, cvir, z, 'vir', '200c')
	
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
