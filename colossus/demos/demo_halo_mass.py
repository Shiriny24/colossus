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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.halo import mass_defs

###################################################################################################

def main():

	#compareDensityThresholds()
	#demonstrateMassDefinitions()

	return

###################################################################################################

def compareDensityThresholds():

	cosmology.setCosmology('bolshoi')
	
	N_Z = 40
	z_bins = np.linspace(0.0, 2.5, N_Z)
	rho_vir = np.array([0.0] * N_Z)
	rho_m = np.array([0.0] * N_Z)
	rho_c = np.array([0.0] * N_Z)
	
	plt.figure(figsize = (3.7, 3.7))
	plt.xlabel(r'$z$')
	plt.ylabel(r'$\rho\ (M_{\odot} h^2/{\rm kpc}^3)$')
	plt.yscale('log')
	plt.xlim(0.0, 2.5)
	plt.ylim(1E4, 6E5)
	plt.gca().xaxis.set_major_locator(MultipleLocator(1.0))
	plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))

	for i in range(N_Z):
		rho_vir[i] = mass_so.densityThreshold(z_bins[i], 'vir')
		rho_m[i] = mass_so.densityThreshold(z_bins[i], '180m')
		rho_c[i] = mass_so.densityThreshold(z_bins[i], '180c')

	plt.plot(z_bins, rho_c, '--', color = 'darkblue', label = r'$\rho_{\rm 180c}$', lw = 1.2)
	plt.plot(z_bins, rho_vir, '-', color = 'firebrick', label = r'$\rho_{\rm vir}$', lw = 1.2)
	plt.plot(z_bins, rho_m, ':', color = 'darkblue', label = r'$\rho_{\rm 180m}$', lw = 1.2)

	plt.legend(loc = 4, labelspacing = 0.1, handlelength = 1.5)
	ltext = plt.gca().get_legend().get_texts()
	plt.setp(ltext, fontsize = 14)

	plt.show()

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
