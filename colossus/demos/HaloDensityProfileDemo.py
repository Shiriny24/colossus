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
import matplotlib.gridspec as gridspec

import Cosmology
import Halo
import HaloDensityProfile

###################################################################################################

def main():

	demonstrateProfiles()
	#demonstrateMassDefinitions()

	return

###################################################################################################

# Compare the Diemer & Kravtsov 2014, NFW, and Einasto profiles of a massive cluster halo

def demonstrateProfiles():
	
	M = 1E15
	mdef = 'vir'
	z = 0.0
	c = 5.0
	cosmo = Cosmology.setCosmology('WMAP9')
	R = Halo.M_to_R(M, z, mdef)
	
	rR_min = 1E-3
	rR_max = 1E1
	rR = 10**numpy.arange(numpy.log(rR_min), numpy.log(rR_max), 0.02)
	r = rR * R
	rho_m = cosmo.rho_m(z)
	
	# Compute profile density and slope
	prof_dk14 = HaloDensityProfile.DK14Profile(M = M, c = c, z = z, mdef = mdef, be = 1.0, se = 1.5)
	prof_nfw = HaloDensityProfile.NFWProfile(M = M, c = c, z = z, mdef = mdef)
	prof_ein = HaloDensityProfile.EinastoProfile(M = M, c = c, z = z, mdef = mdef)
	
	rho_dk14 = prof_dk14.density(r)
	rho_nfw = prof_nfw.density(r)
	rho_ein = prof_ein.density(r)
	
	slope_dk14 = prof_dk14.densityDerivativeLog(r)
	slope_nfw = prof_nfw.densityDerivativeLog(r)
	slope_ein = prof_ein.densityDerivativeLog(r)
	
	# Plot
	fig = plt.figure(figsize = (5.5, 10.0))
	gs = gridspec.GridSpec(2, 1)
	plt.subplots_adjust(left = 0.2, right = 0.95, top = 0.95, bottom = 0.1, hspace = 0.1)
	p1 = fig.add_subplot(gs[0])	
	p2 = fig.add_subplot(gs[1])

	plt.sca(p1)		
	plt.loglog()
	plt.ylabel(r'$\rho / \rho_m$')
	p1.set_xticklabels([])	
	plt.xlim(rR_min, rR_max)
	plt.ylim(1E-1, 5E6)
	
	plt.plot(rR, rho_nfw / rho_m, '--', color = 'deepskyblue', label = 'NFW')
	plt.plot(rR, rho_dk14 / rho_m, '-', color = 'darkblue', label = 'DK14')
	plt.plot(rR, rho_ein / rho_m, '-.', color = 'firebrick', label = 'Einasto')
	
	plt.sca(p2)
	plt.xscale('log')
	plt.xlim(rR_min, rR_max)
	plt.ylim(-4.5, -0.5)
	plt.xlabel(r'$r / R_{\rm vir}$')
	plt.ylabel(r'$d \log(\rho) / d \log(r)$')

	plt.plot(rR, slope_nfw, '--', color = 'deepskyblue')
	plt.plot(rR, slope_dk14, '-', color = 'darkblue')
	plt.plot(rR, slope_ein, '-.', color = 'firebrick')
	
	plt.sca(p1)
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
