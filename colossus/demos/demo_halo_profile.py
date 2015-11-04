###################################################################################################
#
# demo_halo_profile.py      (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################
#
# Sample code demonstrating the usage of the halo.profile.py module. 
#
###################################################################################################

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from colossus.cosmology import cosmology
from colossus.halo import basics
from colossus.halo import profile

###################################################################################################

def main():

	demonstrateProfiles()
	#demonstrateFitting()
	#demonstrateMassDefinitions()

	return

###################################################################################################

# Compare the Diemer & Kravtsov 2014, NFW, and Einasto profiles of a massive cluster halo

def demonstrateProfiles():
	
	# Choose halo parameters
	M = 1E15
	mdef = 'vir'
	z = 0.0
	c = 5.0
	cosmo = cosmology.setCosmology('WMAP9')
	R = basics.M_to_R(M, z, mdef)
	
	# Choose a set of radii
	rR_min = 1E-3
	rR_max = 1E1
	rR = 10**np.arange(np.log(rR_min), np.log(rR_max), 0.02)
	r = rR * R
	rho_m = cosmo.rho_m(z)
	
	# Initialize three profiles
	p = [None, None, None]
	p[0] = profile.EinastoProfile(M = M, c = c, z = z, mdef = mdef)
	p[1] = profile.NFWProfile(M = M, c = c, z = z, mdef = mdef)
	p[2] = profile.DK14Profile(M = M, c = c, z = z, mdef = mdef, outer_terms = [])
	print(p[2].par)
	colors = ['darkblue', 'firebrick', 'deepskyblue']
	ls = ['-.', '--', '-']
	labels = ['Einasto', 'NFW', 'DK14']
	
	# Prepare plot
	fig = plt.figure(figsize = (5.5, 10.0))
	gs = gridspec.GridSpec(2, 1)
	plt.subplots_adjust(left = 0.2, right = 0.95, top = 0.95, bottom = 0.1, hspace = 0.1)
	p1 = fig.add_subplot(gs[0])	
	p2 = fig.add_subplot(gs[1])

	# Prepare density panel
	plt.sca(p1)		
	plt.loglog()
	plt.ylabel(r'$\rho / \rho_m$')
	p1.set_xticklabels([])	
	plt.xlim(rR_min, rR_max)
	plt.ylim(1E-1, 5E6)
	
	# Plot density
	for i in range(len(p)):
		rho = p[i].density(r)
		plt.plot(rR, rho / rho_m, ls = ls[i], color = colors[i], label = labels[i])

	# Prepare slope panel
	plt.sca(p2)
	plt.xscale('log')
	plt.xlim(rR_min, rR_max)
	plt.ylim(-4.5, -0.5)
	plt.xlabel(r'$r / R_{\rm vir}$')
	plt.ylabel(r'$d \log(\rho) / d \log(r)$')
	
	# Plot slope
	for i in range(len(p)):
		slope = p[i].densityDerivativeLog(r)
		plt.plot(rR, slope, ls = ls[i], color = colors[i], label = labels[i])

	# Finalize plot
	plt.sca(p1)
	plt.legend()
	plt.savefig('HaloDensityProfileDemo.pdf')
	
	return

###################################################################################################

def demonstrateFitting():

	# Create a "true" NFW profile
	cosmology.setCosmology('WMAP9')
	rhos = 1E6
	rs = 50.0
	prof = profile.NFWProfile(rhos = rhos, rs = rs)
	
	# Create a fake dataset with some noise
	r = 10**np.arange(0.1, 3.0, 0.3)
	rr = 10**np.arange(0.0, 3.0, 0.1)
	rho_data = prof.density(r)
	sigma = 0.25 * rho_data
	np.random.seed(156)
	rho_data += np.random.normal(0.0, sigma, len(r))

	# Move the profile parameters away from the initial values
	prof.setParameterArray([prof.par['rhos'] * 0.4, prof.par['rs'] * 3.0])

	# Fit to the fake data using least-squares, compute the fitted profile
	prof.fit(r, rho_data, 'rho', q_err = sigma, method = 'leastsq')	
	rho_fit_leastsq = prof.density(rr)
	
	# Fit to the fake data using MCMC, compute the fitted profile
	prof.fit(r, rho_data, 'rho', q_err = sigma, method = 'mcmc', convergence_step = 500)	
	rho_fit_mcmc = prof.density(rr)
	
	# Plot
	plt.figure()
	plt.loglog()
	plt.xlabel('r(kpc/h)')
	plt.ylabel('Density')
	plt.errorbar(r, rho_data, yerr = sigma, fmt = 'o', ms = 5.0)
	plt.plot(rr, rho_fit_leastsq, '-', label = 'leastsq')
	plt.plot(rr, rho_fit_mcmc, '--', label = 'mcmc')
	plt.legend()
	plt.show()
	
	return

###################################################################################################

# Convert one mass definition to another, assuming an NFW profile

def demonstrateMassDefinitions():
	
	Mvir = 1E12
	cvir = 10.0
	z = 0.0
	cosmology.setCosmology('WMAP9')

	Rvir = basics.M_to_R(Mvir, z, 'vir')

	print(("We start with the following halo, defined using the virial mass definition:"))	
	print(("Mvir:   %.2e Msun / h" % Mvir))
	print(("Rvir:   %.2e kpc / h" % Rvir))
	print(("cvir:   %.2f" % cvir))
	
	M200c, R200c, c200c = profile.changeMassDefinition(Mvir, cvir, z, 'vir', '200c')
	
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
