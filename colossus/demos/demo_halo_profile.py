###################################################################################################
#
# demo_halo_profile.py      (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################
#
# Sample code demonstrating the usage of the halo.profile module. 
#
###################################################################################################

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from colossus.utils import utilities
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.halo import profile_outer
from colossus.halo import profile_nfw
from colossus.halo import profile_einasto
from colossus.halo import profile_dk14
from colossus.halo import profile_base
from colossus.halo import profile_spline

###################################################################################################

def main():

	demoProfiles()

	#demoFittingLeastsq(profile = 'einasto', quantity = 'M', scatter = 0.1)
	#demoFittingLeastsq(profile = 'nfw', quantity = 'Sigma', scatter = 0.1)
	#demoFittingLeastsq(profile = 'dk14', quantity = 'rho', scatter = 0.1)
	
	#demoFittingMCMC()

	return

###################################################################################################

# This function compares three different implementations of the NFW density profile: 
# - the exact, analytic form
# - the generic implementation of the HaloDensityProfile base class, where only the density is 
#   computed analytically, but all other functions numerically ('Numerical')
# - a discrete profile where density and/or mass are given as arrays. Three cases are tested, with
#   only rho, only M, and both ('ArrayRho', 'ArrayM', and 'ArrayRhoM').

def testProfileAccuracy():

	# Define a class to test the generic base class profile. The density is the same
	# as for the NFW profile, but all other functions of the base class are not overwritten.
	
	class TestProfile(profile_base.HaloDensityProfile):
		
		def __init__(self, rhos, rs):
	
			profile_base.HaloDensityProfile.__init__(self)
			self.rhos = rhos
			self.rs = rs
			
			return
		
		def density(self, r):
		
			x = r / self.rs
			density = self.rhos / x / (1.0 + x)**2
			
			return density
	
	# Set test cosmology
	cosmology.setCosmology('WMAP9')
	
	# Properties of the test halo
	M = 1E12
	c = 10.0
	mdef = 'vir'
	z = 0.0
	
	# Radii and reshifts where to test
	r_test = np.array([0.011, 1.13, 10.12, 102.3, 505.0])
	z_test = 1.0
	mdef_test = '200c'

	# Parameters for the finite-resolution NFW profile; here we want to test whether this method
	# converges to the correct solution, so the resolution is high.
	r_min = 1E-2
	r_max = 1E4
	N = 1000

	# PROFILE 1: Analytical NFW profile
	prof1 = profile_nfw.NFWProfile(M = M, c = c, z = z, mdef = mdef)
	rs = prof1['rs']
	rhos = prof1.rhos

	# PROFILE 2: Only the density is analytical, the rest numerical
	prof2 = TestProfile(rhos = rhos, rs = rs)
	
	# PROFILES 3/4/5: User-defined NFW with finite resolution
	log_min = np.log10(r_min)
	log_max = np.log10(r_max)
	bin_width = (log_max - log_min) / N
	r_ = 10**np.arange(log_min, log_max + bin_width, bin_width)
	rho_ = prof1.density(r_)
	M_ = prof1.enclosedMass(r_)
	prof3 = profile_spline.SplineDensityProfile(r = r_, rho = rho_)
	prof4 = profile_spline.SplineDensityProfile(r = r_, M = M_)
	prof5 = profile_spline.SplineDensityProfile(r = r_, rho = rho_, M = M_)

	# Test for all profiles
	profs = [prof1, prof2, prof3, prof4, prof5]
	prof_names = ['Reference', 'Numerical', 'ArrayRho', 'ArrayM', 'ArrayRhoM']

	utilities.printLine()
	print(("Profile properties as a function of radius"))
	utilities.printLine()
	print(("Density"))
	for i in range(len(profs)):
		res = profs[i].density(r_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], np.abs(np.max(diff)))))
	
	utilities.printLine()
	print(("Density Linear Derivative"))
	for i in range(len(profs)):
		res = profs[i].densityDerivativeLin(r_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], np.abs(np.max(diff)))))

	utilities.printLine()
	print(("Density Logarithmic Derivative"))
	for i in range(len(profs)):
		res = profs[i].densityDerivativeLog(r_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], np.abs(np.max(diff)))))
	
	utilities.printLine()
	print(("Enclosed mass"))
	for i in range(len(profs)):
		res = profs[i].enclosedMass(r_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], np.abs(np.max(diff)))))

	utilities.printLine()
	print(("Surface density"))
	for i in range(len(profs)):
		res = profs[i].surfaceDensity(r_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], np.abs(np.max(diff)))))

	utilities.printLine()
	print(("Circular velocity"))
	for i in range(len(profs)):
		res = profs[i].circularVelocity(r_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], np.abs(np.max(diff)))))
	
	utilities.printLine()
	print(("Rmax"))
	for i in range(len(profs)):
		_, res = profs[i].Vmax()
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], diff)))
	
	utilities.printLine()
	print(("Vmax"))
	for i in range(len(profs)):
		res, _ = profs[i].Vmax()
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], diff)))

	utilities.printLine()
	print(("Spherical overdensity radii and masses"))
	utilities.printLine()
	print(("Spherical overdensity radius"))
	for i in range(len(profs)):
		res = profs[i].RDelta(z_test, mdef_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], np.abs(np.max(diff)))))

	utilities.printLine()
	print(("Spherical overdensity mass"))
	for i in range(len(profs)):
		res = profs[i].MDelta(z_test, mdef_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], np.abs(np.max(diff)))))
	
	return

###################################################################################################

# Compare the Diemer & Kravtsov 2014, NFW, and Einasto profiles of a massive cluster halo. We also
# add various descriptions of the outer profile and plot the logarithmic slope.

def demoProfiles():
	
	# Choose halo parameters
	M = 1E15
	mdef = 'vir'
	z = 0.0
	c = 5.0
	cosmo = cosmology.setCosmology('WMAP9')
	R = mass_so.M_to_R(M, z, mdef)
	
	# Choose a set of radii
	rR_min = 1E-3
	rR_max = 5E1
	rR = 10**np.arange(np.log10(rR_min), np.log10(rR_max), 0.02)
	r = rR * R
	rho_m = cosmo.rho_m(z)
	
	# Initialize profiles; create an outer power-law term with a pivot at 1 Mpc/h
	outer_term = profile_outer.OuterTermMeanDensity(z = z)
	p = []
	labels = []
	p.append(profile_nfw.NFWProfile(M = M, c = c, z = z, mdef = mdef))
	labels.append('NFW (no outer)')
	p.append(profile_nfw.NFWProfile(M = M, c = c, z = z, mdef = mdef, outer_terms = [outer_term]))
	labels.append('NFW (mean)')
	p.append(profile_einasto.EinastoProfile(M = M, c = c, z = z, mdef = mdef))
	labels.append('Einasto (no outer)')
	p.append(profile_dk14.DK14Profile(M = M, c = c, z = z, mdef = mdef, outer_term_names = ['mean']))
	labels.append('DK14 (mean)')
	p.append(profile_dk14.DK14Profile(M = M, c = c, z = z, mdef = mdef, 
									outer_term_names = ['mean', 'cf'], be = 1.0, se = 1.5))
	labels.append('DK14 (mean + pl)')

	colors = ['darkblue', 'darkblue', 'firebrick', 'deepskyblue', 'deepskyblue']
	ls = ['-', '-.', '-', '--', '-']

	# Prepare plot
	fig = plt.figure(figsize = (5.5, 10.0))
	gs = gridspec.GridSpec(2, 1)
	plt.subplots_adjust(left = 0.2, right = 0.95, top = 0.95, bottom = 0.1, hspace = 0.1)
	p1 = fig.add_subplot(gs[0])	
	p2 = fig.add_subplot(gs[1])

	# Density panel
	plt.sca(p1)		
	plt.loglog()
	plt.ylabel(r'$\rho / \rho_m$')
	p1.set_xticklabels([])	
	plt.xlim(rR_min, rR_max)
	plt.ylim(1E-1, 5E6)
	for i in range(len(p)):
		rho = p[i].density(r)
		plt.plot(rR, rho / rho_m, ls = ls[i], color = colors[i], label = labels[i])

	# Slope panel
	plt.sca(p2)
	plt.xscale('log')
	plt.xlim(rR_min, rR_max)
	plt.ylim(-5.5, 0.5)
	plt.xlabel(r'$r / R_{\rm vir}$')
	plt.ylabel(r'$d \log(\rho) / d \log(r)$')
	for i in range(len(p)):
		slope = p[i].densityDerivativeLog(r)
		plt.plot(rR, slope, ls = ls[i], color = colors[i], label = labels[i])

	# Finalize plot
	plt.sca(p1)
	plt.legend()
	plt.show()
	
	return

###################################################################################################

# This function explores fitting density profiles. The user can choose three profiles ('nfw',
# 'einasto', and 'dk14') as well as different quantities to fit ('rho', 'M', 'Sigma'). Depending
# on the profile, the quantity, and the scatter chosen, the fit can take more or less time and 
# converges more or less well.

def demoFittingLeastsq(profile = 'nfw', quantity = 'rho', scatter = 0.2):
	
	cosmology.setCosmology('bolshoi')
	M = 1E12
	c = 6.0
	mdef = 'vir'
	z = 0.0
	r = 10**np.arange(0.1, 3.6, 0.1)
	
	if profile == 'nfw':
		prof = profile_nfw.NFWProfile(M = M, c = c, z = z, mdef = mdef)
		mask = np.array([True, True])
	
	elif profile == 'einasto':
		prof = profile_einasto.EinastoProfile(M = M, c = c, z = z, mdef = mdef)
		mask = np.array([True, True, True])

	elif profile == 'dk14':
		prof = profile_dk14.DK14Profile(M = M, c = c, z = z, mdef = mdef, be = 1.0, se = 1.5)
		# 'rhos', 'rs', 'rt', 'alpha', 'beta', 'gamma', 'be', 'se'
		mask = np.array([True, True, True, False, False, True, True, True])
	
	else:
		raise Exception('Invalid profile')

	# Our initial guess is wrong by 50% in all parameters
	x_true = prof.getParameterArray(mask)
	ini_guess = x_true * 1.5
	prof.setParameterArray(ini_guess, mask = mask)

	# Add scatter proportional to the profile itself
	q_true = prof.quantities[quantity](r)
	scatter_sigma = scatter * 0.3
	np.random.seed(155)
	q_err = np.abs(np.random.normal(scatter, scatter_sigma, (len(r)))) * q_true
	q = q_true.copy()
	for i in range(len(r)):
		q[i] += np.random.normal(0.0, q_err[i])
	
	# Perform the fit
	dict = prof.fit(r, q, quantity, q_err = q_err, verbose = True, mask = mask, tolerance = 1E-4)
	x = prof.getParameterArray(mask = mask)
	
	print('Solution Accuracy')
	print(x / x_true - 1.0)
	
	plt.figure()
	plt.loglog()
	plt.errorbar(r, q, yerr = q_err, fmt = '.', marker = 'o', ms = 4.0, label = 'Data')
	plt.plot(r, q_true, '--', color = 'gray', label = 'True')
	plt.plot(r, dict['q_fit'], '-', label = 'Fit')
	plt.legend()
	plt.show()
	
	return

###################################################################################################

def demoFittingMCMC():

	# Create a "true" NFW profile
	cosmology.setCosmology('WMAP9')
	rhos = 1E6
	rs = 50.0
	prof = profile_nfw.NFWProfile(rhos = rhos, rs = rs)
	
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
# Trigger
###################################################################################################

if __name__ == "__main__":
	main()
