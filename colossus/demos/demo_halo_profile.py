###################################################################################################
#
# demo_halo_profile.py      (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
Sample code demonstrating the usage of the various halo.profile_* modules.
"""

###################################################################################################

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.halo import profile_outer
from colossus.halo import profile_nfw
from colossus.halo import profile_einasto
from colossus.halo import profile_dk14
from colossus.halo import profile_base

###################################################################################################

def main():

	demoProfiles()
	#demoNewProfile()

	#demoFittingLeastsq(profile = 'einasto', quantity = 'M', scatter = 0.1)
	#demoFittingLeastsq(profile = 'nfw', quantity = 'Sigma', scatter = 0.1)
	#demoFittingLeastsq(profile = 'dk14', quantity = 'rho', scatter = 0.1)
	
	#demoFittingMCMC()

	return

###################################################################################################

def demoProfiles():
	"""
	Compare the Diemer & Kravtsov 2014, NFW, and Einasto profiles of a massive cluster halo. We 
	also add various descriptions of the outer profile and plot the logarithmic slope.
	"""
	
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
									outer_term_names = ['mean', 'pl'], be = 1.0, se = 1.5))
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

def demoNewProfile():
	"""
	Derive a new profile class (for the Hernquist profile) from the base class and compare it to
	an NFW profile, using methods implemented in the HaloDensityProfile super class.
	"""
	
	class HernquistProfile(profile_base.HaloDensityProfile):
		
		def __init__(self, rhos, rs):
			
			self.par_names = ['rhos', 'rs']
			self.opt_names = []
			profile_base.HaloDensityProfile.__init__(self)
			
			self.par['rhos'] = rhos
			self.par['rs'] = rs
			
			return
		
		def densityInner(self, r):
		
			x = r / self.par['rs']
			density = self.par['rhos'] / x / (1.0 + x)**3
			
			return density

	# ---------------------------------------------------------------------------------------------

	rhos = 1E8
	rs = 200.0
	r = 10**np.arange(1.0, 3.0, 0.1)
	
	pNFW = profile_nfw.NFWProfile(rhos = rhos, rs = rs)
	pHernquist = HernquistProfile(rhos, rs)
	M_NFW = pNFW.enclosedMass(r)
	M_Hernquist = pHernquist.enclosedMass(r)

	plt.figure()
	plt.loglog()
	plt.xlabel(r'$r\ (h^{-1}\rm kpc)$')
	plt.ylabel(r'$\mathrm{Enclosed\ mass}\ (h^{-1} M_{\odot})$')
	plt.plot(r, M_NFW, '-', label = 'NFW')
	plt.plot(r, M_Hernquist, '--', label = 'Hernquist')
	plt.legend(loc = 2)
	plt.show()

	return

###################################################################################################

def demoFittingLeastsq(profile = 'nfw', quantity = 'rho', scatter = 0.2):
	"""
	This function explores fitting density profiles. The user can choose three profiles ('nfw',
	'einasto', and 'dk14') as well as different quantities to fit ('rho', 'M', 'Sigma'). Depending
	on the profile, the quantity, and the scatter chosen, the fit can take more or less time and 
	converges more or less well.
	"""
	
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
	"""
	Fit an NFW profile using both least-squares and Markov-Chain Monte Carlo techniques, and 
	compare the results.
	"""

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
