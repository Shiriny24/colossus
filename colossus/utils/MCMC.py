###################################################################################################
#
# MCMC.py 			(c) Andrey Kravtsov, Benedikt Diemer
#						University of Chicago
#     				    bdiemer@oddjob.uchicago.edu
#
###################################################################################################

"""
This module implements an Markov Chain Monte-Carlo sampler based on the Goodman & Weare (2010)
algorithm. It was written by Andrey Kravtsov and adapted for Colossus.

---------------------------------------------------------------------------------------------------
Basic usage
---------------------------------------------------------------------------------------------------



***************************************************************************************************

***************************************************************************************************

"""

import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

import acor

import Utilities

###################################################################################################

def run():
	
	return

###################################################################################################

def initWalkers(nparams, x0, step, nwalkers = 100):
	"""
	distribute initial positions of walkers in an isotropic Gaussian around the initial point
	"""
	
	numpy.random.seed(156)
	
	# In this implementation the walkers are split into 2 subgroups and thus nwalkers must be 
	# divisible by 2
	if nwalkers % 2:
		raise ValueError("MCMCsample_init: nwalkers must be divisible by 2!")
	
	x = numpy.zeros([2, nwalkers / 2, nparams])
	
	for i in range(nparams):
		x[:, :, i] = numpy.reshape(numpy.random.normal(x0[i], step[i], nwalkers), (2, nwalkers / 2))

	return x

###################################################################################################

def runMCMC(L_func, x, nparams, nwalkers = 100, nRval = 100, args = (), converged_GR = 0.01):
	
	# Parameters used to draw random number with the GW10 proposal distribution
	ap = 2.0
	api = 1.0 / ap
	afact = (ap - 1.0)
	
	# initialize some auxiliary arrays and variables 
	chain = []
	Rval = []
	
	naccept = 0
	ntry = 0
	nchain = 0
	mw = numpy.zeros((nwalkers, nparams))
	sw = numpy.zeros((nwalkers, nparams))
	m = numpy.zeros(nparams)
	Wgr = numpy.zeros(nparams)
	Bgr = numpy.zeros(nparams)
	Rgr = numpy.zeros(nparams)
	
	mutx = []
	taux = []
	for i in range(nparams): 
		mutx.append([])
		taux.append([])
		Rval.append([])
	
	gxo = numpy.zeros((2, nwalkers / 2))
	gxo[0, :] = L_func(x[0, :, :], args)
	gxo[1, :] = L_func(x[1, :, :], args)
	
	converged = False
	while not converged:
		# For parallelization (not implemented here but the MPI code can be found in code examples)
		# the walkers are split into two complementary sub-groups (see GW10)
		for kd in range(2):
			
			k = abs(kd - 1)
			
			# Vectorized inner loop of walkers stretch move in the Goodman & Weare sampling algorithm
			xchunk = x[k, :, :]
			jcompl = numpy.random.randint(0, nwalkers / 2, nwalkers / 2)
			xcompl = x[kd, jcompl, :]
			gxold  = gxo[k, :]
			
			# The next few steps implement Goodman & Weare sampling algorithm
			zf = numpy.random.rand(nwalkers / 2)   
			zf = zf * afact
			zr = (1.0 + zf) * (1.0 + zf) * api
			
			# Duplicate zr for nparams
			zrtile = numpy.transpose(numpy.tile(zr, (nparams, 1))) 
			xtry  = xcompl + zrtile * (xchunk - xcompl)
			gxtry = L_func(xtry, args)
			gx = gxold 
			ilow = numpy.where(gx < 1.0E-50)
			gx[ilow] = 1.0E-50

			# Guard against underflow in regions of very low p
			gr = gxtry / gx
			iacc = numpy.where(gr > 1.0)
			xchunk[iacc] = xtry[iacc]
			gxold[iacc] = gxtry[iacc]
			aprob = numpy.power(zr, nparams - 1) * gxtry / gx
			u = numpy.random.uniform(0.0, 1.0, numpy.shape(xchunk)[0])        
			iprob = numpy.where(aprob > u)
			xchunk[iprob] = xtry[iprob]
			gxold[iprob] = gxtry[iprob]
			naccept += len(iprob[0])
			x[k, :, :] = xchunk
			gxo[k, :] = gxold        
			
			for i in range(nwalkers / 2):
				chain.append(numpy.array(x[k, i, :]))
			
			for i in range(nwalkers / 2):
				mw[k * nwalkers / 2 + i, :] += x[k, i, :]
				sw[k * nwalkers / 2 + i, :] += x[k, i, :]**2
				ntry += 1
		
		nchain += 1
		
		# Compute means for the auto-correlation time estimate
		for i in range(nparams):
			mutx[i].append(numpy.sum(x[:, :, i]) / (nwalkers))
		
		# Compute Gelman-Rubin indicator for all parameters
		if nchain >= nwalkers / 2 and nchain % nRval == 0:
			
			# Calculate Gelman & Rubin convergence indicator
			mwc = mw / (nchain - 1.0)
			swc = sw / (nchain - 1.0) - numpy.power(mwc, 2)
			
			for i in range(nparams):
				# Within chain variance
				Wgr[i] = numpy.sum(swc[:, i]) / nwalkers
				# Mean of the means over Nwalkers
				m[i] = numpy.sum(mwc[:, i]) / nwalkers
				# Between chain variance
				Bgr[i] = nchain * numpy.sum(numpy.power(mwc[:, i] - m[i], 2)) / (nwalkers - 1.0)
				# Gelman-Rubin R factor
				Rgr[i] = (1.0 - 1.0 / nchain + Bgr[i] / Wgr[i] / nchain) * (nwalkers + 1.0) \
					/ nwalkers - (nchain - 1.0) / (nchain * nwalkers)
				tacorx = acor.acor(mutx[i])[0]
				taux[i].append(numpy.max(tacorx))
				Rval[i].append(Rgr[i] - 1.0)
			
			print "nchain=",nchain
			print "R values for parameters:", Rgr
			print "tcorr =", numpy.max(tacorx)
			
			if numpy.max(numpy.abs(Rgr - 1.0)) < converged_GR:
				converged = True

	print "MCMC sampler generated ", ntry, " samples using", nwalkers, " walkers"
	print "with step acceptance ratio of", 1.0 * naccept / ntry
	
	# Chop of burn-in period, and thin samples on auto-correlation time following Sokal's (1996) 
	# recommendations
	nthin = int(tacorx)
	nburn = int(20 * nwalkers * nthin)
	#print nburn, nthin
	
	chain = numpy.array(chain)
	#chain = chain[nburn::nthin, :]
	chain = chain[nburn:, :]
	
	R = numpy.array(Rval)
	
	return chain, R

###################################################################################################

def analyzeChain(chain, param_names, percentiles = [68.27, 95.45, 99.73], do_print = True):
	
	N_params = len(chain[0])

	mean = numpy.mean(chain, axis = 0)
	median = numpy.median(chain, axis = 0)
	stddev = numpy.std(chain, axis = 0)

	N_perc = len(percentiles)
	p = numpy.zeros((N_perc, 2, N_params), numpy.float)
	for i in range(N_perc):
		half_percentile = (100.0 - percentiles[i]) / 2.0
		p[i, 0, :] = numpy.percentile(chain, half_percentile, axis = 0)
		p[i, 1, :] = numpy.percentile(chain, 100.0 - half_percentile, axis = 0)

	if do_print:
		for i in range(N_params):
			
			Utilities.printLine()
			print 'Statistics for parameter %d, %s:' % (i + 1, param_names[i])
			print 'Mean:              %+7.3e' % (mean[i])
			print 'Median:            %+7.3e' % (median[i])
			print 'Std. dev.:         %+7.3e' % (stddev[i])
			
			for j in range(N_perc):
				print '%4.1f%% interval:    %+7.3e .. %+7.3e' % (percentiles[j], p[j, 0, i], p[j, 1, i])
		Utilities.printLine()

	return mean, median, stddev, p

###################################################################################################

def plotChain(chain, L_func, param_labels, args = ()):

	def conf_interval(x, pdf, conf_level):
		return numpy.sum(pdf[pdf > x]) - conf_level

	n_samples = len(chain)
	n_params = len(chain[0])

	# Prepare panels
	margin_lb = 1.0
	margin_rt = 0.5
	panel_size = 2.5
	size = n_params * panel_size + margin_lb + margin_rt
	fig = plt.figure(figsize = (size, size))
	gs = gridspec.GridSpec(n_params, n_params)
	margin_lb_frac = margin_lb / size
	margin_rt_frac = margin_rt / size
	plt.subplots_adjust(left = margin_lb_frac, bottom = margin_lb_frac, right = 1.0 - margin_rt_frac, \
					top = 1.0 - margin_rt_frac, hspace = margin_rt_frac, wspace = margin_rt_frac)
	panels = [[None for dummy in range(n_params)] for dummy in range(n_params)] 
	for i in range(n_params):
		for j in range(n_params):
			if i >= j:
				pan = fig.add_subplot(gs[i, j])
				panels[i][j] = pan
				if i < n_params - 1:
					pan.set_xticklabels([])
				else:
					plt.xlabel(param_labels[j])
				if j > 0:
					pan.set_yticklabels([])
				else:
					plt.ylabel(param_labels[i])
			else:
				panels[i][j] = None
					
	# Plot 1D histograms
	nbins = min(50, n_samples / 20.0)
	minmax = numpy.zeros((n_params, 2), numpy.float)
	for i in range(n_params):
		ci = chain[:, i]
		plt.sca(panels[i][i])
		_, bins, _ = plt.hist(ci, bins = nbins)
		minmax[i, 0] = bins[0]
		minmax[i, 1] = bins[-1]
		diff = minmax[i, 1] - minmax[i, 0]
		minmax[i, 0] -= 0.03 * diff
		minmax[i, 1] += 0.03 * diff
		plt.xlim(minmax[i, 0], minmax[i, 1])
	
	# Plot 2D histograms
	for i in range(n_params):
		ci = chain[:, i]
		for j in range(n_params):
			cj = chain[:, j]
			if i > j:
				plt.sca(panels[i][j])
				plt.hist2d(cj, ci, bins = 100, norm = LogNorm(), normed = 1)
				plt.ylim(minmax[i, 0], minmax[i, 1])
				plt.xlim(minmax[j, 0], minmax[j, 1])

	return

###################################################################################################

def plotGelmanRubin(R):

	plt.figure()
	plt.yscale('log')

	N_params = len(R)
	N_samples = len(R[0])
	for i in range(N_params):
		plt.plot(numpy.arange(N_samples), R[i, :], '-')
		
	return

###################################################################################################
