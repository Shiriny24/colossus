###################################################################################################
#
# MCMC.py 			(c) Andrey Kravtsov, Benedikt Diemer
#						University of Chicago
#     				    bdiemer@oddjob.uchicago.edu
#
###################################################################################################

"""
This module implements an Markov Chain Monte-Carlo sampler based on the Goodman & Weare (2010)
algorithm. It was written by Andrey Kravtsov and adapted for Colossus by Benedikt Diemer.

---------------------------------------------------------------------------------------------------
Basic usage
---------------------------------------------------------------------------------------------------

Say we want to find the best-fit parameters x of some model. The likelihood for this model should 
take on a form like this::

	def likelihood(x, data, some_other_variables):

Then all we need to do to obtain the best-fit parameters is to make some initial guess for x that
is stored in the vector x_initial, and run::

	args = data, some_other_variables
	MCMC.run(x_initial, likelihood, args = args)
	
The :func:`run` function is a simple wrapper around the main stages of the MCMC samples. If, for
example, we wish to obtain the mean best-fit parameters and plot the output, we can execute the 
following code::

	args = data, some_other_variables
	walkers = MCMC.initWalkers(x_initial)
	chain_thin, chain_full, _ = MCMC.runChain(likelihood, walkers, args = args)
	mean, _, _, _ = MCMC.analyzeChain(chain_thin, param_names = param_names)
	MCMC.plotChain(chain_full, param_names)

There are numerous more advanced parameters that can be adjusted. Please see the documentation of 
the individual functions below.

---------------------------------------------------------------------------------------------------
Detailed Documentation
---------------------------------------------------------------------------------------------------
"""

import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

from . import Utilities

###################################################################################################

def run(x_initial, L_func, args = (), verbose = True, \
		# Options for the initial walker placement
		initial_step = 0.1, nwalkers = 100, random_seed = None, \
		# Options for the MCMC chain
		convergence_step = 100, converged_GR = 0.01, 
		# Options for the analysis of the chain
		param_names = None, percentiles = [68.27, 95.45, 99.73]):
	"""
	Wrapper for the lower-level MCMC functions.
	
	This function combines the creation of walkers, running of the chain and analysis functions 
	into one. Please see the documentation of the individual functions for the meaning of the 
	input parameters.
	"""
	
	walkers = initWalkers(x_initial, initial_step = initial_step, nwalkers = nwalkers, \
					random_seed = random_seed)
	chain_thin, _, _ = runChain(L_func, walkers, args = args, convergence_step = convergence_step, \
					converged_GR = converged_GR, verbose = verbose)
	mean, median, stddev, p = analyzeChain(chain_thin, param_names = param_names,
					percentiles = percentiles, verbose = verbose)

	return mean, median, stddev, p

###################################################################################################

def initWalkers(x_initial, initial_step = 0.1, nwalkers = 100, random_seed = None):
	"""
	Create a set of MCMC walkers.
	
	This function distributes the initial positions of the walkers in an isotropic Gaussian around 
	the initial guess provided by the user. The output of this function serves as an input for the
	:func:`runChain` function.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	x_initial: array_like
		A one-dimensional numpy array. The length of this array is the number of parameters 
		varied in the MCMC run. 
	initial_step: array_like
		The width of the Gaussian in parameter i. Can either be a numpy array of the same 
		length as x_initial, or a number. In the latter case, the number is multiplied with 
		x_initial, i.e. the same fractional width of the Gaussian is applied in all parameter
		dimensions.
	nwalkers: int
		The number of walkers created. In this implementation, the walkers are split into two 
		subgroups, meaning their number must be divisible by two.
	random_seed: int
		If not None, this random seed is used when generated the walker positions. 
	
	Returns
	-----------------------------------------------------------------------------------------------
	walkers: array_like
		A three-dimensional numpy array of dimensions 2, nwalkers/2, nparams. This array can be 
		passed to the :func:`runChain` function.
	"""
	
	if nwalkers % 2:
		raise ValueError("The number of walkers must be divisible by 2.")

	nparams = len(x_initial)
	walkers = numpy.zeros([2, nwalkers / 2, nparams])
	
	if random_seed is not None:
		numpy.random.seed(random_seed)
	
	if Utilities.isArray(initial_step):
		step_array = initial_step
	else:
		step_array = x_initial * initial_step
		
	for i in range(nparams):
		walkers[:, :, i] = numpy.reshape(numpy.random.normal(x_initial[i], step_array[i], nwalkers), \
										(2, nwalkers / 2))

	return walkers

###################################################################################################

def runChain(L_func, walkers, args = (), convergence_step = 100, converged_GR = 0.01, verbose = True):
	"""
	Run an MCMC chain.
	
	Run an MCMC chain using the Goodman & Weare (2010) algorithm. 
	
	Parameters
	-----------------------------------------------------------------------------------------------
	L_func: function
		The likelihood function which is maximized. This function needs to accept two parameters:
		a 2-dimensional array with dimensions (N, nparams) where N is an arbitrary number, and the
		extra arguments given in the args tuple.
	walkers: array_like	
		A three-dimensional numpy array with the initial coordinates of the walkers. This array
		must have dimensions [2, nwalkers/2, nparams], and can be generated with the 
		:func:`initWalkers` function. 
	args: tuple
		The extra arguments for the likelihood function.
	convergence_step: int
		Save and output (if verbose) the Gelman-Rubin indicator, autocorrelation time etc. every 
		convergence_step steps. 
	converged_GR: float
		The maximum difference between different chains, according to the Gelman-Rubin criterion.
		Once the GR indicator is lower than this number in all parameters, the chain is ended.
	verbose: bool
		Output information about the progress of the chain.
		
	Returns
	-----------------------------------------------------------------------------------------------
	chain_thin: array_like
		A numpy array of dimensions [n_independent_samples, nparams] with the parameters at each 
		step in the chain. In this thin chain, only every nth step is output, where n is the 
		auto-correlation time, meaning that the samples in this chain are truly independent. The 
		chain can be analyzed with the :func:`analyzeChain` function.
	chain_full: array_like
		Like the thin chain, but including all steps. Thus, the samples in this chain are not 
		indepedent from each other. However, the full chain often gives better plotting results.
	R: array_like
		A numpy array containing the GR indicator at each step when it was saved.
	"""

	# ---------------------------------------------------------------------------------------------
	
	def autocorrelationTime(x, maxlag = 50):
		
		nt = len(x)
		ft = numpy.fft.fft(x - numpy.mean(x), n = 2 * nt)
		corr_func = numpy.fft.ifft(ft * numpy.conjugate(ft))[0:nt].real
		corr_func /= corr_func[0]
		tau = 1.0 + 2.0 * numpy.sum(corr_func[1:maxlag])
		
		return tau

	# ---------------------------------------------------------------------------------------------

	if len(walkers.shape) != 3:
		raise ValueError("The walkers array must be 3-dimensional.")
	if len(walkers) != 2:
		raise ValueError("The first dimension of the walkers array must have length 1.")
	nwalkers = len(walkers[0]) * 2
	nparams = len(walkers[0][0])
	
	if verbose:
		print(('Running MCMC with the following settings:'))
		print(('Number of parameters:                 %6d' % (nparams)))
		print(('Number of walkers:                    %6d' % (nwalkers)))
		print(('Save conv. indicators every:          %6d' % (convergence_step)))
		print(('Finish when Gelman-Rubin less than:   %6.4f' % (converged_GR)))
		Utilities.printLine()
	
	# Create a copy since this array will be changed later
	x = numpy.copy(walkers)
	
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
	gxo[0, :] = L_func(x[0, :, :], *args)
	gxo[1, :] = L_func(x[1, :, :], *args)
	
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
			gxtry = L_func(xtry, *args)
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
			
			for i in range(nwalkers // 2):
				chain.append(numpy.array(x[k, i, :]))
			
			for i in range(nwalkers // 2):
				mw[k * nwalkers // 2 + i, :] += x[k, i, :]
				sw[k * nwalkers // 2 + i, :] += x[k, i, :]**2
				ntry += 1
		
		nchain += 1
		
		# Compute means for the auto-correlation time estimate
		for i in range(nparams):
			mutx[i].append(numpy.sum(x[:, :, i]) / (nwalkers))
		
		# Compute Gelman-Rubin indicator for all parameters
		if nchain % convergence_step == 0 and nchain >= nwalkers / 2 and nchain > 1:
			
			# Calculate Gelman & Rubin convergence indicator
			mwc = mw / (nchain - 1.0)
			swc = sw / (nchain - 1.0) - numpy.power(mwc, 2)
			
			for i in range(nparams):

				# Compute and store the autocorrelation time
				tacorx = autocorrelationTime(mutx[i])
				taux[i].append(numpy.max(tacorx))

				# Within chain variance
				Wgr[i] = numpy.sum(swc[:, i]) / nwalkers
				# Mean of the means over Nwalkers
				m[i] = numpy.sum(mwc[:, i]) / nwalkers
				# Between chain variance
				Bgr[i] = nchain * numpy.sum(numpy.power(mwc[:, i] - m[i], 2)) / (nwalkers - 1.0)
				# Gelman-Rubin R factor
				Rgr[i] = (1.0 - 1.0 / nchain + Bgr[i] / Wgr[i] / nchain) * (nwalkers + 1.0) \
					/ nwalkers - (nchain - 1.0) / (nchain * nwalkers)
				Rval[i].append(Rgr[i] - 1.0)
			
			if verbose:
				msg = 'Step %6d, autocorr. time %5.1f, GR = [' % (nchain, numpy.max(tacorx))
				for i in range(len(Rgr)):
					msg += ' %6.3f' % Rgr[i]
				msg += ']'
				print(msg)
			
			if numpy.max(numpy.abs(Rgr - 1.0)) < converged_GR:
				converged = True

	# Chop of burn-in period, and thin samples on auto-correlation time following Sokal's (1996) 
	# recommendations
	nthin = int(tacorx)
	nburn = int(20 * nwalkers * nthin)
	chain = numpy.array(chain)
	chain_full = chain[nburn:, :]
	chain_thin = chain_full[::nthin, :]

	R = numpy.array(Rval)

	if verbose:
		Utilities.printLine()
		print(('Acceptance ratio:                        %7.3f' % (1.0 * naccept / ntry)))
		print(('Total number of samples:                 %7d' % (ntry)))
		print(('Samples in burn-in:                      %7d' % (nburn)))
		print(('Samples without burn-in (full chain):    %7d' % (len(chain_full))))
		print(('Thinning factor (autocorr. time):        %7d' % (nthin)))
		print(('Independent samples (thin chain):        %7d' % (len(chain_thin))))

	return chain_thin, chain_full, R

###################################################################################################

def analyzeChain(chain, param_names = None, percentiles = [68.27, 95.45, 99.73], verbose = True):
	"""
	Analyze an MCMC chain.
	
	An MCMC chain represents a statistical sample of the likelihood in question. This function 
	computes more convenient parameters such as the mean, median, and various percentiles for each
	of the parameters.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	chain: array_like
		A numpy array of dimensions [nsteps, nparams] with the parameters at each step in the 
		chain. The chain is created by the :func:`runChain` function.
	param_names: array_like
		Optional; a list of strings which are used when outputting the parameters. 
	percentiles: array_like
		A list with percentages which are output. By default, the classical 1, 2, and 3 sigma
		probabilities are used.
	verbose: bool
		Print the results.
		
	Returns
	-----------------------------------------------------------------------------------------------
	x_mean: array_like
		The mean of the chain for each parameter; has length nparams.
	x_median: array_like
		The median of the chain for each parameter; has length nparams.
	x_stddev: array_like
		The standard deviation of the chain for each parameter; has length nparams.
	x_percentiles: array_like
		The lower and upper values of each parameter that contain a certain percentile of the 
		probability; has dimensions [n_percentages, 2, nparams] where the second dimension contains
		the lower/upper values. 
	"""

	nparams = len(chain[0])

	x_mean = numpy.mean(chain, axis = 0)
	x_median = numpy.median(chain, axis = 0)
	x_stddev = numpy.std(chain, axis = 0)

	nperc = len(percentiles)
	x_percentiles = numpy.zeros((nperc, 2, nparams), numpy.float)
	for i in range(nperc):
		half_percentile = (100.0 - percentiles[i]) / 2.0
		x_percentiles[i, 0, :] = numpy.percentile(chain, half_percentile, axis = 0)
		x_percentiles[i, 1, :] = numpy.percentile(chain, 100.0 - half_percentile, axis = 0)

	if verbose:
		for i in range(nparams):
			
			Utilities.printLine()
			msg = 'Statistics for parameter %d'
			if param_names is not None:
				msg += ', %s:' % param_names[i]
			else:
				msg += ':'
			print(msg)
			print(('Mean:              %+7.3e' % (x_mean[i])))
			print(('Median:            %+7.3e' % (x_median[i])))
			print(('Std. dev.:         %+7.3e' % (x_stddev[i])))
			
			for j in range(nperc):
				print(('%4.1f%% interval:    %+7.3e .. %+7.3e' \
					% (percentiles[j], x_percentiles[j, 0, i], x_percentiles[j, 1, i])))

	return x_mean, x_median, x_stddev, x_percentiles

###################################################################################################

def plotChain(chain, param_labels):
	"""
	Plot a summary of an MCMC chain.
	
	This function creates a triangle plot with a 2D histogram for each combination of parameters,
	and a 1D histogram for each parameter. The plot is not automatically saved or shown, the user
	can determine how to use the plot after executing this function.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	chain: array_like
		A numpy array of dimensions [nsteps, nparams] with the parameters at each step in the 
		chain. The chain is created by the :func:`runChain` function.
	param_labels: array_like
		A list of strings which are used when plotting the parameters. 
	"""
	
	def conf_interval(x, pdf, conf_level):
		return numpy.sum(pdf[pdf > x]) - conf_level

	nsamples = len(chain)
	nparams = len(chain[0])

	# Prepare panels
	margin_lb = 1.0
	margin_rt = 0.5
	panel_size = 2.5
	size = nparams * panel_size + margin_lb + margin_rt
	fig = plt.figure(figsize = (size, size))
	gs = gridspec.GridSpec(nparams, nparams)
	margin_lb_frac = margin_lb / size
	margin_rt_frac = margin_rt / size
	plt.subplots_adjust(left = margin_lb_frac, bottom = margin_lb_frac, right = 1.0 - margin_rt_frac, \
					top = 1.0 - margin_rt_frac, hspace = margin_rt_frac, wspace = margin_rt_frac)
	panels = [[None for dummy in range(nparams)] for dummy in range(nparams)] 
	for i in range(nparams):
		for j in range(nparams):
			if i >= j:
				pan = fig.add_subplot(gs[i, j])
				panels[i][j] = pan
				if i < nparams - 1:
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
	nbins = min(50, nsamples / 20.0)
	minmax = numpy.zeros((nparams, 2), numpy.float)
	for i in range(nparams):
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
	for i in range(nparams):
		ci = chain[:, i]
		for j in range(nparams):
			cj = chain[:, j]
			if i > j:
				plt.sca(panels[i][j])
				plt.hist2d(cj, ci, bins = 100, norm = LogNorm(), normed = 1)
				plt.ylim(minmax[i, 0], minmax[i, 1])
				plt.xlim(minmax[j, 0], minmax[j, 1])

	return

###################################################################################################
