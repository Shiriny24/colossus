###################################################################################################
#
# MCMCDemo.py               (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import numpy
import matplotlib.pyplot as plt

from colossus.utils import MCMC

###################################################################################################

def main():

	MCMC_test_Gaussian(plot_output = True)

	return

###################################################################################################

# This function demonstrates the use of the MCMC module. If we want to plot the chain, we need to 
# obtain it from the runChain function. If we only want a basic analysis, such as the mean and 
# median of the chain, we can use the very simple run() function.

def MCMC_test_Gaussian(plot_output = True):

	n_params = 2
	param_names = ['x1', 'x2']
	x_initial = numpy.ones((n_params), numpy.float)
	
	if plot_output:
		walkers = MCMC.initWalkers(x_initial, nwalkers = 200, random_seed = 156)
		chain_thin, chain_full, _ = MCMC.runChain(likelihood, walkers)
		MCMC.analyzeChain(chain_thin, param_names = param_names)
		MCMC.plotChain(chain_full, param_names)
		plt.savefig('MCMC_Gaussian.pdf')

	else:
		MCMC.run(x_initial, likelihood)	
		
	return

###################################################################################################

# The likelihood of two Gaussian parameters with means 0, different sigmas, and a strong correlation 
# between the parameters.

def likelihood(x):
	
	sig1 = 1.0
	sig2 = 2.0
	r = 0.95
	r2 = r * r
	res = numpy.exp(-0.5 * ((x[:, 0] / sig1)**2 + (x[:, 1] / sig2)**2 - 2.0 * r * x[:, 0] * x[:, 1] \
				/ (sig1 * sig2)) / (1.0 - r2)) / (2 * numpy.pi * sig1 * sig2) / numpy.sqrt(1.0 - r2)
	
	return res

###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
	main()
