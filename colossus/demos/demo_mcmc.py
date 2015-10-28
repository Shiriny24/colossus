###################################################################################################
#
# demo_mcmc.py              (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from colossus.utils import mcmc

###################################################################################################

def main():

	testMCMC(plot_output = True)

	return

###################################################################################################

# This function demonstrates the use of the MCMC module. If we want to plot the chain, we need to 
# obtain it from the runChain function. If we only want a basic analysis, such as the mean and 
# median of the chain, we can use the very simple run() function.

def testMCMC(plot_output = True):

	n_params = 2
	param_names = ['x1', 'x2']
	x_initial = np.ones((n_params), np.float)
	
	if plot_output:
		walkers = mcmc.initWalkers(x_initial, nwalkers = 200, random_seed = 156)
		chain_thin, chain_full, _ = mcmc.runChain(likelihood, walkers)
		mcmc.analyzeChain(chain_thin, param_names = param_names)
		mcmc.plotChain(chain_full, param_names)
		plt.savefig('MCMC_Gaussian.pdf')

	else:
		mcmc.run(x_initial, likelihood)	
		
	return

###################################################################################################

# The likelihood of two Gaussian parameters with means 0, different sigmas, and a strong correlation 
# between the parameters.

def likelihood(x):
	
	sig1 = 1.0
	sig2 = 2.0
	r = 0.95
	r2 = r * r
	res = np.exp(-0.5 * ((x[:, 0] / sig1)**2 + (x[:, 1] / sig2)**2 - 2.0 * r * x[:, 0] * x[:, 1] \
				/ (sig1 * sig2)) / (1.0 - r2)) / (2 * np.pi * sig1 * sig2) / np.sqrt(1.0 - r2)
	
	return res

###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
	main()
