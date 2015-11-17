###################################################################################################
#
# demo_mcmc.py              (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
Sample code demonstrating the usage of the various utils.mcmc module.
"""

###################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from colossus.utils import mcmc

###################################################################################################

def main():

	demoMCMC()

	return

###################################################################################################


def demoMCMC():
	"""
	This function demonstrates the use of the MCMC module for a likelihood with correlated
	parameters.
	"""
	
	n_params = 2
	param_names = ['x1', 'x2']
	x_initial = np.ones((n_params), np.float)	
	walkers = mcmc.initWalkers(x_initial, nwalkers = 200, random_seed = 156)
	chain_thin, chain_full, _ = mcmc.runChain(_likelihood, walkers)
	mcmc.analyzeChain(chain_thin, param_names = param_names)
	mcmc.plotChain(chain_full, param_names)
	plt.show()

	return

###################################################################################################

def _likelihood(x):
	
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
