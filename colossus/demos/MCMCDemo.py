import numpy
import matplotlib.pyplot as plt

from utils import MCMC

###################################################################################################

def main():

	MCMC_test_Gaussian()

	return

###################################################################################################

def MCMC_test_Gaussian():

	n_params = 2
	param_names = ['x1', 'x2']
	
	# Set average initial position of the walkers
	x_initial = numpy.ones((n_params), numpy.float)
	walkers = MCMC.initWalkers(x_initial, nwalkers = 200, random_seed = 156)
	chain, _ = MCMC.runChain(likelihood, walkers)
	
	# Analysis and plots
	MCMC.analyzeChain(chain, param_names = param_names)
	MCMC.plotChain(chain, param_names)
	plt.savefig('MCMC_Gaussian.pdf')
	
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
