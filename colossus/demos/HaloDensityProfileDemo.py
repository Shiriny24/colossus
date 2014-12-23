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

import Utilities
import Cosmology
import Halo
import HaloDensityProfile

###################################################################################################

def main():

	#testProfileAccuracy()
	#demonstrateMassDefinitions()

	return

###################################################################################################

# This function compares three different implementations of the NFW density profile: 
# - the exact, analytic form
# - the generic implementation of the HaloDensityProfile base class, where only the density is 
#   computed analytically, but all other functions numerically ('Numerical')
# - a discrete profile where density and/or mass are given as arrays. Three cases are tested, with
#   only rho, only M, and both ('ArrayRho', 'ArrayM', and 'ArrayRhoM').

def testProfileAccuracy():

	# Define a class to test the generic base class HaloDensityProfile. The density is the same
	# as for the NFW profile, but all other functions of the base class are not overwritten.
	
	class TestProfile(HaloDensityProfile.HaloDensityProfile):
		
		def __init__(self, rhos, rs):
	
			HaloDensityProfile.HaloDensityProfile.__init__(self)
			self.rhos = rhos
			self.rs = rs
			
			return
		
		def density(self, r):
		
			x = r / self.rs
			density = self.rhos / x / (1.0 + x)**2
			
			return density
	
	# Set test cosmology
	Cosmology.setCosmology('WMAP9')
	
	# Properties of the test halo
	M = 1E12
	c = 10.0
	mdef = 'vir'
	z = 0.0
	
	# Radii and reshifts where to test
	r_test = numpy.array([0.011, 1.13, 10.12, 102.3, 505.0])
	z_test = 1.0
	mdef_test = '200c'

	# Parameters for the finite-resolution NFW profile; here we want to test whether this method
	# converges to the correct solution, so the resolution is high.
	r_min = 1E-2
	r_max = 1E4
	N = 1000

	# PROFILE 1: Analytical NFW profile
	prof1 = HaloDensityProfile.NFWProfile(M = M, c = c, z = z, mdef = mdef)
	rs = prof1.rs
	rhos = prof1.rhos

	# PROFILE 2: Only the density is analytical, the rest numerical
	prof2 = TestProfile(rhos = rhos, rs = rs)
	
	# PROFILES 3/4/5: User-defined NFW with finite resolution
	log_min = numpy.log10(r_min)
	log_max = numpy.log10(r_max)
	bin_width = (log_max - log_min) / N
	r_ = 10**numpy.arange(log_min, log_max + bin_width, bin_width)
	rho_ = prof1.density(r_)
	M_ = prof1.enclosedMass(r_)
	prof3 = HaloDensityProfile.SplineDensityProfile(r = r_, rho = rho_)
	prof4 = HaloDensityProfile.SplineDensityProfile(r = r_, M = M_)
	prof5 = HaloDensityProfile.SplineDensityProfile(r = r_, rho = rho_, M = M_)

	# Test for all profiles
	profs = [prof1, prof2, prof3, prof4, prof5]
	prof_names = ['Reference', 'Numerical', 'ArrayRho', 'ArrayM', 'ArrayRhoM']

	Utilities.printLine()
	print(("Profile properties as a function of radius"))
	Utilities.printLine()
	print(("Density"))
	for i in range(len(profs)):
		res = profs[i].density(r_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], numpy.abs(numpy.max(diff)))))
	
	Utilities.printLine()
	print(("Density Linear Derivative"))
	for i in range(len(profs)):
		res = profs[i].densityDerivativeLin(r_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], numpy.abs(numpy.max(diff)))))

	Utilities.printLine()
	print(("Density Logarithmic Derivative"))
	for i in range(len(profs)):
		res = profs[i].densityDerivativeLog(r_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], numpy.abs(numpy.max(diff)))))
	
	Utilities.printLine()
	print(("Enclosed mass"))
	for i in range(len(profs)):
		res = profs[i].enclosedMass(r_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], numpy.abs(numpy.max(diff)))))

	Utilities.printLine()
	print(("Surface density"))
	for i in range(len(profs)):
		res = profs[i].surfaceDensity(r_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], numpy.abs(numpy.max(diff)))))

	Utilities.printLine()
	print(("Spherical overdensity radii and masses"))
	Utilities.printLine()
	print(("Spherical overdensity radius"))
	for i in range(len(profs)):
		res = profs[i].RDelta(z_test, mdef_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], numpy.abs(numpy.max(diff)))))

	Utilities.printLine()
	print(("Spherical overdensity mass"))
	for i in range(len(profs)):
		res = profs[i].MDelta(z_test, mdef_test)
		if i == 0:
			ref = res
		else:
			diff = (res - ref) / ref
			print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], numpy.abs(numpy.max(diff)))))
	
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
