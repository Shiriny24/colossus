###################################################################################################
#
# defaults.py 	        (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
Default settings used in colossus.
"""

###################################################################################################
# COSMOLOGY
###################################################################################################

COSMOLOGY_TCMB0 = 2.7255
"""The default CMB temperature in Kelvin."""
COSMOLOGY_NEFF = 3.046
"""The default number of effective neutrino species."""

###################################################################################################
# HALO BIAS
###################################################################################################

HALO_BIAS_MODEL = 'tinker10'
"""The default halo bias model."""

###################################################################################################
# HALO CONCENTRATION
###################################################################################################

HALO_CONCENTRATION_MODEL = 'diemer15'
"""The default concentration model."""
HALO_CONCENTRATION_STATISTIC = 'median'
"""The default statistic used (mean or median). This only applies to those models that distinguish
between mean and median statistics."""

###################################################################################################
# HALO MASS
###################################################################################################

HALO_MASS_CONVERSION_PROFILE = 'nfw'
"""The default profile used for mass conversions. Whenever spherical overdensity mass definitions
are converted into one another, we have to assume a form of the density profile. The simplicity
of the NFW profile makes this computation very fast."""

###################################################################################################
# HALO PROFILE (BASE CLASS)
###################################################################################################

HALO_PROFILE_ENCLOSED_MASS_ACCURACY = 1E-6
"""Integration accuracy for enclosed mass."""
HALO_PROFILE_SURFACE_DENSITY_ACCURACY = 1E-4
"""Integration accuracy for surface density."""

###################################################################################################
# HALO PROFILE (SPECIFIC INNER PROFILES)
###################################################################################################

HALO_PROFILE_DK14_BE = 1.0
"""The default normalization of the power-law outer profile for the DK14 profile."""
HALO_PROFILE_DK14_SE = 1.5
"""The default slope of the power-law outer profile for the DK14 profile."""

###################################################################################################
# HALO PROFILE (SPECIFIC OUTER PROFILE TERMS)
###################################################################################################

HALO_PROFILE_OUTER_PL_MAXRHO = 1000.0
"""The default maximum density the power-law outer profile term can contribute to the total 
density. If this number is set too high, the power-law profile can lead to a spurious density 
contribution at very small radii, if it is set too high the power-law term will not contribute
at all."""

###################################################################################################
# MCMC
###################################################################################################

MCMC_N_WALKERS = 100
"""The number of chains (called walkers) run in parallel."""
MCMC_INITIAL_STEP = 0.1
"""A guess at the initial step taken by the walkers."""
MCMC_CONVERGENCE_STEP = 100
"""Test the convergence of the MCMC chains every n steps."""
MCMC_CONVERGED_GR = 0.01
"""Take the chains to have converged when the Gelman-Rubin statistic is smaller than this number
in all parameters."""
MCMC_OUTPUT_EVERY_N = 100
"""Output the current state of the chain every n steps."""
