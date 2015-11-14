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
# HALO PROFILE
###################################################################################################

HALO_PROFILE_DK14_BE = 1.0
"""The default normalization of the power-law outer profile for the DK14 profile."""
HALO_PROFILE_DK14_SE = 1.5
"""The default slope of the power-law outer profile for the DK14 profile."""

HALO_PROFILE_OUTER_PL_MAXRHO = 1000.0
"""The default maximum density the power-law outer profile term can contribute to the total 
density. If this number is set too high, the power-law profile can lead to a spurious density 
contribution at very small radii, if it is set too high the power-law term will not contribute
at all."""
