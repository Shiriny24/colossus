###################################################################################################
#
# constants.py 	        (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
Useful physical and astronomical constants.
"""

###################################################################################################
# PHYSICS CONSTANTS IN CGS
###################################################################################################

C = 2.99792458E10
"""The speed of light in cm/s."""
M_PROTON = 1.67262178e-24
"""The mass of a proton in gram."""
KB = 1.38064852E-16
"""The Boltzmann constant in erg/K."""

###################################################################################################
# ASTRONOMY UNIT CONVERSIONS
###################################################################################################

PC  = 3.08568025E18
"""A parsec in centimeters."""
KPC = 3.08568025E21
"""A kiloparsec in centimeters."""
MPC = 3.08568025E24 
"""A megaparsec in centimeters."""
YEAR = 31556926.0
"""A year in seconds."""
MSUN = 1.98892E33
"""A solar mass, :math:`M_{\odot}`, in grams."""
G = 4.30172E-6
"""The gravitational constant G in :math:`kpc \ km^2 / M_{\odot} / s^2`."""

###################################################################################################
# ASTRONOMY CONSTANTS
###################################################################################################

RHO_CRIT_0_KPC3 = 2.774848e+02
"""The critical density of the universe at z = 0 in units of :math:`M_{\odot} h^2 / kpc^3`."""
RHO_CRIT_0_MPC3 = 2.774848e+11
"""The critical density of the universe at z = 0 in units of :math:`M_{\odot} h^2 / Mpc^3`."""
DELTA_COLLAPSE = 1.686
"""The threshold overdensity for halo collapse according to the top-hat collapse model."""
