###################################################################################################
#
# profile_composite.py      (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This unit implements a constructor for profiles that consist of an inner (orbiting, or 1-halo) term
and one ore more outer (infalling, 2-halo) terms. Please see 
:doc:`halo_profile` for a general introduction and :doc:`tutorials` for coding examples.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

from colossus.halo import profile_outer
from colossus.halo import profile_nfw
from colossus.halo import profile_hernquist
from colossus.halo import profile_einasto
from colossus.halo import profile_dk14
from colossus.halo import profile_diemer22

###################################################################################################

def compositeProfile(inner_name = None, outer_names = ['mean', 'pl'], **kwargs):
	"""
	A wrapper function to create a profile with one or many outer profile terms.
	
	At large radii, fitting functions for halo density profiles only make sense if they are 
	combined with a description of the profile of infalling matter and/or the two-halo term, that is,
	the statistical contribution from other halos. This function provides a convenient way to 
	construct such profiles without having to set the properties of the outer terms manually. Valid 
	short codes for the inner and outer terms are listed in :doc:`halo_profile`.
	
	The function can take any combination of keyword arguments that is accepted by the constructors
	of the various profile terms. Note that some parameters, such as ``z``, can be accepted by
	multiple constructors; this is by design. 
	
	Parameters
	-----------------------------------------------------------------------------------------------
	inner_name: str
		A shortcode for a density profile class (see :doc:`halo_profile` for a list).
	outer_names: array_like
		A list of shortcodes for one or more outer (infalling) terms (see :doc:`halo_profile` for 
		a list).
	kwargs: kwargs
		The arguments passed to the profile constructors.
	"""

	if inner_name == 'nfw':
		inner_cls = profile_nfw.NFWProfile
	elif inner_name == 'hernquist':
		inner_cls = profile_hernquist.HernquistProfile
	elif inner_name == 'einasto':
		inner_cls = profile_einasto.EinastoProfile
	elif inner_name == 'dk14':
		inner_cls = profile_dk14.DK14Profile
	elif inner_name == 'diemer22':
		inner_cls = profile_diemer22.ModelAProfile
	else:
		raise Exception('Unknown type of inner profile, %s.' % (str(inner_name)))
	
	outer_terms = []
	for i in range(len(outer_names)):
		if outer_names[i] == 'mean':
			outer_cls = profile_outer.OuterTermMeanDensity
		elif outer_names[i] == 'cf':
			outer_cls = profile_outer.OuterTermCorrelationFunction
		elif outer_names[i] == 'pl':
			outer_cls = profile_outer.OuterTermPowerLaw
		elif outer_names[i] == 'infalling':
			outer_cls = profile_outer.OuterTermInfalling
		else:
			raise Exception('Unknown outer term name, %s.' % (outer_names[i]))
		outer_obj = outer_cls(**kwargs)
		outer_terms.append(outer_obj)

	inner_obj = inner_cls(outer_terms = outer_terms, **kwargs)
	
	return inner_obj

###################################################################################################
