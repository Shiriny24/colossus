###################################################################################################
#
# mass_adv.py               (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module represents a collection of advanced utilities related to halo mass definitions. 

---------------------------------------------------------------------------------------------------
Changing mass definitions assuming a concentration
---------------------------------------------------------------------------------------------------

The :func:`halo.mass_defs.changeMassDefinition()` function needs to know the concentration of a 
halo. For convenience, the following function uses a concentration model to estimate the 
concentration::

	M200m, R200m, c200m = changeMassDefinitionCModel(1E12, 1.0, 'vir', '200m')
	
By default, the function uses the ``diemer_15`` concentration model (see the documentation of the
:mod:`halo.concentration` module). This function is not included in the :mod:`halo.mass_defs`
module in order to avoid circular dependencies.

---------------------------------------------------------------------------------------------------
Alternative mass definitions
---------------------------------------------------------------------------------------------------

Two alternative mass definitions were suggested in More, Diemer & Kravtsov 2015, namely the 
splashback radius and the mass within four scale radii. For routines relating to the former, please
see the :mod:`halo.splashback` module. In this module, only :math:`M_{<4r_s}` is implemented. This 
mass definition quantifies the mass in the inner part of the halo. During the fast accretion 
regime, this mass definition tracks :math:`M_{vir}`, but when the halo stops accreting it 
approaches a constant. 

.. autosummary::	
	M4rs

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

from colossus import defaults
from colossus.halo import mass_defs
from colossus.halo import profile_nfw
from colossus.halo import concentration

###################################################################################################

def changeMassDefinitionCModel(M, z, mdef_in, mdef_out, 
							profile = defaults.HALO_MASS_CONVERSION_PROFILE, 
							c_model = defaults.HALO_CONCENTRATION_MODEL):
	"""
	Change the spherical overdensity mass definition, using a model for the concentration.
	
	This function is a wrapper for the :func:`halo.mass_defs.changeMassDefinition()` function. 
	Instead of forcing the user to provide concentrations, they are computed from a model indicated 
	by the ``c_model`` parameter.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	M_i: array_like
		The initial halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z_i: float
		The initial redshift.
	mdef_i: str
		The initial mass definition.
	mdef_f: str
		The final mass definition (can be the same as mdef_i, or different).
	profile: str
		The functional form of the profile assumed in the computation; can be ``nfw`` or ``dk14``.
	c_model: str
		The identifier of a concentration model (see :mod:`halo.concentration` for valid inputs).

	Returns
	-----------------------------------------------------------------------------------------------
	Mnew: array_like
		The new halo mass in :math:`M_{\odot}/h`; has the same dimensions as M_i.
	Rnew: array_like
		The new halo radius in physical kpc/h; has the same dimensions as M_i.
	cnew: array_like
		The new concentration (now referring to the new mass definition); has the same dimensions 
		as M_i.
		
	See also
	-----------------------------------------------------------------------------------------------
	halo.mass_defs.pseudoEvolve: Evolve the spherical overdensity radius for a fixed profile.
	halo.mass_defs.changeMassDefinition: Change the spherical overdensity mass definition.
	"""
	
	c = concentration.concentration(M, mdef_in, z, model = c_model)
	
	return mass_defs.changeMassDefinition(M, c, z, mdef_in, mdef_out, profile = profile)

###################################################################################################

def M4rs(M, z, mdef, c = None):
	"""
	Convert a spherical overdensity mass to :math:`M_{<4rs}`.
	
	See the section on mass definitions for the definition of :math:`M_{<4rs}`.

	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Spherical overdensity halo mass in :math:`M_{\odot} / h`; can be a number or a numpy
		array.
	z: float
		Redshift
	mdef: str
		The spherical overdensity mass definition in which M (and optionally c) are given.
	c: array_like
		Concentration. If this parameter is not passed, concentration is automatically 
		computed. Must have the same dimensions as M.
		
	Returns
	-----------------------------------------------------------------------------------------------
	M4rs: array_like
		The mass within 4 scale radii, :math:`M_{<4rs}`, in :math:`M_{\odot} / h`; has the 
		same dimensions as M.
	"""

	if c is None:
		c = concentration.concentration(M, mdef, z)
	
	Mfrs = M * profile_nfw.NFWProfile.mu(4.0) / profile_nfw.NFWProfile.mu(c)
	
	return Mfrs

###################################################################################################
