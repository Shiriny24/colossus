###################################################################################################
#
# bias.py                   (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
Halo bias quantifies the excess clustering of halos over the clustering of dark matter. Bias is,
in general, a function of halo mass and scale, but this module implements only scale-free bias
models.

---------------------------------------------------------------------------------------------------
Basic usage
---------------------------------------------------------------------------------------------------

Bias can be evaluated based on either mass or peak height:

.. autosummary:: 
    haloBias
	haloBiasFromNu

The parameters that need to be passed depend on the bias model to some degree, and on whether the 
mass needs to be converted to peak height first::

	M = 1E14
	z = 0.0
	nu = peaks.peakHeight(M, z)
	b = bias.haloBiasFromNu(nu, model = 'sheth01')
	b = bias.haloBias(M, model = 'tinker10', z = z, mdef = 'vir')

The simplest bias model (cole89) is based on the peak-background split model and was derived in a 
number of different papers. The rest of the models was calibrated using numerical simulations:

============== =========================== =========================== =========================================
ID             Parameters                  z-dependence                Reference
============== =========================== =========================== =========================================
cole89         M/nu                        None                        `Cole & Kaiser 1989 <http://adsabs.harvard.edu/abs/1989MNRAS.237.1127C>`_
sheth01        M/nu                        None                        `Sheth et al. 2001 <http://adsabs.harvard.edu/abs/2001MNRAS.323....1S>`_
tinker10       M/nu, z, mdef               Through mass definition     `Tinker et al. 2010 <http://adsabs.harvard.edu/abs/2010ApJ...724..878T>`_       
============== =========================== =========================== =========================================

The Tinker et al. 2010 model was calibrated for a range of overdensities with respect to the mean
density of the universe. Thus, depending on the mass definition used, this model can predict a 
slight redshift evolution.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
from collections import OrderedDict

from colossus.utils import constants
from colossus import defaults
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.halo import mass_so

###################################################################################################

class HaloBiasModel():
	"""
	This object contains certain characteristics of a halo bias model. Currently, this object
	is empty, but the ``models`` variable is a dictionary of :class:`HaloBiasModel` objects 
	containing all available models.
	"""
		
	def __init__(self):
		return

###################################################################################################

models = OrderedDict()

models['cole89'] = HaloBiasModel()
models['sheth01'] = HaloBiasModel()
models['tinker10'] = HaloBiasModel()

###################################################################################################
# HALO BIAS
###################################################################################################

def haloBiasFromNu(nu, z = None, mdef = None, model = defaults.HALO_BIAS_MODEL):
	"""
	The halo bias at a given peak height. 

	Redshift and mass definition are necessary only for particular models (see table above).
	
	Parameters
	-----------------------------------------------------------------------------------------------
	nu: array_like
		Peak height; can be a number or a numpy array.
	z: array_like
		Redshift; can be a number or a numpy array. Only necessary for certain models.
	mdef: str
		The mass definition corresponding to the mass that was used to evaluate the peak height.
		Only necessary for certain models.
	model: str
		The bias model used.
	
	Returns
	-----------------------------------------------------------------------------------------------
	bias: array_like
		Halo bias; has the same dimensions as nu or z.

	See also
	-----------------------------------------------------------------------------------------------
	haloBias: The halo bias at a given mass. 
	"""
	
	if model == 'cole89':
		bias = modelCole89(nu)
	elif model == 'sheth01':
		bias = modelSheth01(nu)
	elif model == 'tinker10':
		bias = modelTinker10(nu, z, mdef)
	else:
		msg = 'Unkown model, %s.' % (model)
		raise Exception(msg)

	return bias

###################################################################################################

def haloBias(M, z, mdef = None, model = defaults.HALO_BIAS_MODEL):
	"""
	The halo bias at a given mass. 

	This function is a wrapper around haloBiasFromNu. The mass definition is necessary only for 
	certain models whereas the redshift is always necessary in order to convert mass to peak 
	height.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	M: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: array_like
		Redshift; can be a number or a numpy array.
	mdef: str
		The mass definition in which M is given. Only necessary for certain models.
	model: str
		The bias model used.

	Returns
	-----------------------------------------------------------------------------------------------
	bias: array_like
		Halo bias; has the same dimensions as M or z.

	See also
	-----------------------------------------------------------------------------------------------
	haloBiasFromNu: The halo bias at a given peak height. 
	"""
		
	nu = peaks.peakHeight(M, z)
	bias = haloBiasFromNu(nu, z = z, mdef = mdef, model = model)
	
	return bias

###################################################################################################

def twoHaloTerm(r, M, z, mdef, model = defaults.HALO_BIAS_MODEL):
	"""
	The 2-halo term as a function of radius and halo mass. 

	The 2-halo term in the halo-matter correlation function describes the excess density around 
	halos due to the proximity of other peaks. This contribution can be approximated as the matter-
	matter correlation function times a linear bias which depends on the peak height of the halo.
	Sometimes this term includes an additional factor of the mean density which is omitted here. 
	
	Note that this 2-halo term is also implemented as an outer profile in the colossus halo module,
	see the documentation of :mod:`halo.profile_outer`.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	r: array_like
		Halocentric radius in comoving Mpc/h; can be a number or a numpy array.
	M: float
		Halo mass in :math:`M_{\odot}/h`
	z: float
		Redshift
	mdef: str
		The mass definition in which M is given.
	model: str
		The bias model used.

	Returns
	-----------------------------------------------------------------------------------------------
	rho_2h: array_like
		The density due to the 2-halo term in physical :math:`M_{\odot}h^2/kpc^3`; has the same 
		dimensions as r.
	"""	
	
	cosmo = cosmology.getCurrent()
	bias = haloBias(M, z, mdef, model = model)
	xi = cosmo.correlationFunction(r, z)
	rho_2h = cosmo.rho_m(z) * bias * xi
	
	return rho_2h

###################################################################################################
# SPECIFIC MODELS
###################################################################################################

def modelCole89(nu):
	"""
	The peak-background split prediction for halo bias 
	
	For a derivation of this model, see e.g. Cole & Kaiser 1989 or Mo & White 1996.

	Parameters
	-----------------------------------------------------------------------------------------------
	nu: array_like
		Peak height; can be a number or a numpy array.
		
	Returns
	-----------------------------------------------------------------------------------------------
	bias: array_like
		Halo bias; has the same dimensions as nu or z.
	"""
	
	delta_c = peaks.collapseOverdensity()
	bias = 1.0 + (nu**2 - 1.0) / delta_c
	
	return bias

###################################################################################################

def modelSheth01(nu):
	"""
	The halo bias at a given peak height according to Sheth et al. 2001. 
	
	Parameters
	-----------------------------------------------------------------------------------------------
	nu: array_like
		Peak height; can be a number or a numpy array.
		
	Returns
	-----------------------------------------------------------------------------------------------
	bias: array_like
		Halo bias; has the same dimensions as nu or z.
	"""
	
	a = 0.707
	b = 0.5
	c = 0.6
	roota = np.sqrt(a)
	anu2 = a * nu**2
	anu2c = anu2**c
	t1 = b * (1.0 - c) * (1.0 - 0.5 * c)
	delta_sc = peaks.collapseOverdensity()
	bias = 1.0 +  1.0 / (roota * delta_sc) * (roota * anu2 + roota * b * anu2**(1.0 - c) - anu2c / (anu2c + t1))

	return bias

###################################################################################################

def modelTinker10(nu, z, mdef):
	"""
	The halo bias at a given peak height according to Tinker et al. 2010. 

	The halo bias, using the approximation of Tinker et al. 2010, ApJ 724, 878. The mass definition,
	mdef, must correspond to the mass that was used to evaluate the peak height. Note that the 
	Tinker bias function is universal in redshift at fixed peak height, but only for mass 
	definitions defined wrt the mean density of the universe. For other definitions, :math:`\\Delta_m`
	evolves with redshift, leading to an evolving bias at fixed peak height. 
	
	Parameters
	-----------------------------------------------------------------------------------------------
	nu: array_like
		Peak height; can be a number or a numpy array.
	z: array_like
		Redshift; can be a number or a numpy array.
	mdef: str
		The mass definition
		
	Returns
	-----------------------------------------------------------------------------------------------
	bias: array_like
		Halo bias; has the same dimensions as nu or z.
	"""
	
	if z is None:
		raise Exception('The Tinker et al. 2010 model needs a redshift to be passed.')
	if mdef is None:
		raise Exception('The Tinker et al. 2010 model needs a mass definition to be passed.')
	
	cosmo = cosmology.getCurrent()
	Delta = mass_so.densityThreshold(z, mdef) / cosmo.rho_m(z)
	y = np.log10(Delta)

	A = 1.0 + 0.24 * y * np.exp(-1.0 * (4.0 / y)**4)
	a = 0.44 * y - 0.88
	B = 0.183
	b = 1.5
	C = 0.019 + 0.107 * y + 0.19 * np.exp(-1.0 * (4.0 / y)**4)
	c = 2.4
	
	bias = 1.0 - A * nu**a / (nu**a + constants.DELTA_COLLAPSE**a) + B * nu**b + C * nu**c
	
	return bias

###################################################################################################
