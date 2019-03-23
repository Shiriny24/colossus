###################################################################################################
#
# cosmology.py              (c) Benedikt Diemer
#     				    	    benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module is an implementation of the standard FLRW cosmology with a number of dark energy models
including :math:`\Lambda CDM`, wCDM, and varying dark energy equations of state. The cosmology 
object models the contributions from dark matter, baryons, curvature, photons, neutrinos, and 
dark energy.
 
---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

In Colossus, the cosmology is set globally, and all functions respect that global cosmology. 
Colossus does not set a default cosmology, meaning that the user must set a cosmology before 
using any cosmological functions or any other functions that rely on the Cosmology module. This
documentation contains coding examples of the most common operations. Much more extensive code
samples can be found in the :doc:`tutorials`.

***************************************************************************************************
Setting and getting cosmologies
***************************************************************************************************

First, we import the cosmology module::

	from colossus.cosmology import cosmology
	
Setting a cosmology is almost always achieved with the :func:`setCosmology` function, which can be 
used in multiple ways:

* Set one of the pre-defined cosmologies::
	
	cosmology.setCosmology('planck18')

* Set one of the pre-defined cosmologies, but overwrite certain parameters::
	
	cosmology.setCosmology('planck18', {'print_warnings': False})

* Add a new cosmology to the global list of available cosmologies. This has the advantage that the 
  new cosmology can be set from anywhere in the code. Only the main cosmological parameters are 
  mandatory, all other parameters can be left to their default values::
	
	params = {'flat': True, 'H0': 67.2, 'Om0': 0.31, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
	cosmology.addCosmology('myCosmo', params)
	cosmo = cosmology.setCosmology('myCosmo')

* Set a new cosmology without adding it to the global list of available cosmologies::
	
	params = {'flat': True, 'H0': 67.2, 'Om0': 0.31, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
	cosmo = cosmology.setCosmology('myCosmo', params)

* Set a self-similar cosmology with a power-law power spectrum of a certain slope, and the 
  default settings set in the ``powerlaw`` cosmology::
	
	cosmo = cosmology.setCosmology('powerlaw_-2.60')

Whichever way a cosmology is set, the current cosmology is stored in a global variable and 
can be obtained at any time::
	
	cosmo = cosmology.getCurrent()

For more extensive examples, please see the :doc:`tutorials`.

***************************************************************************************************
Changing and switching cosmologies
***************************************************************************************************

The current cosmology can also be set to an already existing cosmology object, for example when
switching between cosmologies::

	cosmo1 = cosmology.setCosmology('WMAP9')
	cosmo2 = cosmology.setCosmology('planck18')
	cosmology.setCurrent(cosmo1)

The user can change the cosmological parameters of an existing cosmology object at run-time, but 
MUST call the update function :func:`~cosmology.cosmology.Cosmology.checkForChangedCosmology` 
directly after the changes. This function ensures that the parameters are consistent (e.g., 
flatness) and that no outdated cached quantities are used::

	cosmo = cosmology.setCosmology('WMAP9')
	cosmo.Om0 = 0.31
	cosmo.checkForChangedCosmology()

***************************************************************************************************
Summary of getter and setter functions
***************************************************************************************************

.. autosummary::
	setCosmology
	addCosmology
	setCurrent
	getCurrent

---------------------------------------------------------------------------------------------------
Standard cosmologies
---------------------------------------------------------------------------------------------------

The following sets of cosmological parameters can be chosen using the 
:func:`~cosmology.cosmology.setCosmology` function:

.. table::
	:widths: auto

	================== ================================================================================ =========== ===============================================
	ID                 Paper                                                                            Location    Explanation
	================== ================================================================================ =========== ===============================================
	planck18-only      `Planck Collab. 2018 <https://arxiv.org/abs/1807.06209>`_                        Table 2     Best-fit, Planck only (column 5) 					
	planck18           `Planck Collab. 2018 <https://arxiv.org/abs/1807.06209>`_ 	                    Table 2     Best-fit with BAO (column 6)			
	planck15-only  	   `Planck Collab. 2015 <http://adsabs.harvard.edu/abs/2016A%26A...594A..13P>`_     Table 4     Best-fit, Planck only (column 2) 					
	planck15           `Planck Collab. 2015 <http://adsabs.harvard.edu/abs/2016A%26A...594A..13P>`_ 	Table 4     Best-fit with ext (column 6)			
	planck13-only      `Planck Collab. 2013 <http://adsabs.harvard.edu/abs/2014A%26A...571A..16P>`_     Table 2     Best-fit, Planck only 					
	planck13           `Planck Collab. 2013 <http://adsabs.harvard.edu/abs/2014A%26A...571A..16P>`_     Table 5     Best-fit with BAO etc. 					
	WMAP9-only         `Hinshaw et al. 2013 <http://adsabs.harvard.edu/abs/2013ApJS..208...19H>`_       Table 2     Max. likelihood, WMAP only 				
	WMAP9-ML           `Hinshaw et al. 2013 <http://adsabs.harvard.edu/abs/2013ApJS..208...19H>`_       Table 2     Max. likelihood, with eCMB, BAO and H0 	
	WMAP9              `Hinshaw et al. 2013 <http://adsabs.harvard.edu/abs/2013ApJS..208...19H>`_       Table 4     Best-fit, with eCMB, BAO and H0 		
	WMAP7-only         `Komatsu et al. 2011 <http://adsabs.harvard.edu/abs/2011ApJS..192...18K>`_       Table 1     Max. likelihood, WMAP only 				
	WMAP7-ML           `Komatsu et al. 2011 <http://adsabs.harvard.edu/abs/2011ApJS..192...18K>`_       Table 1     Max. likelihood, with BAO and H0 		
	WMAP7 	           `Komatsu et al. 2011 <http://adsabs.harvard.edu/abs/2011ApJS..192...18K>`_	    Table 1     Best-fit, with BAO and H0 				
	WMAP5-only         `Komatsu et al. 2009 <http://adsabs.harvard.edu/abs/2009ApJS..180..330K>`_	    Table 1     Max. likelihood, WMAP only 			
	WMAP5-ML           `Komatsu et al. 2009 <http://adsabs.harvard.edu/abs/2009ApJS..180..330K>`_	    Table 1     Max. likelihood, with BAO and SN 		
	WMAP5 	           `Komatsu et al. 2009 <http://adsabs.harvard.edu/abs/2009ApJS..180..330K>`_	    Table 1     Best-fit, with BAO and SN 			
	WMAP3-ML           `Spergel et al. 2007 <http://adsabs.harvard.edu/abs/2007ApJS..170..377S>`_       Table 2     Max.likelihood, WMAP only 				
	WMAP3              `Spergel et al. 2007 <http://adsabs.harvard.edu/abs/2007ApJS..170..377S>`_       Table 5     Best fit, WMAP only 					
	WMAP1-ML           `Spergel et al. 2003 <http://adsabs.harvard.edu/abs/2003ApJS..148..175S>`_       Table 1/4   Max.likelihood, WMAP only 				
	WMAP1              `Spergel et al. 2003 <http://adsabs.harvard.edu/abs/2003ApJS..148..175S>`_       Table 7/4   Best fit, WMAP only 					
	illustris          `Vogelsberger et al. 2014 <http://adsabs.harvard.edu/abs/2014MNRAS.444.1518V>`_  --          Cosmology of the Illustris simulation
	bolshoi	           `Klypin et al. 2011 <http://adsabs.harvard.edu/abs/2011ApJ...740..102K>`_        --          Cosmology of the Bolshoi simulation
	multidark-planck   `Klypin et al. 2016 <http://adsabs.harvard.edu/abs/2016MNRAS.457.4340K>`_        Table 1     Cosmology of the Multidark-Planck simulations
	millennium         `Springel et al. 2005 <http://adsabs.harvard.edu/abs/2005Natur.435..629S>`_      --          Cosmology of the Millennium simulation 
	EdS                --                                                                               --          Einstein-de Sitter cosmology
	powerlaw           --                                                                               --          Default settings for power-law cosms.
	================== ================================================================================ =========== ===============================================

Those cosmologies that refer to particular simulations (such as ``bolshoi`` and ``millennium``) are
generally set to ignore relativistic species, i.e. photons and neutrinos, because they are not
modeled in the simulations. The ``EdS`` cosmology refers to an Einstein-de Sitter model, i.e. a flat
cosmology with only dark matter and :math:`\Omega_{\\rm m} = 1`.

---------------------------------------------------------------------------------------------------
Dark energy and curvature
---------------------------------------------------------------------------------------------------

All the default parameter sets above represent flat :math:`\Lambda CDM` cosmologies, i.e. model 
dark energy as a cosmological constant and contain no curvature. To add curvature, the default for
flatness must be overwritten, and the dark energy content of the universe must be set (which is 
otherwise computed from the matter and relativistic contributions)::

	params = cosmology.cosmologies['planck18']
	params['flat'] = False
	params['Ode0'] = 0.75
	cosmo = cosmology.setCosmology('planck_curvature', params)
	
Multiple models for the dark energy equation of state parameter :math:`w(z)` are implemented, 
namely a cosmological constant (:math:`w=-1`), a constant :math:`w`, a linearly varying 
:math:`w(z) = w_0 + w_a (1 - a)`, and arbitrary user-supplied functions for :math:`w(z)`. To set, 
for example, a linearly varying EOS, we change the ``de_model`` parameter::

	params = cosmology.cosmologies['planck18']
	params['de_model'] = 'w0wa'
	params['w0'] = -0.8
	params['wa'] = 0.1
	cosmo = cosmology.setCosmology('planck_w0wa', params)

We can implement more exotic models by supplying an arbitrary function::

	def wz_func(z):
		return -1.0 + 0.1 * z
		
	params = cosmology.cosmologies['planck18']
	params['de_model'] = 'user'
	params['wz_function'] = wz_func
	cosmo = cosmology.setCosmology('planck_wz', params)

---------------------------------------------------------------------------------------------------
Power spectrum models
---------------------------------------------------------------------------------------------------

By default, Colossus relies on fitting functions for the matter power spectrum which, in turn,
is the basis for the variance and correlation function. These models are implemented in the 
:mod:`~cosmology.power_spectrum` module, documented at the bottom of this file.

---------------------------------------------------------------------------------------------------
Derivatives and inverses
---------------------------------------------------------------------------------------------------

Almost all cosmology functions that are interpolated (e.g., 
:func:`~cosmology.cosmology.Cosmology.age`, 
:func:`~cosmology.cosmology.Cosmology.luminosityDistance()` or 
:func:`~cosmology.cosmology.Cosmology.sigma()`) can be evaluated as an nth derivative. Please note 
that some functions are interpolated in log space, resulting in a logarithmic derivative, while 
others are interpolated and differentiated in linear space. Please see the function documentations 
below for details.

The derivative functions were not systematically tested for accuracy. Their accuracy will depend
on how well the function in question is represented by the interpolating spline approximation. In 
general, the accuracy of the derivatives will be worse that the error quoted on the function itself,
and get worse with the order of the derivative.

Furthermore, the inverse of interpolated functions can be evaluated by passing ``inverse = True``.
In this case, for a function y(x), x(y) is returned instead. Those functions raise an Exception if
the requested value lies outside the range of the interpolating spline.

The inverse and derivative flags can be combined to give the derivative of the inverse, i.e. dx/dy. 
Once again, please check the function documentation whether that derivative is in linear or 
logarithmic units.

---------------------------------------------------------------------------------------------------
Performance optimization and accuracy
---------------------------------------------------------------------------------------------------

This module is optimized for fast performance, particularly in computationally intensive
functions such as the correlation function. Almost all quantities are, by 
default, tabulated, stored in files, and re-loaded when the same cosmology is set again (see the 
:mod:`~utils.storage` module for details). For some rare applications (for example, MCMC chains 
where functions are evaluated few times, but for a large number of cosmologies), the user can turn 
this behavior off::

	cosmo = cosmology.setCosmology('planck18', {'interpolation': False, 'persistence': ''})

For more details, please see the documentation of the ``interpolation`` and ``persistence`` 
parameters. In order to turn off the interpolation temporarily, the user can simply switch the 
``interpolation`` parameter off::
	
	cosmo.interpolation = False
	Pk = cosmo.matterPowerSpectrum(k)
	cosmo.interpolation = True

In this example, the power spectrum is evaluated directly without interpolation. The 
interpolation is fairly accurate (see specific notes in the function documentation), meaning that 
it is very rarely necessary to use the exact routines. 

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
import scipy.integrate
from collections import OrderedDict

from colossus import defaults
from colossus import settings
from colossus.cosmology import power_spectrum
from colossus.utils import utilities
from colossus.utils import constants
from colossus.utils import storage

###################################################################################################
# Global variables for cosmology object and pre-set cosmologies
###################################################################################################

# This variable should never be used by the user directly, but instead be handled with getCurrent
# and setCosmology.
current_cosmo = None

# The following named cosmologies can be set by calling setCosmology(name). Note that changes in
# cosmological parameters are tracked to the fourth digit, which is why all parameters are rounded
# to at most four digits. See documentation at the top of this file for references.
cosmologies = OrderedDict()
cosmologies['planck18-only']    = {'flat': True, 'H0': 67.36, 'Om0': 0.3153, 'Ob0': 0.0493, 'sigma8': 0.8111, 'ns': 0.9649}
cosmologies['planck18']      	= {'flat': True, 'H0': 67.66, 'Om0': 0.3111, 'Ob0': 0.0490, 'sigma8': 0.8102, 'ns': 0.9665}
cosmologies['planck15-only'] 	= {'flat': True, 'H0': 67.81, 'Om0': 0.3080, 'Ob0': 0.0484, 'sigma8': 0.8149, 'ns': 0.9677}
cosmologies['planck15']      	= {'flat': True, 'H0': 67.74, 'Om0': 0.3089, 'Ob0': 0.0486, 'sigma8': 0.8159, 'ns': 0.9667}
cosmologies['planck13-only'] 	= {'flat': True, 'H0': 67.11, 'Om0': 0.3175, 'Ob0': 0.0490, 'sigma8': 0.8344, 'ns': 0.9624}
cosmologies['planck13']      	= {'flat': True, 'H0': 67.77, 'Om0': 0.3071, 'Ob0': 0.0483, 'sigma8': 0.8288, 'ns': 0.9611}
cosmologies['WMAP9-only']       = {'flat': True, 'H0': 69.70, 'Om0': 0.2814, 'Ob0': 0.0464, 'sigma8': 0.8200, 'ns': 0.9710}
cosmologies['WMAP9-ML']         = {'flat': True, 'H0': 69.70, 'Om0': 0.2821, 'Ob0': 0.0461, 'sigma8': 0.8170, 'ns': 0.9646}
cosmologies['WMAP9']         	= {'flat': True, 'H0': 69.32, 'Om0': 0.2865, 'Ob0': 0.0463, 'sigma8': 0.8200, 'ns': 0.9608}
cosmologies['WMAP7-only']    	= {'flat': True, 'H0': 70.30, 'Om0': 0.2711, 'Ob0': 0.0451, 'sigma8': 0.8090, 'ns': 0.9660}
cosmologies['WMAP7-ML']      	= {'flat': True, 'H0': 70.40, 'Om0': 0.2715, 'Ob0': 0.0455, 'sigma8': 0.8100, 'ns': 0.9670}
cosmologies['WMAP7']         	= {'flat': True, 'H0': 70.20, 'Om0': 0.2743, 'Ob0': 0.0458, 'sigma8': 0.8160, 'ns': 0.9680}
cosmologies['WMAP5-only']    	= {'flat': True, 'H0': 72.40, 'Om0': 0.2495, 'Ob0': 0.0432, 'sigma8': 0.7870, 'ns': 0.9610}
cosmologies['WMAP5-ML']      	= {'flat': True, 'H0': 70.20, 'Om0': 0.2769, 'Ob0': 0.0459, 'sigma8': 0.8170, 'ns': 0.9620}
cosmologies['WMAP5']         	= {'flat': True, 'H0': 70.50, 'Om0': 0.2732, 'Ob0': 0.0456, 'sigma8': 0.8120, 'ns': 0.9600}
cosmologies['WMAP3-ML']      	= {'flat': True, 'H0': 73.20, 'Om0': 0.2370, 'Ob0': 0.0414, 'sigma8': 0.7560, 'ns': 0.9540}
cosmologies['WMAP3']         	= {'flat': True, 'H0': 73.50, 'Om0': 0.2342, 'Ob0': 0.0413, 'sigma8': 0.7420, 'ns': 0.9510}
cosmologies['WMAP1-ML']         = {'flat': True, 'H0': 68.00, 'Om0': 0.3136, 'Ob0': 0.0497, 'sigma8': 0.9000, 'ns': 0.9700}
cosmologies['WMAP1']            = {'flat': True, 'H0': 72.00, 'Om0': 0.2700, 'Ob0': 0.0463, 'sigma8': 0.9000, 'ns': 0.9900}
cosmologies['illustris']        = {'flat': True, 'H0': 70.40, 'Om0': 0.2726, 'Ob0': 0.0456, 'sigma8': 0.8090, 'ns': 0.9630, 'relspecies': False}
cosmologies['bolshoi']          = {'flat': True, 'H0': 70.00, 'Om0': 0.2700, 'Ob0': 0.0469, 'sigma8': 0.8200, 'ns': 0.9500, 'relspecies': False}
cosmologies['multidark-planck'] = {'flat': True, 'H0': 67.77, 'Om0': 0.3071, 'Ob0': 0.0482, 'sigma8': 0.8288, 'ns': 0.9600, 'relspecies': False}
cosmologies['millennium']    	= {'flat': True, 'H0': 73.00, 'Om0': 0.2500, 'Ob0': 0.0450, 'sigma8': 0.9000, 'ns': 1.0000, 'relspecies': False}
cosmologies['EdS']           	= {'flat': True, 'H0': 70.00, 'Om0': 1.0000, 'Ob0': 0.0000, 'sigma8': 0.8200, 'ns': 1.0000, 'relspecies': False}
cosmologies['powerlaw']      	= {'flat': True, 'H0': 70.00, 'Om0': 1.0000, 'Ob0': 0.0000, 'sigma8': 0.8200, 'ns': 1.0000, 'relspecies': False}

###################################################################################################
# Cosmology class
###################################################################################################

class Cosmology(object):
	"""
	A cosmology is set via the parameters passed to the constructor. Any parameter whose default
	value is ``None`` must be set by the user. The easiest way to set these parameters is to use 
	the :func:`setCosmology()` function with one of the pre-defined sets of cosmological parameters 
	listed above. 
	
	In addition, the user can choose between different equations of state for dark energy, 
	including an arbitrary :math:`w(z)` function.
	
	A few cosmological parameters are free in principle, but are well constrained and have a 
	sub-dominant impact on the computations. For such parameters, default values are pre-set so 
	that the user does not have to choose them manually. This includes the CMB temperature today 
	(``Tcmb0`` = 2.7255 K, 
	`Fixsen 2009 <https://ui.adsabs.harvard.edu//#abs/2009ApJ...707..916F/abstract>`_) 
	and the effective number of neutrino species (``Neff`` = 3.046, 
	`Planck Collaboration 2018 <https://arxiv.org/abs/1807.06209>`_). These 
	values are compatible with the most recent observational measurements and can be changed by 
	the user if necessary.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	name: str		
		A name for the cosmology, e.g. ``WMAP9`` or a user-defined name. If a user-defined set of
		cosmological parameters is used, it is advisable to use a name that does not represent any
		of the pre-set cosmologies.
	flat: bool
		If flat, there is no curvature, :math:`\Omega_{\\rm k} = 0`, and the dark energy content of the 
		universe is computed as
		:math:`\Omega_{\\rm de} = 1 - \Omega_{\\rm m} - \Omega_{\\gamma} - \Omega_{\\nu}` where 
		:math:`\Omega_{\\rm m}` is the density of matter (dark matter and baryons) in units of the
		critical density, :math:`\Omega_{\\gamma}` is the density of photons, and 
		:math:`\Omega_{\\nu}` the density of neutrinos. If ``flat == False``, the ``Ode0`` parameter
		must be passed.
	Om0: float
		:math:`\Omega_{\\rm m}`, the matter density in units of the critical density at z = 0 (includes 
		all non-relativistic matter, i.e., dark matter and baryons but not neutrinos).
	Ode0: float
		:math:`\Omega_{\\rm de}`, the dark energy density in units of the critical density at z = 0. 
		This parameter is ignored if ``flat == True``.
	Ob0: float
		:math:`\Omega_{\\rm b}`, the baryon density in units of the critical density at z = 0.
	H0: float
		The Hubble constant in km/s/Mpc.
	sigma8: float
		The normalization of the power spectrum, i.e. the variance when the field is filtered with a 
		top hat filter of radius 8 Mpc/h. See the :func:`sigma` function for details on the variance.
	ns: float
		The tilt of the primordial power spectrum.
	de_model: str
		An identifier indicating which dark energy equation of state is to be used. The DE equation
		of state can either be a cosmological constant (``de_model = lambda``), a constant w 
		(``de_model = w0``, the ``w0`` parameter must be set), a linear function of the scale 
		factor according to the parameterization of 
		`Linder 2003 <http://adsabs.harvard.edu/abs/2003PhRvL..90i1301L>`_ where 
		:math:`w(z) = w_0 + w_a (1 - a)`  (``de_model = w0wa``, the ``w0`` and ``wa`` parameters 
		must be set), or a function supplied by the user (``de_model = user``). In the latter case, 
		the w(z) function must be passed using the ``wz_function`` parameter.
	w0: float
		If ``de_model == w0``, this variable gives the constant dark energy equation of state 
		parameter w. If ``de_model == w0wa``, this variable gives the constant component w (see
		``de_model`` parameter).
	wa: float
		If ``de_model == w0wa``, this variable gives the varying component of w, otherwise it is
		ignored (see ``de_model`` parameter).
	wz_function: function
		If ``de_model == user``, this field must give a function that represents the dark energy 
		equation of state. This function must take z as its only input variable and return w(z).
	relspecies: bool
		If ``relspecies == False``, all relativistic contributions to the energy density of the 
		universe (such as photons and neutrinos) are ignored. If ``relspecies == True``, their
		energy densities are computed based on the ``Tcmb0`` and ``Neff`` parameters.
	Tcmb0: float
		The temperature of the CMB at z = 0 in Kelvin.
	Neff: float
		The effective number of neutrino species.
	power_law: bool
		Create a self-similar cosmology with a power-law matter power spectrum, 
		:math:`P(k) = k^{\\rm power\_law\_n}`. 
	power_law_n: float
		See ``power_law``.
	interpolation: bool
		By default, lookup tables are created for certain computationally intensive quantities, 
		cutting down the computation times for future calculations. If ``interpolation == False``,
		all interpolation is switched off. This can be useful when evaluating quantities for many
		different cosmologies (where computing the tables takes a prohibitively long time). 
		However, many functions will be *much* slower if this setting is ``False``, and the 
		derivatives and inverses will not work. Thus, please use ``interpolation == False`` only 
		if absolutely necessary.
	persistence: str 
		By default, interpolation tables and other data are stored in a permanent file for
		each cosmology. This avoids re-computing the tables when the same cosmology is set again. 
		However, if either read or write file access is to be avoided (for example in MCMC chains),
		the user can set this parameter to any combination of read (``'r'``) and write (``'w'``), 
		such as ``'rw'`` (read and write, the default), ``'r'`` (read only), ``'w'`` (write only), 
		or ``''`` (no persistence).
	print_info: bool
		Output information to the console.
	print_warnings: bool
		Output warnings to the console.
	"""
	
	def __init__(self, name = None,
		flat = True, Om0 = None, Ode0 = None, Ob0 = None, H0 = None, sigma8 = None, ns = None,
		de_model = 'lambda', w0 = None, wa = None, wz_function = None,
		relspecies = True, Tcmb0 = defaults.COSMOLOGY_TCMB0, Neff = defaults.COSMOLOGY_NEFF,
		power_law = False, power_law_n = 0.0,
		interpolation = True, persistence = settings.PERSISTENCE,
		print_info = False, print_warnings = True):
		
		if name is None:
			raise Exception('A name for the cosmology must be set.')
		if Om0 is None:
			raise Exception('Parameter Om0 must be set.')
		if Ob0 is None:
			raise Exception('Parameter Ob0 must be set.')
		if H0 is None:
			raise Exception('Parameter H0 must be set.')
		if sigma8 is None:
			raise Exception('Parameter sigma8 must be set.')
		if ns is None:
			raise Exception('Parameter ns must be set.')
		if Tcmb0 is None:
			raise Exception('Parameter Tcmb0 must be set.')
		if Neff is None:
			raise Exception('Parameter Neff must be set.')
		if power_law and power_law_n is None:
			raise Exception('For a power-law cosmology, power_law_n must be set.')
		
		if not flat and Ode0 is None:
			raise Exception('Ode0 must be set for non-flat cosmologies.')
		if Ode0 is not None and Ode0 < 0.0:
			raise Exception('Ode0 cannot be negative.')
		if not de_model in ['lambda', 'w0', 'w0wa', 'user']:
			raise Exception('Unknown dark energy type, %s. Valid types include lambda, w0, w0wa, and user.' % (de_model))
		if de_model == 'user' and wz_function is None:
			raise Exception('If de_model is user, a function must be passed for wz_function.')
		if de_model == 'lambda':
			w0 = -1
			wa = None
	
		# Copy the cosmological parameters into the class
		self.name = name
		self.flat = flat
		self.Om0 = Om0
		self.Ode0 = Ode0
		self.Ob0 = Ob0
		self.H0 = H0
		self.sigma8 = sigma8
		self.ns = ns
		self.de_model = de_model
		self.w0 = w0
		self.wa = wa
		self.wz_function = wz_function
		self.relspecies = relspecies
		self.Tcmb0 = Tcmb0
		self.Neff = Neff
		self.power_law = power_law
		self.power_law_n = power_law_n

		# Compute some derived cosmological variables
		self.h = H0 / 100.0
		self.h2 = self.h**2
		self.Omh2 = self.Om0 * self.h2
		self.Ombh2 = self.Ob0 * self.h2
		
		if self.relspecies:
			# To convert the CMB temperature into a fractional energy density, we follow these
			# steps:
			# 
			# rho_gamma   = 4 sigma_SB / c * T_CMB^4 [erg/cm^3]
			#             = 4 sigma_SB / c^3 * T_CMB^4 [g/cm^3]
			#
			# where sigmaSB = 5.670373E-5 erg/cm^2/s/K^4. Then,
			#
			# Omega_gamma = rho_gamma / (Msun/g) * (kpc/cm)^3 / h^2 / constants.RHO_CRIT_0_KPC3
			#
			# Most of these steps can be summarized in one constant.
			self.Ogamma0 = 4.48131796342E-07 * self.Tcmb0**4 / self.h2
			
			# The energy density in neutrinos is 7/8 (4/11)^(4/3) times the energy density in 
			# photons, per effective neutrino species.
			self.Onu0 = 0.22710731766 * self.Neff * self.Ogamma0
			
			# The density of relativistic species is the sum of the photon and neutrino densities.
			self.Or0 = self.Ogamma0 + self.Onu0
			
			# For convenience, compute the epoch of matter-radiation equality
			self.a_eq = self.Or0 / self.Om0
		else:
			self.Ogamma0 = 0.0
			self.Onu0 = 0.0
			self.Or0 = 0.0

		# Make sure flatness is obeyed
		self._ensureConsistency()
		
		# Flag for interpolation tables, printing etc
		self.interpolation = interpolation
		self.print_info = print_info
		self.print_warnings = print_warnings
		
		# Create a storage object
		self.storageUser = storage.StorageUser('cosmology', persistence, self.getName, 
									self._getHashableString, self._ensureConsistency)
				
		# Lookup table for functions of z. This table runs from the future (a = 200.0) to 
		# a = 0.005. Due to some interpolation errors at the extrema of the range, the table 
		# runs to slightly lower and higher z than the interpolation is allowed for.
		self.z_min = -0.995
		self.z_min_compute = -0.998
		self.z_max = 200.01
		self.z_max_compute = 500.0
		self.z_Nbins = 50
		
		# Lookup table for P(k). The Pk_norm field is only needed if interpolation == False.
		# Note that the binning is highly irregular for P(k), since much more resolution is
		# needed at the BAO scale and around the bend in the power spectrum. Thus, the binning
		# is split into multiple regions with different resolutions.
		self.k_Pk = [1E-20, 1E-4, 5E-2, 1E0, 1E6, 1E20]
		self.k_Pk_Nbins = [10, 30, 60, 20, 10]
		
		# Lookup table for sigma. Note that the nominal accuracy to which the integral is 
		# evaluated should match with the accuracy of the interpolation which is set by Nbins.
		# Here, they are matched to be accurate to better than ~3E-3.
		self.R_min_sigma = 1E-12
		self.R_max_sigma = 1E3
		self.R_Nbins_sigma = 18.0
		self.accuracy_sigma = 3E-3
	
		# Lookup table for correlation function xi
		self.R_xi = [1E-3, 5E1, 5E2]
		self.R_xi_Nbins = [30, 40]
		self.accuracy_xi = 1E-5

		return

	###############################################################################################

	def __str__(self):
		
		de_str = 'de_model = %s, ' % (str(self.de_model))
		if self.de_model in ['lambda', 'user']:
			pass
		elif self.de_model == 'w0':
			de_str += 'w0 = %.4f, ' % (self.w0)
		elif self.de_model == 'w0wa':
			de_str += 'w0 = %.4f, wa = %.4f, ' % (self.w0, self.wa)
		else:
			raise Exception('Unknown dark energy type, %s.' % self.de_model)
		
		pl_str = 'powerlaw = %s' % (str(self.power_law))
		if self.power_law:
			pl_str += ', PLn = %.4f' % (self.power_law_n)
			
		s = 'Cosmology "%s" \n' \
			'    flat = %s, Om0 = %.4f, Ode0 = %.4f, Ob0 = %.4f, H0 = %.2f, sigma8 = %.4f, ns = %.4f\n' \
			'    %srelspecies = %s, Tcmb0 = %.4f, Neff = %.4f, %s' \
			% (self.name, 
			str(self.flat), self.Om0, self.Ode0, self.Ob0, self.H0, self.sigma8, self.ns, 
			de_str, str(self.relspecies), self.Tcmb0, self.Neff, pl_str)
		
		return s

	###############################################################################################

	# Compute a unique hash for the current cosmology name and parameters. If any of them change,
	# the hash will change, causing an update of stored quantities.
		
	def _getHashableString(self):
	
		param_string = 'name_%s' % (self.name)
		param_string += '_flat_%s' % (str(self.flat))
		param_string += '_Om0_%.6f' % (self.Om0)
		param_string += '_Ode0_%.6f' % (self.Ode0)
		param_string += '_Ob0_%.6f' % (self.Ob0)
		param_string += '_H0_%.6f' % (self.H0)
		param_string += '_sigma8_%.6f' % (self.sigma8)
		param_string += '_ns_%.6f' % (self.ns)

		param_string += '_detype_%s' % (self.de_model)
		if self.w0 is not None:
			param_string += '_w0_%.6f' % (self.w0)
		if self.wa is not None:
			param_string += '_wa_%.6f' % (self.wa)

		param_string += '_relspecies_%s' % (str(self.relspecies))	
		param_string += '_Tcmb0_%.6f' % (self.Tcmb0)
		param_string += '_Neff_%.6f' % (self.Neff)
		
		param_string += '_PL_%s' % (str(self.power_law))
		param_string += '_PLn_%.6f' % (self.power_law_n)
	
		return param_string

	###############################################################################################

	def getName(self):
		"""
		Return the name of this cosmology.
		"""
		
		return self.name

	###############################################################################################

	def checkForChangedCosmology(self):
		"""
		Check whether the cosmological parameters have been changed by the user. If there are 
		changes, all pre-computed quantities (e.g., interpolation tables) are discarded and 
		re-computed if necessary.
		"""
		
		if self.storageUser.checkForChangedHash():
			if self.print_warnings:
				print("Cosmology: Detected change in cosmological parameters.")
			self._ensureConsistency()
			self.storageUser.resetStorage()
			
		return
	
	###############################################################################################
	# Utilities for internal use
	###############################################################################################
	
	# Depending on whether the cosmology is flat or not, Ode0 and Ok0 take on certain values.

	def _ensureConsistency(self):
		
		if self.flat:
			self.Ode0 = 1.0 - self.Om0 - self.Or0
			self.Ok0 = 0.0
			if self.Ode0 < 0.0:
				raise Exception('Ode0 cannot be less than zero. If Om = 1, relativistic species must be off.')
		else:
			self.Ok0 = 1.0 - self.Ode0 - self.Om0 - self.Or0

		return

	###############################################################################################
	# Basic cosmology calculations
	###############################################################################################
	
	# The redshift scaling of dark energy. This function should not be used directly but goes into
	# the results of Ez(), rho_de(), and Ode(). The general case of a user-defined function is
	# given in Linder 2003 Equation 5. For the w0-wa parameterization, this integral evaluates to
	# an analytical expression.
	
	def _rho_de_z(self, z):
		
		def _de_integrand(ln_zp1):
			z = np.exp(ln_zp1) -  1.0
			ret = 1.0 + self.wz_function(z)
			return ret
		
		if self.de_model == 'lambda':
			
			de_z = 1.0
		
		elif self.de_model == 'w0':
			
			de_z = (1.0 + z)**(3.0 * (1.0 + self.w0))
		
		elif self.de_model == 'w0wa':
			
			a = 1.0 / (1.0 + z)
			de_z = a**(-3.0 * (1.0 + self.w0 + self.wa)) * np.exp(-3.0 * self.wa * (1.0 - a))
		
		elif self.de_model == 'user':
			
			z_array, is_array = utilities.getArray(z)
			de_z = np.zeros_like(z_array)
			for i in range(len(z_array)):
				integral, _ = scipy.integrate.quad(_de_integrand, 0, np.log(1.0 + z_array[i]))
				de_z[i] = np.exp(3.0 * integral)
			if not is_array:
				de_z = de_z[0]

		else:
			raise Exception('Unknown de_model, %s.' % (self.de_model))
		
		return de_z

	###############################################################################################
	
	def Ez(self, z):
		"""
		The Hubble parameter as a function of redshift, in units of :math:`H_0`.
		
		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		E: array_like
			:math:`H(z) / H_0`; has the same dimensions as ``z``.

		See also
		-------------------------------------------------------------------------------------------
		Hz: The Hubble parameter as a function of redshift.
		"""
		
		zp1 = (1.0 + z)
		sum = self.Om0 * zp1**3 + self.Ode0 * self._rho_de_z(z)
		if not self.flat:
			sum += self.Ok0 * zp1**2
		if self.relspecies:
			sum += self.Or0 * zp1**4
		E = np.sqrt(sum)
		
		return E

	###############################################################################################
	
	def Hz(self, z):
		"""
		The Hubble parameter as a function of redshift.
		
		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		H: array_like
			:math:`H(z)` in units of km/s/Mpc; has the same dimensions as ``z``.

		See also
		-------------------------------------------------------------------------------------------
		Ez: The Hubble parameter as a function of redshift, in units of :math:`H_0`.
		"""
		
		H = self.Ez(z) * self.H0
					
		return H

	###############################################################################################

	def wz(self, z):
		"""
		The dark energy equation of state parameter.
		
		The EOS parameter is defined as :math:`w(z) = P(z) / \\rho(z)`. Depending on its chosen 
		functional form (see the ``de_model`` parameter to :func:`~cosmology.cosmology.Cosmology`), 
		w(z) can be -1, another constant, a linear function of a, or an arbitrary function chosen 
		by the user.
	
		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		w: array_like
			:math:`w(z)`, has the same dimensions as ``z``.
		"""
	
		if self.de_model == 'lambda':
			w = np.ones_like(z) * -1.0
		elif self.de_model == 'w0':
			w = np.ones_like(z) * self.w0
		elif self.de_model == 'w0wa':
			w = self.w0 + self.wa * z / (1.0 + z)
		elif self.de_model == 'user':
			w = self.wz_function(z)
		else:
			raise Exception('Unknown de_model, %s.' % (self.de_model))
				
		return w

	###############################################################################################

	# Standard cosmological integrals. These integrals are not persistently stored in files because
	# they can be evaluated between any two redshifts which would make the tables very large.
	#
	# z_min and z_max can be numpy arrays or numbers. If one of the two is a number and the other an
	# array, the same z_min / z_max is used for all z_min / z_max in the array (this is useful if 
	# z_max = inf, for example).
	
	def _integral(self, integrand, z_min, z_max):

		min_is_array = utilities.isArray(z_min)
		max_is_array = utilities.isArray(z_max)
		use_array = min_is_array or max_is_array
		
		if use_array and not min_is_array:
			z_min_use = np.array([z_min] * len(z_max))
		else:
			z_min_use = z_min
		
		if use_array and not max_is_array:
			z_max_use = np.array([z_max] * len(z_min))
		else:
			z_max_use = z_max
		
		if use_array:
			if min_is_array and max_is_array and len(z_min) != len(z_max):
				raise Exception("If both z_min and z_max are arrays, they need to have the same size.")
			integ = np.zeros_like(z_min_use)
			for i in range(len(z_min_use)):
				integ[i], _ = scipy.integrate.quad(integrand, z_min_use[i], z_max_use[i])
		else:
			integ, _ = scipy.integrate.quad(integrand, z_min, z_max)
		
		return integ
	
	###############################################################################################

	# The integral over 1 / E(z) enters into the comoving distance.

	def _integral_oneOverEz(self, z_min, z_max = np.inf):
		
		def integrand(z):
			return 1.0 / self.Ez(z)
		
		return self._integral(integrand, z_min, z_max)

	###############################################################################################

	# The integral over 1 / E(z) / (1 + z) enters into the age of the universe.

	def _integral_oneOverEz1pz(self, z_min, z_max = np.inf):
		
		def integrand(z):
			return 1.0 / self.Ez(z) / (1.0 + z)
		
		return self._integral(integrand, z_min, z_max)

	###############################################################################################

	# Used by _zFunction

	def _zInterpolator(self, table_name, func, inverse = False, future = True):

		table_name = table_name + '_%s' % (self.name) 
		interpolator = self.storageUser.getStoredObject(table_name, interpolator = True, inverse = inverse)
		
		if interpolator is None:
			if self.print_info:
				print("Computing lookup table in z.")
			
			if future:
				log_min = np.log(1.0 + self.z_min_compute)
			else:
				log_min = 0.0
			log_max = np.log(1.0 + self.z_max_compute)
			zp1_log_table = np.linspace(log_min, log_max, self.z_Nbins)
			z_table = np.exp(zp1_log_table) - 1.0
			x_table = func(z_table)
			
			self.storageUser.storeObject(table_name, np.array([zp1_log_table, x_table]))
				
			if self.print_info:
				print("Lookup table completed.")
			interpolator = self.storageUser.getStoredObject(table_name, interpolator = True, inverse = inverse)
		
		return interpolator

	###############################################################################################

	# General container for methods that are functions of z and use interpolation. This function is
	# somewhat complicated because redshift is stored in a logarithmic table, leading to various
	# complications with derivatives, inverses and so on. The logarithmic spacing leads to a better
	# interpolation accuracy though.
	
	def _zFunction(self, table_name, func, z, inverse = False, future = True, derivative = 0):

		if self.interpolation:
			
			# Get interpolator. If it does not exist, create it.
			interpolator = self._zInterpolator(table_name, func, inverse = inverse, future = future)
			
			# Check limits of z array. If inverse == True, we need to check the limits on 
			# the result function. But even if we are evaluating a z-function it's good to check 
			# the limits on the interpolator. For example, some functions can be evaluated in the
			# future while others cannot.
			if inverse:
				min_ = interpolator.get_knots()[0]
				max_ = interpolator.get_knots()[-1]
			else:
				min_ = np.exp(interpolator.get_knots()[0]) - 1.0
				max_ = np.exp(interpolator.get_knots()[-1]) - 1.0
			
			if np.min(z) < min_:
				if inverse:
					msg = "Value f = %.3f outside range of interpolation table (min. %.3f)." % (np.min(z), min_)
				else:
					msg = "Redshift z = %.3f outside range of interpolation table (min. z is %.3f)." % (np.min(z), min_)
				raise Exception(msg)
				
			if np.max(z) > max_:
				if inverse:
					msg = "Value f = %.3f outside range of interpolation table (max. f is %.3f)." % (np.max(z), max_)
				else:
					msg = "Redshift z = %.3f outside range of interpolation table (max. z is %.3f)." % (np.max(z), max_)
				raise Exception(msg)

			if inverse:
				f = z
				ln_zp1 = interpolator(f, nu = 0)
				zp1 = np.exp(ln_zp1)
				
				if derivative == 0:
					ret = zp1 - 1.0
				elif derivative == 1:
					dx_df = interpolator(f, nu = 1)
					ret = dx_df * zp1
				elif derivative == 2:
					dx_df = interpolator(f, nu = 1)
					dx2_df2 = interpolator(f, nu = 2)
					ret = (dx2_df2 + dx_df**2) * zp1
				elif derivative > 2:
					raise Exception('Only the first and second derivatives are implemented.')
			
			else:			
				zp1 = 1.0 + z
				ln_zp1 = np.log(zp1)
				x = ln_zp1
				ret = interpolator(x, nu = derivative)
				
				if derivative == 1:
					ret /= zp1
				elif derivative == 2:
					df_dx = interpolator(ln_zp1, nu = 1)
					ret = (ret - df_dx) / (1.0 + z)**2
				elif derivative > 2:
					raise Exception('Only the first and second derivatives are implemented.')
				
		else:
			if derivative > 0:
				raise Exception("Derivative can only be evaluated if interpolation == True.")
			if inverse:
				raise Exception("Inverse can only be evaluated if interpolation == True.")

			ret = func(z)
		
		return ret

	###############################################################################################
	# Times & distances
	###############################################################################################
	
	def hubbleTime(self, z):
		"""
		The Hubble time, :math:`1/H(z)`.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		tH: float
			:math:`1/H` in units of Gyr; has the same dimensions as ``z``.

		See also
		-------------------------------------------------------------------------------------------
		lookbackTime: The lookback time since z.
		age: The age of the universe at redshift z.
		"""
		
		tH = 1E-16 * constants.MPC / constants.YEAR / self.h / self.Ez(z)
		
		return tH
	
	###############################################################################################

	def _lookbackTimeExact(self, z):
		
		t = self.hubbleTime(0.0) * self._integral_oneOverEz1pz(0.0, z)

		return t

	###############################################################################################

	def lookbackTime(self, z, derivative = 0, inverse = False):
		"""
		The lookback time since redshift z.
		
		The lookback time corresponds to the difference between the age of the universe at 
		redshift z and today.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift, where :math:`-0.995 < z < 200`; can be a number or a numpy array.
		derivative: int
			If greater than 0, evaluate the nth derivative, :math:`d^nt/dz^n`.
		inverse: bool
			If True, evaluate :math:`z(t)` instead of :math:`t(z)`. In this case, the ``z`` field
			must contain the time(s) in Gyr.

		Returns
		-------------------------------------------------------------------------------------------
		t: array_like
			The lookback time (or its derivative) since z in units of Gigayears; has the same 
			dimensions as z.

		See also
		-------------------------------------------------------------------------------------------
		hubbleTime: The Hubble time, :math:`1/H_0`.
		age: The age of the universe at redshift z.
		"""
		
		t = self._zFunction('lnzp1_lookbacktime', self._lookbackTimeExact, z, derivative = derivative,
						inverse = inverse)
		
		return t
	
	###############################################################################################

	def _ageExact(self, z):
		
		t = self.hubbleTime(0.0) * self._integral_oneOverEz1pz(z, np.inf)
		
		return t
	
	###############################################################################################
	
	def age(self, z, derivative = 0, inverse = False):
		"""
		The age of the universe at redshift z.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift, where :math:`-0.995 < z < 200`; can be a number or a numpy array.
		derivative: int
			If greater than 0, evaluate the nth derivative, :math:`d^nt/dz^n`.
		inverse: bool
			If True, evaluate :math:`z(t)` instead of :math:`t(z)`. In this case, the ``z`` field
			must contain the time(s) in Gyr.

		Returns
		-------------------------------------------------------------------------------------------
		t: array_like
			The age of the universe (or its derivative) at redshift z in Gigayears; has the 
			same dimensions as z.

		See also
		-------------------------------------------------------------------------------------------
		hubbleTime: The Hubble time, :math:`1/H_0`.
		lookbackTime: The lookback time since z.
		"""

		t = self._zFunction('lnzp1_age', self._ageExact, z, derivative = derivative, inverse = inverse)
		
		return t
	
	###############################################################################################

	def comovingDistance(self, z_min = 0.0, z_max = 0.0, transverse = True):
		"""
		The transverse or line-of-sight comoving distance.
		
		This function returns the comoving distance between two points. Depending on the chosen 
		geometry, the output can have two different meanings. If ``transverse = False``, the 
		line-of-sight distance is returned,
		
		.. math::
			d_{\\rm com,los}(z) = \\frac{c}{H_0} \\int_{0}^{z} \\frac{1}{E(z)} .
		
		However, if ``transverse = False``, the function returns the comoving distance between two
		points separated by an angle of one radian at ``z_max`` (if ``z_min`` is zero). This 
		quantity depends on the spatial curvature of the universe,
		
		.. math::
			d_{\\rm com,trans}(z) =  \\left\\{
			\\begin{array}{ll}
			      \\frac{c/H_0}{\\sqrt{\\Omega_{\\rm k,0}}} \\sinh \\left(\\frac{\\sqrt{\\Omega_{\\rm k,0}}}{c/H_0} d_{\\rm com,los} \\right) & \\forall \\, \\Omega_{\\rm k,0} > 0 \\\\
			      d_{\\rm com,los} & \\forall \\, \\Omega_{\\rm k,0} = 0 \\\\
			      \\frac{c/H_0}{\\sqrt{-\\Omega_{\\rm k,0}}} \\sin \\left(\\frac{\\sqrt{-\\Omega_{\\rm k,0}}}{c/H_0} d_{\\rm com,los} \\right) & \\forall \\, \\Omega_{\\rm k,0} < 0 \\\\
			\\end{array} 
			\\right.		
		
		In Colossus, this distance is referred to as the "transverse comoving distance"
		(e.g., Hogg 1999), but a number of other terms are used in the literature, e.g., 
		"comoving angular diameter distance" (Dodelson 2003), "comoving coordinate distance"
		(Mo et al. 2010),or "angular size distance" (Peebles 1993). The latter is not to be 
		confused with the angular diameter distance.
		
		Either ``z_min`` or ``z_max`` can be a numpy array; in those cases, the same ``z_min`` / 
		``z_max`` is applied to all values of the other. If both are numpy arrays, they need to 
		have the same dimensions, and the comoving distance returned corresponds to a series of 
		different ``z_min`` and ``z_max`` values. 

		This function does not use interpolation (unlike the other distance functions) because it
		accepts both ``z_min`` and ``z_max`` parameters which would necessitate a 2D interpolation. 
		Thus, for fast evaluation, the :func:`luminosityDistance` and 
		:func:`angularDiameterDistance` functions should be used.

		Parameters
		-------------------------------------------------------------------------------------------
		zmin: array_like
			Redshift; can be a number or a numpy array.
		zmax: array_like
			Redshift; can be a number or a numpy array.
		transverse: bool
			Whether to return the transverse of line-of-sight comoving distance. The two are the 
			same in flat cosmologies.

		Returns
		-------------------------------------------------------------------------------------------
		d: array_like
			The comoving distance in Mpc/h; has the same dimensions as ``zmin`` and/or ``zmax``.

		See also
		-------------------------------------------------------------------------------------------
		luminosityDistance: The luminosity distance to redshift z.
		angularDiameterDistance: The angular diameter distance to redshift z.
		"""

		d = self._integral_oneOverEz(z_min = z_min, z_max = z_max)
		
		if not self.flat and transverse:
			if self.Ok0 > 0.0:
				sqrt_Ok0 = np.sqrt(self.Ok0)
				d = np.sinh(sqrt_Ok0 * d) / sqrt_Ok0
			else:
				sqrt_Ok0 = np.sqrt(-self.Ok0)
				d = np.sin(sqrt_Ok0 * d) / sqrt_Ok0
		
		# Note that the formula is c/H0 * integral(1/E(z)). The H0 is not written here because the
		# distance returned is in /h units. The 1E-7 factor comes from the conversion between the 
		# speed of light in cm/s to km/s and H0 = 100 * h.
		
		d *= constants.C * 1E-7
		
		return d

	###############################################################################################

	def _luminosityDistanceExact(self, z):
		
		d = self.comovingDistance(z_min = 0.0, z_max = z, transverse = True) * (1.0 + z)

		return d

	###############################################################################################
	
	def luminosityDistance(self, z, derivative = 0, inverse = False):
		"""
		The luminosity distance to redshift z.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift, where :math:`-0.995 < z < 200`; can be a number or a numpy array.
		derivative: int
			If greater than 0, evaluate the nth derivative, :math:`d^nD/dz^n`.
		inverse: bool
			If True, evaluate :math:`z(D)` instead of :math:`D(z)`. In this case, the ``z`` field
			must contain the luminosity distance in Mpc/h.
			
		Returns
		-------------------------------------------------------------------------------------------
		d: array_like
			The luminosity distance (or its derivative) in Mpc/h; has the same dimensions as ``z``.

		See also
		-------------------------------------------------------------------------------------------
		comovingDistance: The comoving distance between redshift :math:`z_{\\rm min}` and :math:`z_{\\rm max}`.
		angularDiameterDistance: The angular diameter distance to redshift z.
		"""
		
		d = self._zFunction('lnzp1_luminositydist', self._luminosityDistanceExact, z,
						future = False, derivative = derivative, inverse = inverse)
		
		return d
	
	###############################################################################################

	def _angularDiameterDistanceExact(self, z):
		
		d = self.comovingDistance(z_min = 0.0, z_max = z, transverse = True) / (1.0 + z)
		
		return d

	###############################################################################################

	def angularDiameterDistance(self, z, derivative = 0):
		"""
		The angular diameter distance to redshift z.
		
		The angular diameter distance is the transverse distance that, at redshift z, corresponds 
		to an angle of one radian. Note that the inverse is not available for this function 
		because it is not strictly increasing or decreasing with redshift, making its inverse
		multi-valued.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift, where :math:`-0.995 < z < 200`; can be a number or a numpy array.
		derivative: int
			If greater than 0, evaluate the nth derivative, :math:`d^nD/dz^n`.

		Returns
		-------------------------------------------------------------------------------------------
		d: array_like
			The angular diameter distance (or its derivative) in Mpc/h; has the same dimensions as 
			``z``.

		See also
		-------------------------------------------------------------------------------------------
		comovingDistance: The comoving distance between redshift :math:`z_{\\rm min}` and :math:`z_{\\rm max}`.
		luminosityDistance: The luminosity distance to redshift z.
		"""

		d = self._zFunction('lnzp1_angdiamdist', self._angularDiameterDistanceExact, z,
						future = False, derivative = derivative, inverse = False)
		
		return d

	###############################################################################################

	# This function is not interpolated because the distance modulus is not defined at z = 0.

	def distanceModulus(self, z):
		"""
		The distance modulus to redshift z in magnitudes.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		mu: array_like
			The distance modulus in magnitudes; has the same dimensions as ``z``.
		"""
		
		mu = 5.0 * np.log10(self.luminosityDistance(z) / self.h * 1E5)
		
		return mu

	###############################################################################################

	def soundHorizon(self):
		"""
		The sound horizon at recombination.

		This function returns the sound horizon in Mpc/h, according to 
		`Eisenstein & Hu 1998 <http://adsabs.harvard.edu/abs/1998ApJ...496..605E>`_, Equation 26. 
		This fitting function is accurate to 2% where :math:`\Omega_{\\rm b} h^2 > 0.0125` and 
		:math:`0.025 < \Omega_{\\rm m} h^2 < 0.5`.

		Returns
		-------------------------------------------------------------------------------------------
		s: float
			The sound horizon at recombination in Mpc/h.
		"""
		
		s = 44.5 * np.log(9.83 / self.Omh2) / np.sqrt(1.0 + 10.0 * self.Ombh2**0.75) * self.h
		
		return s

	###############################################################################################
	# Densities and overdensities
	###############################################################################################
	
	def rho_c(self, z):
		"""
		The critical density of the universe at redshift z.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		rho_critical: array_like
			The critical density in units of physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`; has 
			the same dimensions as ``z``.
		"""
			
		return constants.RHO_CRIT_0_KPC3 * self.Ez(z)**2

	###############################################################################################
	
	def rho_m(self, z):
		"""
		The matter density of the universe at redshift z.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		rho_matter: array_like
			The matter density in units of physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`; has the 
			same dimensions as ``z``.
	
		See also
		-------------------------------------------------------------------------------------------
		Om: The matter density of the universe, in units of the critical density.
		"""
			
		return constants.RHO_CRIT_0_KPC3 * self.Om0 * (1.0 + z)**3

	###############################################################################################
	
	def rho_b(self, z):
		"""
		The baryon density of the universe at redshift z.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		rho_baryon: array_like
			The baryon density in units of physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`; has the 
			same dimensions as ``z``.
	
		See also
		-------------------------------------------------------------------------------------------
		Ob: The baryon density of the universe, in units of the critical density.
		"""

		return constants.RHO_CRIT_0_KPC3 * self.Ob0 * (1.0 + z)**3

	###############################################################################################
	
	def rho_de(self, z):
		"""
		The dark energy density of the universe at redshift z.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		rho_de: float
			The dark energy density in units of physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`; has 
			the same dimensions as ``z``.
	
		See also
		-------------------------------------------------------------------------------------------
		Ode: The dark energy density of the universe, in units of the critical density. 
		"""
		
		return constants.RHO_CRIT_0_KPC3 * self.Ode0 * self._rho_de_z(z)

	###############################################################################################
	
	def rho_gamma(self, z):
		"""
		The photon density of the universe at redshift z.
		
		If ``relspecies == False``, this function returns 0.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		rho_gamma: array_like
			The photon density in units of physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`; has the 
			same dimensions as ``z``.
	
		See also
		-------------------------------------------------------------------------------------------
		Ogamma: The density of photons in the universe, in units of the critical density.
		"""
		
		return constants.RHO_CRIT_0_KPC3 * self.Ogamma0 * (1.0 + z)**4

	###############################################################################################
	
	def rho_nu(self, z):
		"""
		The neutrino density of the universe at redshift z.
		
		If ``relspecies == False``, this function returns 0.
		
		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		rho_nu: array_like
			The neutrino density in units of physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`; has 
			the same dimensions as ``z``.
	
		See also
		-------------------------------------------------------------------------------------------
		Onu: The density of neutrinos in the universe, in units of the critical density.
		"""

		return constants.RHO_CRIT_0_KPC3 * self.Onu0 * (1.0 + z)**4

	###############################################################################################
	
	def rho_r(self, z):
		"""
		The density of relativistic species in the universe at redshift z.
		
		This density is the sum of the photon and neutrino densities. If ``relspecies == False``, 
		this function returns 0.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		rho_relativistic: array_like
			The density of relativistic species in units of physical 
			:math:`M_{\odot} h^2 / {\\rm kpc}^3`; has the same dimensions as ``z``.
	
		See also
		-------------------------------------------------------------------------------------------
		Or: The density of relativistic species in the universe, in units of the critical density.
		"""
			
		return constants.RHO_CRIT_0_KPC3 * self.Or0 * (1.0 + z)**4

	###############################################################################################

	def Om(self, z):
		"""
		The matter density of the universe, in units of the critical density.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		Omega_matter: array_like
			Has the same dimensions as ``z``.

		See also
		-------------------------------------------------------------------------------------------
		rho_m: The matter density of the universe at redshift z.
		"""

		return self.Om0 * (1.0 + z)**3 / (self.Ez(z))**2

	###############################################################################################

	def Ob(self, z):
		"""
		The baryon density of the universe, in units of the critical density.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		Omega_baryon: array_like
			Has the same dimensions as ``z``.

		See also
		-------------------------------------------------------------------------------------------
		rho_b: The baryon density of the universe at redshift z.
		"""

		return self.Ob0 * (1.0 + z)**3 / (self.Ez(z))**2

	###############################################################################################

	def Ode(self, z):
		"""
		The dark energy density of the universe, in units of the critical density. 

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		Omega_de: array_like
			Has the same dimensions as ``z``.

		See also
		-------------------------------------------------------------------------------------------
		rho_de: The dark energy density of the universe at redshift z.
		"""

		return self.Ode0 / (self.Ez(z))**2 * self._rho_de_z(z)

	###############################################################################################

	def Ogamma(self, z):
		"""
		The density of photons in the universe, in units of the critical density.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		Omega_gamma: array_like
			Has the same dimensions as ``z``.

		See also
		-------------------------------------------------------------------------------------------
		rho_gamma: The photon density of the universe at redshift z.
		"""
					
		return self.Ogamma0 * (1.0 + z)**4 / (self.Ez(z))**2

	###############################################################################################

	def Onu(self, z):
		"""
		The density of neutrinos in the universe, in units of the critical density.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		Omega_nu: array_like
			Has the same dimensions as ``z``.

		See also
		-------------------------------------------------------------------------------------------
		rho_nu: The neutrino density of the universe at redshift z.
		"""
					
		return self.Onu0 * (1.0 + z)**4 / (self.Ez(z))**2
	
	###############################################################################################

	def Or(self, z):
		"""
		The density of relativistic species, in units of the critical density. 

		This function returns the sum of the densities of photons and neutrinos.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		Omega_relativistic: array_like
			Has the same dimensions as ``z``.

		See also
		-------------------------------------------------------------------------------------------
		rho_r: The density of relativistic species in the universe at redshift z.
		"""
					
		return self.Or0 * (1.0 + z)**4 / (self.Ez(z))**2

	###############################################################################################

	def Ok(self, z):
		"""
		The curvature density of the universe in units of the critical density. 
		
		In a flat universe, :math:`\Omega_{\\rm k} = 0`.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift; can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		Omega_curvature: array_like
			Has the same dimensions as ``z``.
		"""
					
		return self.Ok0 * (1.0 + z)**2 / (self.Ez(z))**2

	###############################################################################################
	# Structure growth, power spectrum etc.
	###############################################################################################

	def growthFactorUnnormalized(self, z):
		"""
		The linear growth factor, :math:`D_+(z)`.
		
		The growth factor describes the linear evolution of over- and underdensities in the dark
		matter density field. There are three regimes: 
		
		* In the matter-radiation regime, we use an approximate analytical formula (Equation 5 in 
		  `Gnedin et al. 2011 <http://adsabs.harvard.edu/abs/2011ApJS..194...46G>`_. If 
		  relativistic species are ignored, :math:`D_+(z) \propto a`. 
		* In the matter-dominated regime, :math:`D_+(z) \propto a`. 
		* In the matter-dark energy regime, we evaluate :math:`D_+(z)` 
		  through integration as defined in 
		  `Eisenstein & Hu 1999 <http://adsabs.harvard.edu/abs/1999ApJ...511....5E>`_, Equation 8 
		  (see also `Heath 1977 <http://adsabs.harvard.edu/abs/1977MNRAS.179..351H>`_) for 
		  LCDM cosmologies. For cosmologies where :math:`w(z) \\neq -1`, this expression is not
		  valid and we instead solve the ordinary differential equation for the evolution of the
		  growth factor (Equation 11 in 
		  `Linder & Jenkins 2003 <https://ui.adsabs.harvard.edu//#abs/2003MNRAS.346..573L/abstract>`_).
		
		At the transition between the integral and analytic approximation regimes, the two 
		expressions do not quite match up, with differences of the order <1E-3. in order to avoid
		a discontinuity, we introduce a transition regime where the two quantities are linearly
		interpolated.
		
		The normalization is such that the growth factor approaches :math:`D_+(a) = a` in the 
		matter-dominated regime. There are other normalizations of the growth factor (e.g., 
		`Percival 2005 <http://adsabs.harvard.edu/abs/2005A%26A...443..819P>`_, Equation 15), but 
		since we almost always care about the growth factor normalized to z = 0, the normalization 
		does not matter too much (see the :func:`growthFactor` function).
		
		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift, where :math:`-0.995 < z`; the high end of z is only limited by the validity 
			of the analytical approximation mentioned above. Can be a number or a numpy array.

		Returns
		-------------------------------------------------------------------------------------------
		D: array_like
			The linear growth factor; has the same dimensions as ``z``.

		See also
		-------------------------------------------------------------------------------------------
		growthFactor: The linear growth factor normalized to z = 0, :math:`D_+(z) / D_+(0)`.

		Warnings
		-------------------------------------------------------------------------------------------
		This function directly evaluates the growth factor by integration or analytical 
		approximation. In most cases, the :func:`growthFactor` function should be used since it 
		interpolates and is thus much faster.
		"""

		# -----------------------------------------------------------------------------------------
		# The growth factor integral uses E(z), but is not designed to take relativistic species
		# into account. Thus, using the standard E(z) leads to wrong results. Instead, we pretend
		# that the small radiation content at low z behaves like a cosmological constant which 
		# leads to a very small error but means that the formula converges to a at high z.
		
		def Ez_D(z):
			ai = (1.0 + z)
			sum = self.Om0 * ai**3 + self.Ode0 * self._rho_de_z(z)
			if self.relspecies:
				sum += self.Or0
			if not self.flat:
				sum += self.Ok0 * ai**2
			E = np.sqrt(sum)
			return E

		# -----------------------------------------------------------------------------------------
		# The integrand for the integral expression for the growth factor
		
		def integrand(z):
			return (1.0 + z) / (Ez_D(z))**3

		# -----------------------------------------------------------------------------------------
		# Compute the growth factor by integrating the ODEs. This is necessary for non-LCDM dark
		# energy where the integral expression is not valid.
	
		def growthFactorFromODE(z_eval):

			# This function implements equation 11 in Linder & Jenkins 2003. 		
			def derivatives_G(a, y):
		
				z = 1.0 / a - 1.0
				G = y[0]
				Gp = y[1]
				
				wa = self.wz(z)
				Xa = self.Om(z) / self.Ode(z)
				
				t1 = -(3.5 - 1.5 * wa / (1.0 + Xa)) / a
				t2 = -1.5 * (1.0 - wa) / (1.0 + Xa) / a**2
				
				return [Gp, t1 * Gp + t2 * G]
		
			# Regardless of what redshifts are requested, we need to start integrating at high z
			# where we know the initial conditions.
			a_eval = 1.0 / (1.0 + z_eval)
			a_min = np.fmin(np.min(a_eval), 1E-3) * 0.99
			a_max = np.max(a_eval) * 1.01
			
			# For the solve_ivp function to work, the evaluation time array needs to be sorted.
			idxs = np.argsort(a_eval)
			a_eval = a_eval[idxs]

			# The stringent accuracy limit is necessary, there are noticeable errors in the solution 
			# for lower atol. The solution should always converge to D ~ a at very low a.
			dic = scipy.integrate.solve_ivp(derivatives_G, (a_min, a_max), [1.0, 0.0], 
										t_eval = a_eval, atol = 1E-6, rtol = 1E-6)
			G = dic['y'][0, :]
			D = G * a_eval
			
			# Revert array to original order
			idxs_reverse = np.zeros_like(idxs)
			idxs_reverse[idxs] = np.arange(0, len(idxs), 1)
			D = D[idxs_reverse]

			return D

		# -----------------------------------------------------------------------------------------

		z_arr, is_array = utilities.getArray(z)
		D = np.zeros_like(z_arr)
		
		# Create a transition regime centered around z = 10 in log space, but only if relativistic
		# species are present.
		if self.relspecies:
			z_switch = 10.0
			trans_width = 2.0
			zt1 = z_switch * trans_width
			zt2 = z_switch / trans_width
			
			# Split into late (1), early (2) and a transition interval (3)
			a = 1.0 / (1.0 + z_arr)
			mask1 = z_arr < (zt1)
			mask2 = z_arr > (zt2)
			mask3 = mask1 & mask2
		else:
			mask1 = np.ones_like(z_arr, np.bool)
			
		# Compute D from integration at low redshift, or integrate ODEs in the case of non-LCDM
		# dark energy.
		z1 = z_arr[mask1]
		if self.de_model == 'lambda':
			D[mask1] = 5.0 / 2.0 * self.Om0 * Ez_D(z1) * self._integral(integrand, z1, np.inf)
		else:
			if np.count_nonzero(mask1) > 0:
				D[mask1] = growthFactorFromODE(z1)

		# Compute D analytically at high redshift.
		if self.relspecies:	
			D1 = D[mask3]
			a2 = a[mask2]
			x = a2 / self.a_eq
			term1 = np.sqrt(1.0 + x)
			term2 = 2.0 * term1 + (2.0 / 3.0 + x) * np.log((term1 - 1.0) / (term1 + 1.0))
			D[mask2] = a2 + 2.0 / 3.0 * self.a_eq + self.a_eq / (2.0 * np.log(2.0) - 3.0) * term2
			D2 = D[mask3]
	
			# Average in transition regime
			at1 = np.log(1.0 / (zt1 + 1.0))
			at2 = np.log(1.0 / (zt2 + 1.0))
			dloga = at2 - at1
			loga = np.log(a[mask3])
			D[mask3] = (D1 * (loga - at1) + D2 * (at2 - loga)) / dloga

		# Reduce array to number if necessary
		if not is_array:
			D = D[0]
		
		return D

	###############################################################################################

	def _growthFactorExact(self, z):
		
		D = self.growthFactorUnnormalized(z) / self.growthFactorUnnormalized(0.0)
		
		return D

	###############################################################################################

	def growthFactor(self, z, derivative = 0, inverse = False):
		"""
		The linear growth factor normalized to z = 0, :math:`D_+(z) / D_+(0)`.

		The growth factor describes the linear evolution of over- and underdensities in the dark
		matter density field. This function is sped up through interpolation which barely degrades 
		its accuracy, but if you wish to evaluate the exact integral or compute the growth factor 
		for very high redshifts (z > 200), please use the :func:`growthFactorUnnormalized`
		function.

		Parameters
		-------------------------------------------------------------------------------------------
		z: array_like
			Redshift, where :math:`-0.995 < z < 200`; can be a number or a numpy array.
		derivative: int
			If greater than 0, evaluate the nth derivative, :math:`d^nD_+/dz^n`.
		inverse: bool
			If True, evaluate :math:`z(D_+)` instead of :math:`D_+(z)`. In this case, the ``z`` 
			field must contain the normalized growth factor.

		Returns
		-------------------------------------------------------------------------------------------
		D: array_like
			The linear growth factor (or its derivative); has the same dimensions as ``z``.

		See also
		-------------------------------------------------------------------------------------------
		growthFactorUnnormalized: The linear growth factor, :math:`D_+(z)`.
		"""

		# The check for z = 0 is worthwhile as this is a common case, and the interpolation can 
		# give a very slight error for D(0), leading to a value slightly different from unity.
		
		if derivative == 0 and np.max(np.abs(z)) < 1E-10:
			D = np.ones_like(z)
		else:
			D = self._zFunction('lnzp1_growthfactor', self._growthFactorExact, z, derivative = derivative,
						inverse = inverse)

		return D

	###############################################################################################

	def _matterPowerSpectrumName(self, model):
		
		return 'ps_%s' % (model)
	
	###############################################################################################

	def _matterPowerSpectrumNormName(self, model):
		
		return 'ps_norm_%s' % (model)
	
	###############################################################################################

	# Utility to get the min and max k for which a power spectrum is valid. Only for internal use.

	def _matterPowerSpectrumLimits(self, model = defaults.POWER_SPECTRUM_MODEL, path = None):
		
		if path is None:
			k_min = self.k_Pk[0]
			k_max = self.k_Pk[-1]
		else:
			table_name = self._matterPowerSpectrumName(model)
			table = self.storageUser.getStoredObject(table_name, path = path)
			if table is None:
				msg = "Could not load data table, %s." % (table_name)
				raise Exception(msg)
			k_min = 10**table[0][0]
			k_max = 10**table[0][-1]
				
		return k_min, k_max
		
	###############################################################################################

	def _matterPowerSpectrumExact(self, k, model = defaults.POWER_SPECTRUM_MODEL, path = None,
								ignore_norm = False):

		if self.power_law:
			
			model = 'powerlaw'
			Pk = k**self.power_law_n
		
		elif model in power_spectrum.models:
			
			T = power_spectrum.transferFunction(k, self.h, self.Om0, self.Ob0, self.Tcmb0, model = model)
			Pk = T * T * k**self.ns

		else:
			
			table_name = self._matterPowerSpectrumName(model)
			table = self.storageUser.getStoredObject(table_name, path = path)
			
			if table is None:
				msg = "Could not load data table, %s." % (table_name)
				raise Exception(msg)
			k_min = 10**table[0][0]
			if np.min(k) < k_min:
				msg = "k (%.2e) is smaller than min. k in table (%.2e)." % (np.min(k), k_min)
				raise Exception(msg)
			k_max = 10**table[0][-1]
			if np.max(k) > k_max:
				msg = "k (%.2e) is larger than max. k in table (%.2e)." % (np.max(k), k_max)
				raise Exception(msg)

			# We do not store the interpolator here, because it might have the wrong normalization.
			interpolator = self.storageUser.getStoredObject(table_name, interpolator = True, 
														store_interpolator = False)
			Pk = 10**interpolator(np.log10(k))
		
		# This is a little tricky. We need to store the normalization factor somewhere, even if 
		# interpolation = False; otherwise, we get into an infinite loop of computing sigma8, P(k), 
		# sigma8 etc.
		if not ignore_norm:
			norm_name = self._matterPowerSpectrumNormName(model)
			norm = self.storageUser.getStoredObject(norm_name)
			if norm is None:
				ps_args = {'model': model, 'path': path}
				sigma_8Mpc = self._sigmaExact(8.0, filt = 'tophat', ps_args = ps_args, 
											exact_ps = True, ignore_norm = True)
				norm = (self.sigma8 / sigma_8Mpc)**2
				self.storageUser.storeObject(norm_name, norm, persistent = False)
			Pk *= norm

		return Pk

	###############################################################################################

	# Return a spline interpolator for the power spectrum. Generally, P(k) should be evaluated 
	# using the matterPowerSpectrum() function below, but for some performance-critical operations
	# it is faster to obtain the interpolator directly from this function. Note that the lookup 
	# table created here is complicated, with extra resolution around the BAO scale.
	#
	# We need to separately treat the cases of models that can cover the entire range of the 
	# Colossus P(k) lookup table, and user-supplied, tabulate 

	def _matterPowerSpectrumInterpolator(self, model = defaults.POWER_SPECTRUM_MODEL, 
										path = None, inverse = False):
		
		# We need to be a little careful in the case of a path being given. It is possible 
		# that the power spectrum from the corresponding table has been evaluated and thus
		# stored in a table, but that this function has not been executed yet, meaning that
		# the power spectrum has not been normalized yet. In that case, we should not call the
		# function that creates the interpolator because it *will* create an unnormalized
		# interpolator.
		table_name = self._matterPowerSpectrumName(model)
		if path is None:
			interpolator = self.storageUser.getStoredObject(table_name,
										interpolator = True, inverse = inverse)
		else:
			interpolator = None
			norm_name = self._matterPowerSpectrumNormName(model)
			norm = self.storageUser.getStoredObject(norm_name)
			if norm == 1.0:
				interpolator = self.storageUser.getStoredObject(table_name,
										interpolator = True, inverse = inverse)
		
		# If we could not find the interpolator, the underlying data table probably has not been
		# created yet.
		if interpolator is None:

			# We are dealing with a non-user supplied power spectrum, meaning we can decide the
			# k array for the table.
			if path is None:
				
				if self.print_info:
					print("Cosmology.matterPowerSpectrum: Computing lookup table.")				
				
				data_k = np.zeros((np.sum(self.k_Pk_Nbins) + 1), np.float)
				n_regions = len(self.k_Pk_Nbins)
				k_computed = 0
				for i in range(n_regions):
					log_min = np.log10(self.k_Pk[i])
					log_max = np.log10(self.k_Pk[i + 1])
					log_range = log_max - log_min
					bin_width = log_range / self.k_Pk_Nbins[i]
					if i == n_regions - 1:
						data_k[k_computed:k_computed + self.k_Pk_Nbins[i] + 1] = \
							10**np.arange(log_min, log_max + bin_width, bin_width)
					else:
						data_k[k_computed:k_computed + self.k_Pk_Nbins[i]] = \
							10**np.arange(log_min, log_max, bin_width)
					k_computed += self.k_Pk_Nbins[i]
				data_Pk = self._matterPowerSpectrumExact(data_k, model = model, ignore_norm = False)
				
				table_ = np.array([np.log10(data_k), np.log10(data_Pk)])
				self.storageUser.storeObject(table_name, table_)
				if self.print_info:
					print("Cosmology.matterPowerSpectrum: Lookup table completed.")	

			else:
				
				# If the interpolator has not been created yet, we need to normalize the table
				# to the correct sigma8 which happens in the exact Pk function.
				table_name = self._matterPowerSpectrumName(model)
				table = self.storageUser.getStoredObject(table_name, path = path)
				table_k = 10**table[0]
				table_P = self._matterPowerSpectrumExact(table_k, model = model, path = path,
														ignore_norm = False)
				table[1] = np.log10(table_P)
				self.storageUser.storeObject(table_name, table, persistent = False)
				
				# We also need to overwrite the norm with one, otherwise the interpolator will not
				# match up with the result of the exact power spectrum function which will find the
				# new, normed interpolator but still apply the old normalization.
				norm_name = self._matterPowerSpectrumNormName(model)
				self.storageUser.storeObject(norm_name, 1.0, persistent = False)

			interpolator = self.storageUser.getStoredObject(table_name, interpolator = True, 
															inverse = inverse)

		return interpolator

	###############################################################################################

	def matterPowerSpectrum(self, k, z = 0.0, model = defaults.POWER_SPECTRUM_MODEL, path = None,
						derivative = False):
		"""
		The matter power spectrum at a scale k.
		
		By default, the power spectrum is computed using a model for the transfer function 
		(see the :func:`~cosmology.power_spectrum.transferFunction` function). The default 
		`Eisenstein & Hu 1998 <http://adsabs.harvard.edu/abs/1998ApJ...496..605E>`_ 
		approximation is accurate to about 5%, and the interpolation introduces errors 
		significantly smaller than that.
		
		Alternatively, the user can supply the path to a file with a tabulated power spectrum 
		using the ``path`` parameter. The file must contain two columns, :math:`\log_{10}(k)`
		and :math:`\log_{10}(P)` where k and P(k) are in the same units as in this function. This
		table is interpolated with a third-order spline. Note that the tabulated spectrum is 
		normalized to the value if :math:`\sigma_8` set in the cosmology.

		Warnings
		-------------------------------------------------------------------------------------------
		If a user-supplied power spectrum table is used, integrals over the power spectrum such 
		as the variance and correlation function are integrated only within the limits of the
		given power spectrum. By default, Boltzmann codes return a relatively small range in 
		wavenumber. Please increase this range if necessary, and check that the computed quantities
		are converged.
		
		Parameters
		-------------------------------------------------------------------------------------------
		k: array_like
			The wavenumber k (in comoving h/Mpc), where :math:`10^{-20} < k < 10^{20}`; can be a 
			number or a numpy array. If a user-supplied table is used, the limits of that table
			apply.
		z: float
			The redshift at which the power spectrum is evaluated, zero by default. If non-zero,
			the power spectrum is scaled with the linear growth factor, 
			:math:`P(k, z) = P(k, 0) D_{+}^2(z)`.
		model: str
			A model for the power spectrum (see the :mod:`cosmology.power_spectrum` module). If a
			tabulated power spectrum is used (see ``path`` parameter), this name must still be 
			passed. Internally, the power spectrum is saved using this name, so the name must not 
			overlap with any other models.
		path: str
			A path to a file containing the power spectrum as a table, where the two columns are
			:math:`\log_{10}(k)` (in comoving h/Mpc) and :math:`\log_{10}(P)` (in 
			:math:`({\\rm Mpc}/h)^3`).
		derivative: bool
			If False, return P(k). If True, return :math:`d \log(P) / d \log(k)`.
			
		Returns
		-------------------------------------------------------------------------------------------
		Pk: array_like
			The matter power spectrum (or its logarithmic derivative if ``derivative == True``); has 
			the same dimensions as k and units of :math:`({\\rm Mpc}/h)^3`.
		"""

		if self.interpolation:
			
			interpolator = self._matterPowerSpectrumInterpolator(model, path)
			
			# If the requested radius is outside the range, give a detailed error message.
			k_req = np.min(k)
			k_min = 10**interpolator.get_knots()[0]
			if k_req < k_min:
				msg = "k = %.2e is too small (min. k = %.2e)" % (k_req, k_min)
				raise Exception(msg)

			k_req = np.max(k)
			k_max = 10**interpolator.get_knots()[-1]
			if k_req > k_max:
				msg = "k = %.2e is too large (max. k = %.2e)" % (k_req, k_max)
				raise Exception(msg)

			if derivative:
				Pk = interpolator(np.log10(k), nu = 1)
			else:
				Pk = interpolator(np.log10(k))
				Pk = 10**Pk
			
		else:
			
			if derivative:
				raise Exception("Derivative can only be evaluated if interpolation == True.")

			if utilities.isArray(k):
				Pk = np.zeros_like(k)
				for i in range(len(k)):
					Pk[i] = self._matterPowerSpectrumExact(k[i], model = model, path = path,
														ignore_norm = False)
			else:
				Pk = self._matterPowerSpectrumExact(k, model = model, path = path, 
												ignore_norm = False)

		Pk *= self.growthFactor(z)**2

		return Pk
	
	###############################################################################################

	def filterFunction(self, filt, k, R):
		"""
		The Fourier transform of certain filter functions.
		
		The main use of the filter function is in computing the variance, please see the 
		documentation of the :func:`sigma` function for details. This function is dimensionless, 
		the input units are k in comoving h/Mpc and R in comoving Mpc/h. Available filters are
		``tophat``,
		
		.. math::
			\\tilde{W}_{\\rm tophat} = \\frac{3}{(kR)^3} \\left[ \\sin(kR) - kR \\times \\cos(kR) \\right] \,,
			
		a ``gaussian`` filter,
		
		.. math::
			\\tilde{W}_{\\rm gaussian} = \\exp \\left[ \\frac{-(kR)^2}{2} \\right] \\,,
			
		and a ``sharp-k`` filter,
		
		.. math::
			\\tilde{W}_{\\rm sharp-k} = \\Theta(1 - kR) \,,	
		
		where :math:`\\Theta` is the Heaviside step function.	

		Parameters
		-------------------------------------------------------------------------------------------
		filt: str
			Either ``tophat`` (a top-hat filter in real space), ``sharp-k`` (a top-hat filter in
			Fourier space), or ``gaussian`` (a Gaussian in both real and Fourier space).
		k: float
			A wavenumber k (in comoving h/Mpc).
		R: float
			A radius R (in comoving Mpc/h).
			
		Returns
		-------------------------------------------------------------------------------------------
		filter: float
			The value of the filter function.
		"""
		
		x = k * R
		
		if filt == 'tophat':
			if x < 1E-3:
				ret = 1.0
			else:
				ret = 3.0 / x**3 * (np.sin(x) - x * np.cos(x))

		elif filt == 'sharp-k':
			ret = np.heaviside(1.0 - x, 1.0)
			
		elif filt == 'gaussian':
			ret = np.exp(-x**2 * 0.5)
		
		else:
			msg = "Invalid filter, %s." % (filt)
			raise Exception(msg)
			
		return ret

	###############################################################################################

	def _sigmaExact(self, R, j = 0, filt = 'tophat', exact_ps = False, ignore_norm = False, 
				ps_args = defaults.PS_ARGS):

		# -----------------------------------------------------------------------------------------
		def logIntegrand(lnk, ps_interpolator):
			
			k = np.exp(lnk)
			W = self.filterFunction(filt, k, R)
			
			if ps_interpolator is not None:
				Pk = 10**ps_interpolator(np.log10(k))
			else:
				Pk = self._matterPowerSpectrumExact(k, ignore_norm = ignore_norm, **ps_args)
			
			# One factor of k is due to the integration in log-k space
			ret = Pk * W**2 * k**3
			
			# Higher moment terms
			if j > 0:
				ret *= k**(2 * j)
			
			return ret

		# -----------------------------------------------------------------------------------------
		
		if filt == 'tophat' and j > 0:
			raise Exception('Higher-order moments of sigma are not well-defined for tophat filter. Choose filter "gaussian" instead.')
	
		# For power-law cosmologies, we can evaluate sigma analytically. The exact expression 
		# has a dependence on n that in turn depends on the filter used, but the dependence 
		# on radius is simple and independent of the filter. Thus, we use sigma8 to normalize
		# sigma directly. 
		if self.power_law:
			
			n = self.power_law_n + 2 * j
			if n <= -3.0:
				raise Exception('n + 2j must be > -3 for the variance to converge in a power-law cosmology.')
			sigma2 = R**(-3 - n) / (8.0**(-3 - n) / self.sigma8**2)
			sigma = np.sqrt(sigma2)
			
		else:
			
			# If we are getting P(k) from a look-up table, it is a little more efficient to 
			# get the interpolator object and use it directly, rather than using the P(k) function.
			ps_interpolator = None
			if (not exact_ps) and self.interpolation:
				ps_interpolator = self._matterPowerSpectrumInterpolator(**ps_args)
			
			# The infinite integral over k often causes trouble when the tophat filter is used. Thus,
			# we determine sensible limits and integrate over a finite k-volume. The limits are
			# determined by demanding that the integrand is some factor, 1E-6, smaller than at its
			# maximum. For tabulated power spectra, we need to be careful not to exceed their 
			# limits, even if the integrand has not reached the desired low value. Thus, we simply
			# use the limits of the table.
			test_k_min, test_k_max = self._matterPowerSpectrumLimits(**ps_args)

			if ('path' in ps_args) and (ps_args['path'] is not None):

				min_k_use = np.log(test_k_min * 1.0001)
				max_k_use = np.log(test_k_max * 0.9999)

			else:

				test_integrand_min = 1E-6

				test_k_min = max(test_k_min * 1.0001, 1E-7)
				test_k_max = min(test_k_max * 0.9999, 1E15)
				test_k = np.arange(np.log(test_k_min), np.log(test_k_max), 2.0)
				n_test = len(test_k)
				test_k_integrand = test_k * 0.0
				for i in range(n_test):
					test_k_integrand[i] = logIntegrand(test_k[i], ps_interpolator)
				integrand_max = np.max(test_k_integrand)
			
				min_index = 0
				while test_k_integrand[min_index] < integrand_max * test_integrand_min:
					min_index += 1
					if min_index > n_test - 2:
						msg = "Could not find lower integration limit."
						raise Exception(msg)
				min_k_use = test_k[min_index]
				
				min_index -= 1
				max_index = min_index + 1
				while test_k_integrand[max_index] > integrand_max * test_integrand_min:
					max_index += 1	
					if max_index == n_test:
						msg = "Could not find upper integration limit."
						raise Exception(msg)
				max_k_use = test_k[max_index]
						
			args = ps_interpolator
			sigma2, _ = scipy.integrate.quad(logIntegrand, min_k_use, max_k_use,
						args = args, epsabs = 0.0, epsrel = self.accuracy_sigma, limit = 100)
			sigma = np.sqrt(sigma2 / 2.0 / np.pi**2)
		
		if np.isnan(sigma):
			msg = "Result is nan (cosmology %s, filter %s, R %.2e, j %d." % (self.name, filt, R, j)
			raise Exception(msg)
			
		return sigma
	
	###############################################################################################

	# Return a spline interpolator for sigma(R) or R(sigma) if inverse == True. Generally, sigma(R) 
	# should be evaluated using the sigma() function below, but for some performance-critical 
	# operations it is faster to obtain the interpolator directly from this function.If the lookup-
	# table does not exist yet, create it. For sigma, we use a very particular binning scheme. At 
	# low R, sigma is a very smooth function, and very wellapproximated by a spline interpolation 
	# between few points. Around the BAO scale, we need a higher resolution. Thus, the bins are 
	# assigned in reverse log(log) space.

	def _sigmaInterpolator(self, j, filt, inverse, ps_args):
		
		table_name = 'sigma%d_%s_%s_%s' % (j, self.name, ps_args['model'], filt)
		interpolator = self.storageUser.getStoredObject(table_name, interpolator = True, 
													inverse = inverse)
		
		if interpolator is None:
			if self.print_info:
				print("Cosmology.sigma: Computing lookup table.")
			max_log = np.log10(self.R_max_sigma)
			log_range = max_log - np.log10(self.R_min_sigma)
			max_loglog = np.log10(log_range + 1.0)
			loglog_width = max_loglog / self.R_Nbins_sigma
			R_loglog = np.arange(0.0, max_loglog + loglog_width, loglog_width)
			log_R = max_log - 10**R_loglog[::-1] + 1.0
			data_R = 10**log_R
			data_sigma = data_R * 0.0
			for i in range(len(data_R)):
				data_sigma[i] = self._sigmaExact(data_R[i], j = j, filt = filt, 
												ps_args = ps_args)
			table_ = np.array([np.log10(data_R), np.log10(data_sigma)])
			self.storageUser.storeObject(table_name, table_)
			if self.print_info:
				print("Cosmology.sigma: Lookup table completed.")

			interpolator = self.storageUser.getStoredObject(table_name, interpolator = True, inverse = inverse)
	
		return interpolator

	###############################################################################################
	
	def sigma(self, R, z, j = 0, filt = 'tophat', inverse = False, derivative = False, 
							ps_args = defaults.PS_ARGS):
		"""
		The rms variance of the linear density field on a scale R, :math:`\\sigma(R)`.
		
		The variance and its higher moments are defined as the integral
		
		.. math::
			\\sigma^2(R,z) = \\frac{1}{2 \\pi^2} \\int_0^{\\infty} k^2 k^{2j} P(k,z) |\\tilde{W}(kR)|^2 dk

		where :math:`\\tilde{W}(kR)` is the Fourier transform of the :func:`filterFunction`, and 
		:math:`P(k,z) = D_+^2(z)P(k,0)` is the :func:`matterPowerSpectrum`. See the documentation
		of :func:`filterFunction` for possible filters.
		
		By default, the power spectrum is computed using the transfer function approximation of 
		`Eisenstein & Hu 1998 <http://adsabs.harvard.edu/abs/1998ApJ...496..605E>`_ (see the 
		:mod:`cosmology.power_spectrum` module). With this approximation, the variance is accurate
		to about 2% or better (see the Colossus code paper for details). Using a tabulated power 
		spectrum can make this computation more accurate, but please note that the limits of the 
		corresponding table are used for the integration.
		
		Higher moments of the variance (such as :math:`\sigma_1`, :math:`\sigma_2` etc) can be 
		computed by setting j > 0 (see 
		`Bardeen et al. 1986 <http://adsabs.harvard.edu/abs/1986ApJ...304...15B>`_). Furthermore, 
		the logarithmic derivative :math:`d \log(\sigma) / d \log(R)` can be evaluated by setting 
		``derivative == True``.
		
		Parameters
		-------------------------------------------------------------------------------------------
		R: array_like
			The radius of the filter in comoving Mpc/h, where :math:`10^{-12} < R < 10^3`; can be 
			a number or a numpy array.
		z: float
			Redshift; for z > 0, :math:`\sigma(R)` is multiplied by the linear growth factor.
		j: integer
			The order of the integral. j = 0 corresponds to the variance, j = 1 to the same integral 
			with an extra :math:`k^2` term etc; see 
			`Bardeen et al. 1986 <http://adsabs.harvard.edu/abs/1986ApJ...304...15B>`_ for 
			mathematical details.
		filt: str
			Either ``tophat``, ``sharp-k`` or ``gaussian`` (see :func:`filterFunction`). Higher 
			moments (j > 0) can only be computed for the gaussian filter.
		inverse: bool
			If True, compute :math:`R(\sigma)` rather than :math:`\sigma(R)`.
		derivative: bool
			If True, return the logarithmic derivative, :math:`d \log(\sigma) / d \log(R)`, or its
			inverse, :math:`d \log(R) / d \log(\sigma)` if ``inverse == True``.
		ps_args: dict
			Arguments passed to the :func:`matterPowerSpectrum` function.
		
		Returns
		-------------------------------------------------------------------------------------------
		sigma: array_like
			The rms variance; has the same dimensions as ``R``. If inverse and/or derivative are 
			True, the inverse, derivative, or derivative of the inverse are returned. If j > 0, 
			those refer to higher moments.

		See also
		-------------------------------------------------------------------------------------------
		matterPowerSpectrum: The matter power spectrum at a scale k.
		"""

		if self.interpolation:
			interpolator = self._sigmaInterpolator(j, filt, inverse, ps_args)
			
			if not inverse:
	
				# If the requested radius is outside the range, give a detailed error message.
				R_req = np.min(R)
				if R_req < self.R_min_sigma:
					M_min = 4.0 / 3.0 * np.pi * self.R_min_sigma**3 * self.rho_m(0.0) * 1E9
					msg = "R = %.2e is too small (min. R = %.2e, min. M = %.2e)" \
						% (R_req, self.R_min_sigma, M_min)
					raise Exception(msg)
			
				R_req = np.max(R)
				if R_req > self.R_max_sigma:
					M_max = 4.0 / 3.0 * np.pi * self.R_max_sigma**3 * self.rho_m(0.0) * 1E9
					msg = "R = %.2e is too large (max. R = %.2e, max. M = %.2e)" \
						% (R_req, self.R_max_sigma, M_max)
					raise Exception(msg)
	
				if derivative:
					ret = interpolator(np.log10(R), nu = 1)
				else:
					ret = interpolator(np.log10(R))
					ret = 10**ret
					ret *= self.growthFactor(z)

			else:
				
				sigma_ = R / self.growthFactor(z)

				# Get the limits in sigma from storage, or compute and store them. Using the 
				# storage mechanism seems like overkill, but these numbers should be erased if 
				# the cosmology changes and sigma is re-computed.
				sigma_min = self.storageUser.getStoredObject('sigma_min')
				sigma_max = self.storageUser.getStoredObject('sigma_max')
				if sigma_min is None or sigma_min is None:
					knots = interpolator.get_knots()
					sigma_min = 10**np.min(knots)
					sigma_max = 10**np.max(knots)
					self.storageUser.storeObject('sigma_min', sigma_min, persistent = False)
					self.storageUser.storeObject('sigma_max', sigma_max, persistent = False)
				
				# If the requested sigma is outside the range, give a detailed error message.
				sigma_req = np.max(sigma_)
				if sigma_req > sigma_max:
					msg = "sigma = %.2e is too large (max. sigma = %.2e)" % (sigma_req, sigma_max)
					raise Exception(msg)
					
				sigma_req = np.min(sigma_)
				if sigma_req < sigma_min:
					msg = "sigma = %.2e is too small (min. sigma = %.2e)" % (sigma_req, sigma_min)
					raise Exception(msg)
				
				# Interpolate to get R(sigma)
				if derivative: 
					ret = interpolator(np.log10(sigma_), nu = 1)					
				else:
					ret = interpolator(np.log10(sigma_))
					ret = 10**ret

		else:
			
			if inverse:
				raise Exception('R(sigma)  cannot be evaluated with interpolation == False.')
			if derivative:
				raise Exception('Derivative of sigma cannot be evaluated if interpolation == False.')

			if utilities.isArray(R):
				ret = R * 0.0
				for i in range(len(R)):
					ret[i] = self._sigmaExact(R[i], j = j, filt = filt, ps_args = ps_args)
			else:
				ret = self._sigmaExact(R, j = j, filt = filt, ps_args = ps_args)
			ret *= self.growthFactor(z)
		
		return ret

	###############################################################################################

	def _correlationFunctionExact(self, R, ps_args = defaults.PS_ARGS):
		
		f_cut = 0.001

		# -----------------------------------------------------------------------------------------
		# The integrand is exponentially cut off at a scale 1000 * R.
		def integrand(k, R, ps_args, ps_interpolator):
			
			if self.interpolation:
				Pk = 10**ps_interpolator(np.log10(k))
			else:
				Pk = self._matterPowerSpectrumExact(k, **ps_args)
			
			ret = Pk * k / R * np.exp(-(k * R * f_cut)**2)
			
			return ret

		# -----------------------------------------------------------------------------------------
		# If we are getting P(k) from a look-up table, it is a little more efficient to 
		# get the interpolator object and use it directly, rather than using the P(k) function.
		ps_interpolator = None
		if self.interpolation:
			ps_interpolator = self._matterPowerSpectrumInterpolator(**ps_args)

		# Determine the integration limits. The limits chosen here correspond to the cut-off scale
		# introduced in the integrator.
		k_min = 1E-6 / R
		k_max = 10.0 / f_cut / R

		# If we are using a tabulated power spectrum, we just use the limits of that table IF they 
		# are more stringent than those already determined.
		k_min_model, k_max_model = self._matterPowerSpectrumLimits(**ps_args)
		k_min = max(k_min, k_min_model * 1.0001)
		k_max = min(k_max, k_max_model * 0.9999)

		# Use a Clenshaw-Curtis integration, i.e. an integral weighted by sin(kR). 
		args = R, ps_args, ps_interpolator
		xi, _ = scipy.integrate.quad(integrand, k_min, k_max, args = args, epsabs = 0.0,
					epsrel = self.accuracy_xi, limit = 100, weight = 'sin', wvar = R)
		xi /= 2.0 * np.pi**2

		if np.isnan(xi):
			msg = 'Result is nan (cosmology %s, R %.2e).' % (self.name, R)
			raise Exception(msg)

		return xi
	
	###############################################################################################

	# Return a spline interpolator for the correlation function, xi(R). Generally, xi(R) should be 
	# evaluated using the correlationFunction() function below, but for some performance-critical 
	# operations it is faster to obtain the interpolator directly from this function.

	def _correlationFunctionInterpolator(self, ps_args):

		table_name = 'correlation_%s_%s' % (self.name, ps_args['model'])
		interpolator = self.storageUser.getStoredObject(table_name, interpolator = True)
		
		if interpolator is None:
			if self.print_info:
				print("correlationFunction: Computing lookup table. This may take a few minutes, please do not interrupt.")
			
			data_R = np.zeros((np.sum(self.R_xi_Nbins) + 1), np.float)
			n_regions = len(self.R_xi_Nbins)
			k_computed = 0
			for i in range(n_regions):
				log_min = np.log10(self.R_xi[i])
				log_max = np.log10(self.R_xi[i + 1])
				log_range = log_max - log_min
				bin_width = log_range / self.R_xi_Nbins[i]
				if i == n_regions - 1:
					data_R[k_computed:k_computed + self.R_xi_Nbins[i] + 1] = \
						10**np.arange(log_min, log_max + bin_width, bin_width)
				else:
					data_R[k_computed:k_computed + self.R_xi_Nbins[i]] = \
						10**np.arange(log_min, log_max, bin_width)
				k_computed += self.R_xi_Nbins[i]
			
			data_xi = data_R * 0.0
			for i in range(len(data_R)):
				data_xi[i] = self._correlationFunctionExact(data_R[i], ps_args = ps_args)
			table_ = np.array([data_R, data_xi])
			self.storageUser.storeObject(table_name, table_)
			if self.print_info:
				print("correlationFunction: Lookup table completed.")
			interpolator = self.storageUser.getStoredObject(table_name, interpolator = True)
		
		return interpolator

	###############################################################################################

	def correlationFunction(self, R, z = 0.0, derivative = False, ps_args = defaults.PS_ARGS):
		"""
		The linear matter-matter correlation function at radius R.
		
		The linear correlation function is defined as 
		
		.. math::
			\\xi(R,z) = \\frac{1}{2 \\pi^2} \\int_0^\\infty k^2 P(k,z) \\frac{\\sin(kR)}{kR} dk
		
		where P(k) is the :func:`matterPowerSpectrum`. By default, the power spectrum is computed 
		using the transfer function approximation of 
		`Eisenstein & Hu 1998 <http://adsabs.harvard.edu/abs/1998ApJ...496..605E>`_ (see the 
		:mod:`cosmology.power_spectrum` module). With this approximation, the correlation function
		is accurate to ~5% over the range :math:`10^{-2} < R < 200` (see the Colossus code paper 
		for details). Using a tabulated power spectrum can make this computation more accurate, 
		but please note that the limits of the corresponding table are used for the integration. 
		
		Parameters
		-------------------------------------------------------------------------------------------
		R: array_like
			The radius in comoving Mpc/h; can be a number or a numpy array.
		z: float
			Redshift; if non-zero, the correlation function is scaled with the linear growth 
			factor, :math:`\\xi(R, z) = \\xi(R, 0) D_{+}^2(z)`.
		derivative: bool
			If ``derivative == True``, the linear derivative :math:`d \\xi / d R` is returned.
		ps_args: dict
			Arguments passed to the :func:`matterPowerSpectrum` 
			function.

		Returns
		-------------------------------------------------------------------------------------------
		xi: array_like
			The correlation function, or its derivative; has the same dimensions as ``R``.

		See also
		-------------------------------------------------------------------------------------------
		matterPowerSpectrum: The matter power spectrum at a scale k.
		"""
	
		if self.interpolation:
			
			# Load lookup-table
			interpolator = self._correlationFunctionInterpolator(ps_args)
				
			# If the requested radius is outside the range, give a detailed error message.
			R_req = np.min(R)
			if R_req < self.R_xi[0]:
				msg = 'R = %.2e is too small (min. R = %.2e)' % (R_req, self.R_xi[0])
				raise Exception(msg)
		
			R_req = np.max(R)
			if R_req > self.R_xi[-1]:
				msg = 'R = %.2e is too large (max. R = %.2e)' % (R_req, self.R_xi[-1])
				raise Exception(msg)
	
			# Interpolate to get xi(R). Note that the interpolation is performed in linear 
			# space, since xi can be negative.
			if derivative:
				ret = interpolator(R, nu = 1)
			else:
				ret = interpolator(R)
			
		else:

			if derivative:
				raise Exception('Derivative of xi cannot be evaluated if interpolation == False.')

			if utilities.isArray(R):
				ret = R * 0.0
				for i in range(len(R)):
					ret[i] = self._correlationFunctionExact(R[i], ps_args = ps_args)
			else:
				ret = self._correlationFunctionExact(R, ps_args = ps_args)

		if not derivative:
			ret *= self.growthFactor(z)**2

		return	ret

###################################################################################################
# Setter / getter functions for cosmologies
###################################################################################################

def setCosmology(cosmo_name, params = None):
	"""
	Set a cosmology.
	
	This function provides a convenient way to create a cosmology object without setting the 
	parameters of the :class:`Cosmology` class manually. See the Basic Usage section for examples.
	Whichever way the cosmology is set, the global variable is updated so that the :func:`getCurrent` 
	function returns the set cosmology.

	Parameters
	-----------------------------------------------------------------------------------------------
	cosmo_name: str
		The name of the cosmology. Can be the name of a pre-set cosmology, or another name in which
		case the ``params`` dictionary needs to be provided.
	params: dictionary
		The parameters of the constructor of the :class:`Cosmology` class. Not necessary if
		``cosmo_name`` is the name of a pre-set cosmology.

	Returns
	-----------------------------------------------------------------------------------------------
	cosmo: Cosmology
		The created cosmology object.
	"""
	
	if 'powerlaw_' in cosmo_name:
		n = float(cosmo_name.split('_')[1])
		param_dict = cosmologies['powerlaw'].copy()
		param_dict['power_law'] = True
		param_dict['power_law_n'] = n
		if params is not None:
			param_dict.update(params)

	elif cosmo_name == 'powerlaw':
		raise Exception('Power-law cosmology must be called with slope n, e.g. powerlaw_-1.5.')
		
	elif cosmo_name in cosmologies:		
		param_dict = cosmologies[cosmo_name].copy()
		if params is not None:
			param_dict.update(params)
			
	else:
		if params is not None:
			param_dict = params.copy()
		else:
			msg = "Invalid cosmology (%s)." % (cosmo_name)
			raise Exception(msg)
		
	param_dict['name'] = cosmo_name
	cosmo = Cosmology(**(param_dict))
	setCurrent(cosmo)
	
	return cosmo

###################################################################################################

def addCosmology(cosmo_name, params):
	"""
	Add a set of cosmological parameters to the global list.
	
	After this function is executed, the new cosmology can be set using :func:`setCosmology` from 
	anywhere in the code.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	cosmo_name: str
		The name of the cosmology.
	params: dictionary
		A set of parameters for the constructor of the Cosmology class.
	"""
	
	cosmologies[cosmo_name] = params
	
	return 

###################################################################################################

def setCurrent(cosmo):
	"""
	Set the current global cosmology to a cosmology object.
	
	Unlike :func:`setCosmology`, this function does not create a new cosmology object, but allows 
	the user to set a cosmology object to be the current cosmology. This can be useful when switching
	between cosmologies, since many routines use the :func:`getCurrent` routine to obtain the current
	cosmology.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	cosmo: Cosmology
		The cosmology object to be set as the global current cosmology.
	"""

	global current_cosmo
	current_cosmo = cosmo
	
	return

###################################################################################################

def getCurrent():
	"""
	Get the current global cosmology.
	
	This function should be used whenever access to the cosmology is needed. By using the globally
	set cosmology, there is no need to pass cosmology objects around the code. If no cosmology is
	set, this function raises an Exception that reminds the user to set a cosmology.

	Returns
	-----------------------------------------------------------------------------------------------
	cosmo: Cosmology
		The current globally set cosmology. 
	"""
	
	if current_cosmo is None:
		raise Exception('Cosmology is not set.')

	return current_cosmo

###################################################################################################
