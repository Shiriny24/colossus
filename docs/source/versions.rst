===================================================================================================
What's new?
===================================================================================================

See below for a listing of the most important code and interface changes in Colossus, starting 
with version 1.1.

***************************************************************************************************
Version 1.1
***************************************************************************************************

A separate module for large-scale structure, lss, has been added, including functions previously
housed in the cosmology module, the halo bias module, and a new module for the halo mass function.
The following functions were shifted from the cosmology module into the lss module:

* Cosmology.lagrangianR() is now :func:`lss.lss.lagrangianR`
* Cosmology.lagrangianM() is now :func:`lss.lss.lagrangianM`
* Cosmology.collapseOverdensity() is now :func:`lss.lss.collapseOverdensity`
* Cosmology.peakHeight() is now :func:`lss.lss.peakHeight`
* Cosmology.massFromPeakHeight() is now :func:`lss.lss.massFromPeakHeight`
* Cosmology.nonLinearMass() is now :func:`lss.lss.nonLinearMass`
* Cosmology.peakCurvature() is now :func:`lss.lss.peakCurvature`

The new LSS module has led to the following other changes:

* The module halo.bias is now :mod:`lss.bias`
* The :func:`lss.lss.collapseOverdensity()` function has been completely reworked. By default, it 
  still returns the constant collapse overdensity threshold in an Einstein-de Sitter universe. If a 
  redshift is passed, it applies small corrections based on the underlying cosmology. The previous 
  parameters to this function will now cause an error. This change also affects all functions that
  rely on the collapse overdensity, such as :func:`lss.lss.peakHeight()`, 
  :func:`lss.lss.massFromPeakHeight()`, :func:`lss.lss.nonLinearMass()`, and 
  :func:`lss.lss.peakCurvature()`. These functions now take dictionaries of parameters that are 
  passed to the collapse overdensity and sigma functions.
* The halo bias module was improved, with two new models for halo bais (spherical collapse and
  Sheth et al. 2001).
* The lss module contains a new module to compute the halo mass function.

Changes in the cosmology module:

* The power spectrum models were extracted into a separate module, :mod:`cosmology.power_spectrum`.
  The names of the available models were changed from ``eh98`` to ``eisenstein98`` and from 
  ``eh98_smooth`` to ``eisenstein98_zb`` to conform with other colossus modules.
* The ``Pk_source`` parameter was renamed to ``model`` in the :func:`cosmology.cosmology.Cosmology.matterPowerSpectrum`
  function, and to ``ps_model`` in all other functions that rely on the power spectrum.
* The :func:`cosmology.cosmology.Cosmology.matterPowerSpectrum` function now takes redshift as
  an optional parameter.
* Cosmology now allows non-constant dark energy equations of state. 
* The OL0, OL(), and rho_L() parameters and functions were renamed to Ode0, Ode(), and rho_de().
* The text_output option was removed from the cosmology object.

Changes in the halo module: 

* The interface of the SO changing functions in :mod:`halo.mass_defs` has changed. The function
  previously called pseudoEvolve is now called :func:`halo.mass_defs.evolveSO` to reflect its more
  general nature. The :func:`halo.mass_defs.pseudoEvolve` function is a wrapper for evolveSO, and
  has one fewer parameter than previously (no final mass definition).
* Some modules contain a MODELS dictionary or list naming all implemented fitting functions. For
  consistency, all occurrences of MODELS have been renamed to "models", affecting the concentration,
  splashback, and bias modules.
* The klypin14_nu and klypin14_m concentration models were renamed to klypin16_nu and klypin16_m
  to maintain compatibility with the publication of their paper.

Other changes:

* The demo scripts have been converted to Jupyter notebooks
* There is a new storage_unit module as part of utilities. The storage parameter in the cosmology
  module was renamed to persistence, as was the global setting STORAGE (renamed to PERSISTENCE).
