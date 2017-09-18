===================================================================================================
What's new?
===================================================================================================

See below for a listing of the most important code and interface changes in Colossus, starting 
with version 1.1.

***************************************************************************************************
Version 1.1
***************************************************************************************************

* A separate module for large-scale structure, lss, has been added. Some functions that were 
  previously located in the cosmology and halo modules were shifted into this new module, namely:

    * Cosmology.lagrangianR() is now :func:`lss.lss.lagrangianR`
    * Cosmology.lagrangianM() is now :func:`lss.lss.lagrangianM`
    * Cosmology.collapseOverdensity() is now :func:`lss.lss.collapseOverdensity`
    * Cosmology.peakHeight() is now :func:`lss.lss.peakHeight`
    * Cosmology.massFromPeakHeight() is now :func:`lss.lss.massFromPeakHeight`
    * Cosmology.nonLinearMass() is now :func:`lss.lss.nonLinearMass`
    * Cosmology.peakCurvature() is now :func:`lss.lss.peakCurvature`
    * halo.bias is now :mod:`lss.bias`

* The lss module contains a new module to compute the halo mass function.
* The demo scripts have been converted to Jupyter notebooks
* The interface of the SO changing functions in :mod:`halo.mass_defs` has changed. The function
  previously called pseudoEvolve is now called :func:`halo.mass_defs.evolveSO` to reflect its more
  general nature. The :func:`halo.mass_defs.pseudoEvolve` function is a wrapper for evolveSO, and
  has one fewer parameter than previously (no final mass definition).
* Some modules contain a MODELS dictionary or list naming all implemented fitting functions. For
  consistency, all occurrences of MODELS have been renamed to "models", affecting the concentration,
  splashback, and bias modules.