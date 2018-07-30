===========
What's new?
===========

See below for a listing of the most important code and interface changes in Colossus, starting with version 1.1.0.

.. rubric:: Version 1.2.1

* The halo concentration model of Diemer and Joyce 2018 was added, and the Diemer and Kravtsov 2015 model was updated according to Diemer and Joyce 2018.
* Numerous small improvements were made in the documentation.
* The function ``plotChain`` was removed from the :doc:`utils_mcmc` module to avoid including the ``matplotlib`` library. The function is still available as part of the `MCMC tutorial <_static/tutorial_utils_mcmc.html>`_.
* The Planck 2018 cosmology was added (and can be used by setting ``planck18`` or ``planck18-only`` for the cosmology).
* Bug fix: the growth factor was incorrect for :math:`w \neq -1` cosmologies, an error that has been rectified in this release (thanks to Lehman Garrison for catching this bug).
* Bug fix: the ``ps_args`` parameter was not used in the :func:`lss.peaks.massFromPeakHeight` and :func:`lss.peaks.peakCurvature` functions (thanks to Michael Joyce for catching this bug).
* The ``inverse`` option was removed from the :func:`cosmology.cosmology.angularDiameterDistance` function because the inverse is multi-valued and leads to an error.

.. rubric:: Version 1.2.0

Version 1.2.0 is the version that coincided with the first publication of the code paper on arXiv.org. The following major changes were made:

* The documentation was reworked entirely.
* All functions and parameters that were deprecated in 1.1.0 have been removed from the code (rather than outputting warnings).
* The ``qx`` and ``qy`` parameters in the :mod:`halo.splashback` module were renamed to ``q_in`` and ``q_out`` to conform with the rest of the code. A number of other small inconsistencies in splashback radius interface were fixed.

.. rubric:: Version 1.1.0

Version 1.1.0 presents a major change to the Colossus interface, documentation, and tutorial system. The most important changes are that

* A new top-level module for large-scale structure, LSS, has been added, including functions previously housed in the cosmology module, the old halo bias module, and a new module for the halo mass function. The LSS module covers funtions that deal with peaks or halos as a statistical ensemble so that the cosmology module does no longer "know" anything about halos. Conversely, the halo module covers functions that apply to individual halos.
* The demo scripts have been converted to much more extensive Jupyter notebook :doc:`tutorials`. 
* A number of interfaces have been made more homogeneous.
* Wherever possible, deprecated function interfaces are still present for backward compatibility but issue a warning. These functions and parameters will be removed in the next version.
* This documentation has been reorganized and improved, and its location has shifted to https://bdiemer.bitbucket.io/colossus.

The following functions are now housed in the LSS module:

* Cosmology.lagrangianR() is now :func:`lss.peaks.lagrangianR`
* Cosmology.lagrangianM() is now :func:`lss.peaks.lagrangianM`
* Cosmology.collapseOverdensity() is now :func:`lss.peaks.collapseOverdensity`
* Cosmology.peakHeight() is now :func:`lss.peaks.peakHeight`
* Cosmology.massFromPeakHeight() is now :func:`lss.peaks.massFromPeakHeight`
* Cosmology.nonLinearMass() is now :func:`lss.peaks.nonLinearMass`
* Cosmology.peakCurvature() is now :func:`lss.peaks.peakCurvature`
* The module halo.bias is now :mod:`lss.bias`.
* The LSS module contains a brand new module to compute the halo mass function, :mod:`lss.mass_function`.
  
The following changes apply to interfaces across modules:

* Any module that implements models (e.g., fitting functions for concentration), now features an ordered dictionary called ``models`` that contains class objects with the properties of the respective models (which vary from module to module). This change affects the power spectrum, bias, halo mass function, concentration, and splashback modules. These new model dictionaries replace the previous ``MODELS`` lists that were present in some of the modules.
* There is a new storage module as part of utilities. The storage parameter in the cosmology module was renamed to persistence, as was the global setting ``STORAGE`` (renamed to ``PERSISTENCE``). The storage module can now be used by other modules or from outside of Colossus.

Changes in the cosmology module:

* Cosmology now allows for a non-constant dark energy equations of state. The implemented dark energy models include a fixed or varying equation of state (see :class:`~cosmology.cosmology.Cosmology` class for more information). As a result, the OL0, OL(), and rho_L() parameters and functions were renamed to ``Ode0``, ``Ode()``, and ``rho_de()``.
* The power spectrum models were extracted into a separate module, :mod:`cosmology.power_spectrum`. The names of the available models were changed from ``eh98`` to ``eisenstein98`` and from ``eh98_smooth`` to ``eisenstein98_zb`` to conform with other Colossus modules.
* The ``Pk_source`` parameter was renamed to ``model`` in the :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` function. In functions that call the power spectrum, the user can pass a ``ps_args`` dictionary containing kwargs that are passed to the power spectrum function.
* The :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` function now takes redshift as an optional parameter.
* The ``text_output`` option was removed from the cosmology object.
* The :func:`~cosmology.cosmology.Cosmology.soundHorizon()` function now returns the sound horizon in Mpc/h rather than Mpc in order to be consistent with the rest of the cosmology module.

Changes in the LSS module:

* The :func:`~lss.peaks.collapseOverdensity()` function has been completely reworked. By default, it still returns the constant collapse overdensity threshold in an Einstein-de Sitter universe. If a redshift is passed, it applies small corrections based on the underlying cosmology. The previous parameters to this function will now cause an error. This change also affects all functions that rely on the collapse overdensity, such as :func:`~lss.peaks.peakHeight()`, :func:`~lss.peaks.massFromPeakHeight()`, :func:`~lss.peaks.nonLinearMass()`, and :func:`~lss.peaks.peakCurvature()`. These functions now accept dictionaries of parameters that are passed to the collapse overdensity and :func:`~cosmology.cosmology.Cosmology.sigma` functions.
* The halo bias module was extended with two new models for halo bias.
* The input units to the :func:`~lss.bias.twoHaloTerm` function are now in comoving Mpc/h rather than physical kpc/h in order to conform to the unit system of the LSS module.

Changes in the halo module: 

* The interface of the SO changing functions in :mod:`halo.mass_defs` has changed. The function previously called pseudoEvolve is now called :func:`~halo.mass_defs.evolveSO` to reflect its more general nature. The :func:`~halo.mass_defs.pseudoEvolve` function is a wrapper for evolveSO, and has one fewer parameter than previously (no final mass definition).
* The :class:`~halo.profile_dk14.DK14Profile` constructor does not take R200m as an input any more and instead computes it self-consistently regardless of what the other inputs are. In this new version, the redshift always needs to be passed to the constructor. These changes fix a bug with outer profiles that themselves rely on R200m as an input. Furthermore, the normalization of power-law outer profiles is no longer adjusted in order to maintain a constant amplitude of R200m changes. It is up to the user to ensure that the behavior of the outer profile makes sense physically.
* The ``klypin14_nu`` and ``klypin14_m`` concentration models were renamed to ``klypin16_nu`` and ``klypin16_m`` to maintain compatibility with the publication date of their paper.
  