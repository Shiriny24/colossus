===========
What's new?
===========

See below for a listing of the most important code and interface changes in Colossus, starting with
version 1.1.0.

.. rubric:: Version 1.2.14

Changes in this version include:

* The user can now pass power spectrum arguments to the Diemer & Joyce 2019 concentration model,
  for example, in order to use a non-standard power spectrum.
* The code returns more informative error messages when tabulated power spectra are used.

.. rubric:: Version 1.2.13

Colossus has migrated from mercurial (hg) to git, and this version simply updates the documentation
and readme files. The reason for this migration is that BitBucket is retiring its mercurial support
in 2020, but this decision is just a symptom of a broader trend.

Some may wonder why the code has not been migrated to GitHub instead of BitBucket, now that the
repository system does not matter any more. The answer is that GitHub does not support the current
development model, namely a private fork of the public repository. Moreover, previous issues and
commit details cannot be transferred to GitHub and would be lost. Thus, Colossus is now a git
repository, but is still hosted on BitBucket.

.. rubric:: Version 1.2.12

This version contains some minor bug fixes, namely:

* Improved error checking in :doc:`halo_profile_spline`. 
* Fixed bug when trying to compute outer profile for objects that have no outer profile.
* Some calculations relating to dark energy, including the growth factor, can fail at far-future
  times when the w0-wa dark energy model is active. This happens because dark energy grows
  exponentially, leading to some very large values. Now, the default redshift range is reduced from
  a=200 to a=10 for w0wa and user-defined cosmologies. Thank to Antonio Villareal for pointing out
  this bug!

.. rubric:: Version 1.2.11

Fixes a bug in the :doc:`lss_mass_function` module, where redshift was not correctly passed to 
the sigma function.

.. rubric:: Version 1.2.10

The changes in this version were largely inspired by a detailed comparison with the 
`Core Cosmology Library <https://github.com/LSSTDESC/CCL>`_ (CCL) by the LSST-DESC. 

* Physical and astronomical constants were updated to IAU 2015 / PDG 2018 standard, including
  the definition of parsec/kpc/Mpc and the solar mass. Those changes translate into changes in 
  the gravitational constant in astronomical units and the critical density of the universe, which
  in turn are used in numerous functions.

  .. note::
    This change affects most outputs from Colossus, but only by factors up to 1E-4 or less. All
    stored pickles will automatically be recomputed following this change.

* Added the ``sugiyama95`` transfer function model.
* When manually changing cosmology, all derived parameters are now automatically updated. 
  Previously, changes to T_CMB0 and Neff did not have any effect. Thanks to Sebastian Bocquet for
  pointing out this issue!
* The :doc:`lss_mass_function` module now correctly passes additional arguments to the power
  spectrum, variance, and collapse overdensity functions. This only makes a difference to the
  results if the user passes additional parameters such as a tabulated power spectrum. Thanks to
  Wojciech Hellwing for finding this bug!

.. rubric:: Version 1.2.9

* Removed reference to packaging package by adding manual version comparison function.
* Added unit tests for versioning and storage.
* Added unit tests for derived constants.
* Added a new :doc:`faq` page to the documentation.

.. rubric:: Version 1.2.6

* Fixed small discrepancy in the unit system. The gravitational constant was adjusted by a factor
  of 4E-5, leading to the same discrepancy in the critical density of the universe. Thanks to Tom
  McClintock for pointing out this bug!

  .. note::
    This change affects numerous outputs from Colossus, but only by factors of around 4E-5 (and
    much less in most cases).

* Added a system to automatically delete outdated storage files. If files older than a certain
  version are found, a warning is displayed, the file is deleted, and the computations will be
  done from scratch.
* Fixed bug in the Bocquet et al. 2016 mass function for the M200c and M500c mass definitions
  (thanks to Michelle Ntampaka for catching this!).

.. rubric:: Version 1.2.5

* Renamed the ``diemer18`` concentration model to ``diemer19`` to match the publication date. 
* Changed the default concentration model from ``diemer15_orig`` to ``diemer19``. 

  .. note::
    This changes the output of all functions that use the default concentration model, namely
    :func:`~halo.concentration.concentration`, :func:`~halo.mass_adv.changeMassDefinitionCModel`, 
    and :func:`~halo.splashback.splashbackRadius`. If the user has specified a concentration model
    (which is possible in all these functions), the output will not change.

* Fixed bug in wCDM growth factor calculation. 
* Added the mass function model of Comparat et al 2017 to the :doc:`lss_mass_function` module.
* Added the bias models of Bhattacharya et al 2011 and Comparat et al 2017 to the :doc:`lss_bias`
  module. Thanks to Johan Comparat for the suggestion!

.. rubric:: Version 1.2.4

This version corresponds to the published version of the code paper.

* The Gaussian filter in the :func:`~cosmology.cosmology.Cosmology.filterFunction` (used to compute 
  the variance of the linear power spectrum, :func:`~cosmology.cosmology.Cosmology.sigma`) was 
  changed by a factor of two to adhere to the common definition.
 
  .. note::
    This change of the Gaussian filter represents a significant, not backward-compatible change.
    If you use the Gaussian filter in ANY of your calculations, please check your results -- they 
    will be affected. Before re-computing your results, please remove all temporary cosmology 
    files in ``~/.colossus/cache/cosmology`` to make sure that the change has taken effect.

  .. note::
    Due to the change in the Gaussian filter, the return of the 
    :func:`~lss.peaks.peakCurvature` function has changed. If you use this function, please check
    your results (and follow the procedure described in the note above).
* Many small fixes to the documentation, thanks to Jerry Maggioncalda for his careful proofreading!
* Activated continuous integration (i.e., automatically running the unit test suite after every
  commit). Thanks to Joseph Kuruvilla for setting that up!

.. rubric:: Version 1.2.3

* The `Diemer & Joyce 2018 <https://ui.adsabs.harvard.edu/?#abs/2018arXiv180907326D>`_
  concentration model is presented in its published form. The routine was
  sped up through a pre-computed, stored interpolation table.
* The :func:`~halo.profile_nfw.NFWProfile.xDelta` function in the :doc:`halo_profile_nfw` module was
  restructured completely. It now uses an interpolation table instead of root finding which means
  that it now allows numpy arrays as input and makes it orders of magnitude faster (depending on 
  the size of the input). The accuracy of the interpolation is better than 1E-7. The function 
  interface has two fewer parameters. 
* The cosmology of the Multidark-Planck simulations was added.

.. rubric:: Version 1.2.2

This version fixes several bugs and adds new features. Changes in the cosmology module include:

* Major bug fix: the growth factor was incorrect for :math:`w \neq -1` cosmologies, an error that
  has been rectified in this release (thanks to Lehman Garrison for catching this bug).
* The redshift interpolation tables in the cosmology module are now spaced equally in
  :math:`\ln(1 + z)` rather than :math:`z`. This change reduces the interpolation errors slightly
  and, more importantly, leads to less ringing in the first derivatives of some quantities, namely
  the linear growth factor. The new interpolation tables carry different names than the old ones,
  meaning that old cache files do not need to be deleted as the two tables can co-exist. Due to the
  changed tables (and the changes to the growth factor), some cosmology functions can exhibit
  differences of the order 0.1% compared to the previous version.
* The Planck 2018 cosmology was added (and can be used by setting ``planck18`` or
  ``planck18-only`` for the cosmology).
* The ``inverse`` option was removed from the
  :func:`~cosmology.cosmology.Cosmology.angularDiameterDistance` function because the inverse is
  multi-valued and leads to an error. 

Changes in the large-scale structure module:

* Three new bias models were added to the :doc:`lss_bias` module, namely those of Jing 1998,
  Seljak & Warren 2004, and Pillepich et al. 2010.
* The function :func:`~lss.peaks.powerSpectrumSlope` was added to the :doc:`lss_peaks` module.
  This function evaluates the slope of the power spectrum or variance at a given peak height and is
  used in the bias and concentration modules.
* Bug fix: the ``ps_args`` parameter was not used in the :func:`~lss.peaks.massFromPeakHeight` and
  :func:`~lss.peaks.peakCurvature` functions (thanks to Michael Joyce for catching this bug).

Changes in the halo module:

* The halo concentration models of Ludlow et al. 2016, Child et al. 2018, and Diemer and Joyce 2018 
  were added.
* The Diemer and Kravtsov 2015 model was updated according to Diemer and Joyce 2018.
* The default concentation model remains the original Diemer & Kravtsov 2015 model, without the
  improvements of Diemer and Joyce 2018. In a near-future release, the default concentration 
  model will switch to their new model which will influence a few functions such as 
  :func:`~halo.mass_adv.changeMassDefinitionCModel`. However, the numerical differences to the 
  previous default model are small.

Other changes:

* The function ``plotChain`` was removed from the :doc:`utils_mcmc` module to avoid including the
  ``matplotlib`` library. The function is still available as part of the
  `MCMC tutorial <_static/tutorial_utils_mcmc.html>`_.
* Numerous small improvements were made in the documentation. 

.. rubric:: Version 1.2.1

Version 1.2.1 is the version that coincided with the first publication of the code paper on
arXiv.org. The following major changes were made:

* The documentation was reworked entirely.
* All functions and parameters that were deprecated in 1.1.0 have been removed from the code
  (rather than outputting warnings).
* The ``qx`` and ``qy`` parameters in the :mod:`halo.splashback` module were renamed to ``q_in``
  and ``q_out`` to conform with the rest of the code. A number of other small inconsistencies in
  splashback radius interface were fixed.

.. rubric:: Version 1.1.0

Version 1.1.0 presents a major change to the Colossus interface, documentation, and tutorial system.
The most important changes are that

* A new top-level module for large-scale structure, LSS, has been added, including functions
  previously housed in the cosmology module, the old halo bias module, and a new module for the
  halo mass function. The LSS module covers funtions that deal with peaks or halos as a statistical
  ensemble so that the cosmology module does no longer "know" anything about halos. Conversely, the
  halo module covers functions that apply to individual halos.
* The demo scripts have been converted to much more extensive Jupyter notebook :doc:`tutorials`. 
* A number of interfaces have been made more homogeneous.
* Wherever possible, deprecated function interfaces are still present for backward compatibility
  but issue a warning. These functions and parameters will be removed in the next version.
* This documentation has been reorganized and improved, and its location has shifted to
  https://bdiemer.bitbucket.io/colossus.

The following functions are now housed in the LSS module:

* Cosmology.lagrangianR() is now :func:`lss.peaks.lagrangianR`
* Cosmology.lagrangianM() is now :func:`lss.peaks.lagrangianM`
* Cosmology.collapseOverdensity() is now :func:`lss.peaks.collapseOverdensity`
* Cosmology.peakHeight() is now :func:`lss.peaks.peakHeight`
* Cosmology.massFromPeakHeight() is now :func:`lss.peaks.massFromPeakHeight`
* Cosmology.nonLinearMass() is now :func:`lss.peaks.nonLinearMass`
* Cosmology.peakCurvature() is now :func:`lss.peaks.peakCurvature`
* The module halo.bias is now :mod:`lss.bias`.
* The LSS module contains a brand new module to compute the halo mass function,
  :mod:`lss.mass_function`.
  
The following changes apply to interfaces across modules:

* Any module that implements models (e.g., fitting functions for concentration), now features an
  ordered dictionary called ``models`` that contains class objects with the properties of the
  respective models (which vary from module to module). This change affects the power spectrum,
  bias, halo mass function, concentration, and splashback modules. These new model dictionaries
  replace the previous ``MODELS`` lists that were present in some of the modules.
* There is a new storage module as part of utilities. The storage parameter in the cosmology
  module was renamed to persistence, as was the global setting ``STORAGE`` (renamed to
  ``PERSISTENCE``). The storage module can now be used by other modules or from outside of Colossus.

Changes in the cosmology module:

* Cosmology now allows for a non-constant dark energy equations of state. The implemented dark
  energy models include a fixed or varying equation of state (see
  :class:`~cosmology.cosmology.Cosmology` class for more information). As a result, the OL0, OL(),
  and rho_L() parameters and functions were renamed to ``Ode0``, ``Ode()``, and ``rho_de()``.
* The power spectrum models were extracted into a separate module,
  :mod:`cosmology.power_spectrum`. The names of the available models were changed from ``eh98`` to
  ``eisenstein98`` and from ``eh98_smooth`` to ``eisenstein98_zb`` to conform with other Colossus
  modules.
* The ``Pk_source`` parameter was renamed to ``model`` in the
  :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` function. In functions that call the
  power spectrum, the user can pass a ``ps_args`` dictionary containing kwargs that are passed to
  the power spectrum function.
* The :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` function now takes redshift as an
  optional parameter.
* The ``text_output`` option was removed from the cosmology object.
* The :func:`~cosmology.cosmology.Cosmology.soundHorizon()` function now returns the sound horizon
  in Mpc/h rather than Mpc in order to be consistent with the rest of the cosmology module.

Changes in the LSS module:

* The :func:`~lss.peaks.collapseOverdensity()` function has been completely reworked. By default,
  it still returns the constant collapse overdensity threshold in an Einstein-de Sitter universe.
  If a redshift is passed, it applies small corrections based on the underlying cosmology. The
  previous parameters to this function will now cause an error. This change also affects all
  functions that rely on the collapse overdensity, such as :func:`~lss.peaks.peakHeight()`,
  :func:`~lss.peaks.massFromPeakHeight()`, :func:`~lss.peaks.nonLinearMass()`, and
  :func:`~lss.peaks.peakCurvature()`. These functions now accept dictionaries of parameters that
  are passed to the collapse overdensity and :func:`~cosmology.cosmology.Cosmology.sigma` functions.
* The halo bias module was extended with two new models for halo bias.
* The input units to the :func:`~lss.bias.twoHaloTerm` function are now in comoving Mpc/h rather
  than physical kpc/h in order to conform to the unit system of the LSS module.

Changes in the halo module: 

* The interface of the SO changing functions in :mod:`halo.mass_defs` has changed. The function
  previously called pseudoEvolve is now called :func:`~halo.mass_defs.evolveSO` to reflect its more
  general nature. The :func:`~halo.mass_defs.pseudoEvolve` function is a wrapper for evolveSO, and
  has one fewer parameter than previously (no final mass definition).
* The :class:`~halo.profile_dk14.DK14Profile` constructor does not take R200m as an input any more
  and instead computes it self-consistently regardless of what the other inputs are. In this new
  version, the redshift always needs to be passed to the constructor. These changes fix a bug with
  outer profiles that themselves rely on R200m as an input. Furthermore, the normalization of
  power-law outer profiles is no longer adjusted in order to maintain a constant amplitude of R200m
  changes. It is up to the user to ensure that the behavior of the outer profile makes sense
  physically.
* The ``klypin14_nu`` and ``klypin14_m`` concentration models were renamed to ``klypin16_nu`` and
  ``klypin16_m`` to maintain compatibility with the publication date of their paper.
  