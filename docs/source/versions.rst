===============
Version history
===============

See below for a listing of the most important code and interface changes in Colossus, starting with
version 1.1.0. You can download older versions from the 
`PyPi History <https://pypi.org/project/colossus/#history>`__ for Colossus.

.. rubric:: Version 1.3.0 (released 06/13/2022)

This update represents a major new version. The two essential changes are a) the inclusion of 
power spectra computed by the CAMB Boltzmann code and b) a totally reworked halo density profile 
module.

First, the :mod:`~cosmology.cosmology` module has been updated significantly to allow for the direct
inclusion of CAMB power spectra. Some of the main changes are:

* The user can request a new ``camb`` power spectrum model that is evaluated internally via CAMB's
  python interface. This is much simpler to use than manually created, tabulated power spectra
  because Colossus correctly sets the important cosmological parameters and CAMB options.
* The ``camb`` module is now an optional dependency that will only ever be important if 
  required. If it is not installed, an error is thrown.
* To accommodate such generalized power spectrum models, the ``transferFunction()`` function has 
  been deprecated and will be removed in a future release. Instead, power spectra or transfer
  functions can be 
  evaluated with the :func:`~cosmology.power_spectrum.powerSpectrum` function. However, in general
  the :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` function within the cosmology
  object should be used for simplicitiy and consistency.
* The parameters to the power spectrum have been further generalized as a ``ps_args`` dictionary.
  If any parameters are passed, this dictionary needs to be kept consistent between all functions.
  For the CAMB model, passed parameters are fed to the initialization of the CAMB object, 
  allowing the user to set a wide range of options. 
* The previous ``path`` argument is now also part of ``ps_args`` in all functions (it was already
  in some functions).
* The public functions :func:`~cosmology.power_spectrum.powerSpectrumModelName` and 
  :func:`~cosmology.power_spectrum.powerSpectrumLimits` have been added to the power spectrum
  module, and :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrumNorm` has been added to the
  :class:`~cosmology.cosmology.Cosmology` class. The latter allows the user insight into the 
  normalization used to keep power spectra consistent to a fixed value of sigma8.

Second, the density profiles module has been reworked entirely. Some of the following changes are 
unfortunately not backwards compatible, as discussed below. The general philosophy of the new 
structure is to generalize the creation of profiles as much as possible, including the addition of 
outer terms. The constructor function signatures have been radically simplified to mostly take 
arbitrary keyword arguments, which are parsed by the respective constructors and by the functions 
called in turn. Please feel free to get in touch if you have trouble migrating your code to the 
new version. The main changes are as follows:

* Generalized construction of profiles

  * The profile constructor was generalized to work with the keyword arguments given by the user. A
    derived class now only needs to list its parameter (and perhaps option) names, and the parent
    constructor attempts to construct a profile from these arguments. If not all native parameters
    are given, the constructor looks for a function called ``setNativeParameters`` that sets the
    parameters based on mass, concentration, and redshift. Aside from this function, the entire
    logic for creating profiles is now contained in the parent class.

  * Child classes now must implement a :func:`~halo.profile_base.HaloDensityProfile.setNativeParameters`
    function that needs to be able to accept any mass definition unless the ``allowed_mdefs``
    parameter is set when calling the parent constructor. 

  * All derived classes must now contain a normalization ``rhos`` because this variable is used to
    renormalize the profile in the presence of outer terms. The exception are parameter-free
    profiles, such as splines.

  * The generalized constructor fixes a previous issue when creating profiles with outer
    (infalling) terms. If the profile parameters are determined from a mass and concentration,
    and the outer profile depends on a radius such as R200m, the process of finding the profile
    parameters is iterative. This iteration was not performed for all profiles.

* Easier construction of composite (inner + outer) profiles
  
  * The new :func:`~halo.profile_composite.compositeProfile` function allows the user to easily
    create any combination of inner and outer profiles using shortcodes.
  * The user is responsible for passing the appropriate parameters to this function; otherwise,
    respective constructors throw errors.

* The new :doc:`halo_profile_diemer22` has been added; this form separately describes the orbiting
  and infalling components and is now recommended over the DK14 profile.

* The DK14 profile has been reworked

  * In the DK14 profile, all options have been removed as they are only needed for the 
    constructor.
  * The ``getDK14ProfileWithOuterTerms`` function has been removed from the DK14 profile module, 
    and has been replaced by the general :func:`~halo.profile_composite.compositeProfile` function. The
    signature is similar, but the parameter names are now consistent with the constructors of the 
    respective outer terms. 
  * The function ``DK14Profile.M4rs`` has been removed. The result can easily
    be found by evaluating the enclosed mass within four scale radii.

* Changes in other profile modules

  * For the NFW profile, the ``fundamentalParameters`` function (which has now been replaced by
    :func:`~halo.profile_nfw.NFWProfile.setNativeParameters`) was a class method, meaning that 
    it could be called without calling the constructor first. This routine has been renamed to 
    :func:`~halo.profile_nfw.NFWProfile.nativeParameters`.
  
* Fitting

  * The transformation between linear and log parameters has been radically simplified. All 
    parameters, including those of the outer profiles, are now by default fit in log space.
    This can lead to slightly different results compared to previous versions. The user can change
    this behavior by overwriting certain functions.
  * The old ``scipy.optimize.leastsq`` function was replaced by the newer 
    ``scipy.optimize.least_squares`` interface, which contains more advanced algorithms such 
    ``trf`` (the new default fitter).
  * The user can now pass parameter bounds in least-squares fits.
  * MCMC fits can now also be performed in log space by default to ensure the positivity of the 
    parameters.

* All profile documentation pages have been overhauled.
* The profile :doc:`tutorials` have been improved and expanded.
* The unit test suite has been improved and expanded.

A few other changes:

* The cosmology setter functions :func:`~cosmology.cosmology.setCosmology` and 
  :func:`~cosmology.cosmology.addCosmology` now support keyword arguments in addition 
  to a dictionary, which makes setting and overwriting parameters more convenient.
* Small bug fix where evaluating the dark energy density for an array of redshifts sometimes
  returned a number instead of an array.
* Convert np.int to int and np.float to float to avoid deprecation warnings.

.. rubric:: Version 1.2.19 (released 09/02/2021)

* Added the final version of the ``ishiyama21`` concentration model, which was renamed from
  ``ishiyama20`` to conform with the published version. This model now also contains fits for 
  the 500c mass definition and for relaxed halos.

.. rubric:: Version 1.2.18 (released 03/18/2021)

* Added the new ``seppi20`` mass function model. Thanks to Riccardo Seppi for the implementation
  and tutorial!
* Fixed a bug in the power spectrum derivative at z > 0 (thanks to Michael Joyce for finding this
  bug!)

.. rubric:: Version 1.2.17 (released 08/07/2020)

This version contains the new ``ishiyama20`` concentration model, which is a recalibration of the
``diemer19`` model based on the Uchuu simulation.

.. rubric:: Version 1.2.16 (released 07/15/2020)

Changes in this version include:

* The cosmology module now supports conversions to and from Astropy, including a number of dark 
  energy models. See the :func:`~cosmology.cosmology.Cosmology.toAstropy` and 
  :func:`~cosmology.cosmology.fromAstropy` functions. Thanks to Steven Murray for the idea and 
  code!
* A new splashback model, ``diemer20``, was added and made the default splashback model. This model
  is a recalibration of the ``diemer17`` model, with percent-level changes.
* The implementation of these splashback models has changed, with some interface changes to the 
  convenience functions.
* The main :func:`~halo.splashback.splashbackModel` function does not provide a default definition
  for the ``diemer17`` and ``diemer20`` models any longer (such as the mean or higher percentiles
  of the particle splashback distribution). The definition matters quite a bit and should be 
  provided by the user to avoid confusion. The function now throws an error if no definition
  is passed.
* A new mass function model, ``diemer20`` has been added (not to be mistaken for the splashback
  radius and mass model of the same name). This model is the first to predict splashback mass
  functions.
* The integration in :func:`~cosmology.cosmology.Cosmology.sigma` was made more robust in the case
  where the user has specified a lower or upper limit to the integration. In particular, the 
  calculation of the tree integration limit was improved and the code now automatically increases
  the number of bins in the intepolation table because the solution oscillates near the cutoff.
* A number of functions in the cosmology and halo modules are now safe to input of integers instead
  of float. For example, the growthFactorUnnormalized function returned wrong values when "1" was
  given instead of "1.0" or "1.". Thanks to Yucheng Zhang for pointing this out! 

.. rubric:: Version 1.2.15 (released 04/15/2020)

Changes in this version include:

* In self-similar cosmologies, the correlation function is now computed from analytical expressions
  rather than numerical integration (thanks to Michael Joyce for the analytical formulae).
* The variance sigma can be computed between user-defined lower and/or upper limits in k-space.
  This feature is useful when calculating the variance in a box of limited size, for example.

.. rubric:: Version 1.2.14 (released 01/23/2020)

Changes in this version include:

* The user can now pass power spectrum arguments to the Diemer & Joyce 2019 concentration model,
  for example, in order to use a non-standard power spectrum.
* The code returns more informative error messages when tabulated power spectra are used.
* The normalization of the power spectrum for self-similar (power-law) cosmologies has been fixed
  for both the tophat and Gaussian filters, and the variance is now computed from the analytical
  expression rather than numerical integration (thanks to Michael Joyce for finding this bug and
  providing the analytical formulae!).

.. rubric:: Version 1.2.13 (released 11/08/2019)

Colossus has migrated from mercurial (hg) to git, and this version simply updates the documentation
and readme files. The reason for this migration is that BitBucket is retiring its mercurial support
in 2020, but this decision is just a symptom of a broader trend.

Some may wonder why the code has not been migrated to GitHub instead of BitBucket, now that the
repository system does not matter any more. The answer is that GitHub does not support the current
development model, namely a private fork of the public repository. Moreover, previous issues and
commit details cannot be transferred to GitHub and would be lost. Thus, Colossus is now a git
repository, but is still hosted on BitBucket.

.. rubric:: Version 1.2.12 (released 10/28/2019)

This version contains some minor bug fixes, namely:

* Improved error checking in :doc:`halo_profile_spline`. 
* Fixed bug when trying to compute outer profile for objects that have no outer profile.
* Some calculations relating to dark energy, including the growth factor, can fail at far-future
  times when the w0-wa dark energy model is active. This happens because dark energy grows
  exponentially, leading to some very large values. Now, the default redshift range is reduced from
  a=200 to a=10 for w0wa and user-defined cosmologies. Thank to Antonio Villareal for pointing out
  this bug!

.. rubric:: Version 1.2.11 (released 08/12/2019)

Fixes a bug in the :doc:`lss_mass_function` module, where redshift was not correctly passed to 
the sigma function.

.. rubric:: Version 1.2.10 (released 08/05/2019)

The changes in this version were largely inspired by a detailed comparison with the 
`Core Cosmology Library <https://github.com/LSSTDESC/CCL>`__ (CCL) by the LSST-DESC. 

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

.. rubric:: Version 1.2.9 (released 03/23/2019)

* Removed reference to packaging package by adding manual version comparison function.
* Added unit tests for versioning and storage.
* Added unit tests for derived constants.
* Added a new :doc:`faq` page to the documentation.

.. rubric:: Version 1.2.6 (released 03/01/2019)

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

.. rubric:: Version 1.2.5 (released 01/30/2019)

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

.. rubric:: Version 1.2.4 (released 10/29/2018)

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
* The `Diemer & Joyce 2018 <https://ui.adsabs.harvard.edu/?#abs/2018arXiv180907326D>`__
  concentration model is presented in its published form. The routine was
  sped up through a pre-computed, stored interpolation table.
* The :func:`~halo.profile_nfw.NFWProfile.xDelta` function in the :doc:`halo_profile_nfw` module was
  restructured completely. It now uses an interpolation table instead of root finding which means
  that it now allows numpy arrays as input and makes it orders of magnitude faster (depending on 
  the size of the input). The accuracy of the interpolation is better than 1E-7. The function 
  interface has two fewer parameters. 
* The cosmology of the Multidark-Planck simulations was added.

.. rubric:: Version 1.2.2 (released 07/31/2018)

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
  `MCMC tutorial <_static/tutorial_utils_mcmc.html>`__.
* Numerous small improvements were made in the documentation. 

.. rubric:: Version 1.2.1 (released 12/13/2017)

Version 1.2.1 is the version that coincided with the first publication of the code paper on
arXiv.org. The following major changes were made:

* The documentation was reworked entirely.
* All functions and parameters that were deprecated in 1.1.0 have been removed from the code
  (rather than outputting warnings).
* The ``qx`` and ``qy`` parameters in the :mod:`halo.splashback` module were renamed to ``q_in``
  and ``q_out`` to conform with the rest of the code. A number of other small inconsistencies in
  splashback radius interface were fixed.

.. rubric:: Version 1.1.0 (released 11/27/2017)

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
  