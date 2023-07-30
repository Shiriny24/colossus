=====================
Halo Density Profiles
=====================

This document describes the Colossus mechanisms for dealing with halo density profiles. For more
extensive code examples, please see the :doc:`tutorials`. For documentation on spherical 
overdensity mass definitions, please see the documentation of the :doc:`halo_mass` module.

------
Basics
------

The halo density profile module is based on a powerful base class, 
:class:`~halo.profile_base.HaloDensityProfile`, from which particular models (such as NFW or 
Einasto profiles) are derived. Some of the major design decisions are as follows:

* The halo density profile is represented in physical units.
* A halo density profile is split into two parts, an inner (orbiting or 1-halo) profile and an 
  outer profile (infalling plus 2-halo term). The outer profile can consist of the sum of a number
  of possible terms, such as the mean density, a power law, or a 2-halo term based on the 
  matter-matter correlation function. These terms can be added to any implementation of the inner 
  profile.
* There are two fundamental aspects to a model of the inner profile: it's functional form, and the 
  values of the parameters of this form. These parameters should be independent from each other, 
  i.e., parameters should not be derivable from the other parameters.
* Other quantities, such as settings and values derived from the parameters, are stored as 
  so-called "options".
* The functional form cannot be changed once the profile object has been instantiated, i.e., 
  the user cannot change the outer profile terms, density function etc. 
* The values of the parameters can be changed either directly by the user or during fitting. After
  such changes, the update() function must be called. Otherwise, internal variables may fall out of
  sync with the profile parameters.
* Some profile forms may require knowledge of cosmological parameters and/or redshift, while some
  others do not (for example, an NFW profile without outer terms is a physical model that is independent
  of cosmology and redshift, whereas an outer term based on the mean density obviously relies on 
  cosmological information). If a profile object relies on cosmology, the user needs to set a 
  cosmology or an exception will be raised.

The following functional forms for the inner (orbiting or 1-halo) and outer (infalling or two-halo)
density profile are currently implemented:

.. table::
   :widths: auto

   ================= =========================================================== ============================================== =========================
   Short code        Class                                                       Explanation                                    Reference                    
   ================= =========================================================== ============================================== =========================
   ``spline``        :class:`~halo.profile_spline.SplineProfile`                 An arbitrary density profile                   --
   ``einasto``       :class:`~halo.profile_einasto.EinastoProfile`               Einasto profile                                `Einasto 1965 <http://adsabs.harvard.edu/abs/1965TrAlm...5...87E>`__
   ``hernquist``     :class:`~halo.profile_hernquist.HernquistProfile`           Hernquist profile                              `Hernquist 1990 <http://adsabs.harvard.edu/abs/1990ApJ...356..359H>`__
   ``nfw``           :class:`~halo.profile_nfw.NFWProfile`                       Navarro-Frenk-White profile                    `Navarro et al. 1997 <http://adsabs.harvard.edu/abs/1997ApJ...490..493N>`__
   ``dk14``          :class:`~halo.profile_dk14.DK14Profile`                     Diemer & Kravtsov profile                      `Diemer & Kravtsov 2014 <http://adsabs.harvard.edu/abs/2014ApJ...789....1D>`__
   ``diemer22``      :class:`~halo.profile_diemer23.ModelAProfile`               Truncated exponential profile (default)        `Diemer 2023 <https://ui.adsabs.harvard.edu/abs/2022arXiv220503420D/abstract>`__
   ``diemer22b``     :class:`~halo.profile_diemer23.ModelBProfile`               Truncated exponential profile (adjusted)       `Diemer 2023 <https://ui.adsabs.harvard.edu/abs/2022arXiv220503420D/abstract>`__
   ``mean``          :class:`~halo.profile_outer.OuterTermMeanDensity`           The mean matter density of the Universe        --
   ``cf``            :class:`~halo.profile_outer.OuterTermCorrelationFunction`   Matter-matter correlation times bias           --
   ``pl``            :class:`~halo.profile_outer.OuterTermPowerLaw`              Power law in overdensity                       --
   ``infalling``     :class:`~halo.profile_outer.OuterTermInfalling`             Power law with smooth transition to constant   `Diemer 2023 <https://ui.adsabs.harvard.edu/abs/2022arXiv220503420D/abstract>`__
   ================= =========================================================== ============================================== =========================

----------------------
Creating profiles
----------------------

All profile models can be created from either their native, internal parameters or from a given
mass and concentration. For example, let us create an NFW profile and inspect its parameters::
    
    from colossus.halo import profile_nfw
    
    p1 = profile_nfw.NFWProfile(rhos = 1E6, rs = 80.0)
    print(p1.par)
    
    >>> OrderedDict([('rhos', 1000000.0), ('rs', 80.0)])
    
If we create a profile from a mass and concentation, we first need to set a cosmology; this is 
also the case for many other functions::

    from colossus.cosmology import cosmology
	
    cosmology.setCosmology('planck18')
    p2 = profile_nfw.NFWProfile(M = 1E12, mdef = 'vir', z = 0.0, c = 10.0)
    print(p2.par)
    
    >>> OrderedDict([('rhos', 6378795.928070417), ('rs', 20.311309856581044)])
    
Regardless of how the profile object was created or exactly how it is implemented under the hood,
the base class allows us to evaluate a large range of functions::
   
    R200m = profile.RDelta(0.0, '200m')
    r = 10**np.linspace(-2.0, 1.0, 100) * R200m

    rho = profile.density(r)
    Sigma = profile.surfaceDensity(r)
    ...

Please consult the documentation of the abstract base class
:class:`~halo.profile_base.HaloDensityProfile` for the basic functionality of profile objects. 
For more examples of how to use the Colossus profile modules, see :doc:`tutorials`.

---------------------------------
Composite inner+outer profiles
---------------------------------

The models for the inner profile listed above are not designed to describe halos out to large radii
because, somewhere around the virial radius, the contribution from infalling matter starts to 
become significant. This term is not modeled in the inner profiles. We can create composite 
profiles either "manually" or using a wrapper function. To demonstrate the first method, we create
an NFW profile to which we add the mean density of the Universe::
    
    from colossus.halo import profile_outer

    outer_term_mean = profile_outer.OuterTermMeanDensity(z = z)
    p = profile_nfw.NFWProfile(M = Mvir, c = cvir, z = z, mdef = 'vir', outer_terms = [outer_term_mean])

The ``outer_terms`` keyword can be used with any class derived from 
:class:`~halo.profile_base.HaloDensityProfile`, and the outer terms are automatically taken into
account when computing the native profile parameters from mass and concentration. However, it is
easier to create profiles using the following wrapper::

    from colossus.halo import profile_composite

    p = profile_composite.compositeProfile('einasto', outer_names = ['mean', 'cf'],
             M = 1E12, mdef = 'vir', z = 0.0, c = 10.0, bias = 5.0)
             
Besides the usual mass and concentration parameters, we also had to pass the bias (in the case of
the correlation-function outer term). Once a composite profile has been created, the outer terms 
are automatically taken into account in all functions such as density, surface density, etc. 
For details on the available outer terms and their parameters, please see :doc:`halo_profile_outer`.
Note that in this particular case, the correlation function becomes negative at large radii; thus,
the integration depth must be limited when computing the surface density.

---------------
Fitting
---------------

Here, fitting refers to finding the parameters of a halo density profile which best describe a
given set of data points. Each point corresponds to a radius and a particular quantity, such as 
density, enclosed mass, surface density, or DeltaSigma. Optionally, the user can pass uncertainties 
on the data points, or even a full covariance matrix. All fitting should be done using the very general 
:func:`~halo.profile_base.HaloDensityProfile.fit` routine. For example, let us fit an NFW profile 
to some density data::

    profile = NFWProfile(M = 1E12, mdef = 'vir', z = 0.0, c = 10.0)
    profile.fit(r, rho, 'rho')
    
Here, ``r`` and ``rho`` are arrays of radii and densities. The current parameters of the 
profile instance are used as an initial guess for the fit, and the profile object is set to the best-fit 
parameters after the fit. Under the hood, the fit function handles multiple different fitting 
methods. By default, the above fit is performed using a least-squares minimization, but we can also 
use an MCMC sampler, for example to fit the surface density profile::

    dic = profile.fit(r, Sigma, 'Sigma', method = 'mcmc', q_cov = covariance_matrix)
    best_fit_params = dic['x_mean']
    uncertainty = dic['percentiles'][0]
    
The :func:`~halo.profile_base.HaloDensityProfile.fit` function accepts many input options, some 
specific to the fitting method used. Please see the detailed documentation for details and the 
:doc:`tutorials` for code examples.

----------------------------
Creating a new profile class
----------------------------

It is easy to create a new form of the density profile in colossus. For example, let us create a
Hernquist profile. This profile already exists in Colossus, but it is a suitable example 
nevertheless. All we have to do is:

* Set the dictionaries for parameters and options to make our profile class "self-aware"
* Call the super class' constructor
* Overwrite the density function (which should be able to take either a number or a numpy array 
  as input)
* Provide a routine to convert mass and concentration into the native parameters.

Here is the code::

	class HernquistProfile(profile_base.HaloDensityProfile):
	
	    def __init__(self, **kwargs):
	        self.par_names = ['rhos', 'rs']
	        self.opt_names = []
	        profile_base.HaloDensityProfile.__init__(self, **kwargs)
	        return
	
	    def densityInner(self, r):
	        x = r / self.par['rs']
	        density = self.par['rhos'] / x / (1.0 + x)**3
	        return density
	    
	    def setNativeParameters(self, M, c, z, mdef, **kwargs):
	        self.par['rs'] = mass_so.M_to_R(M, z, mdef) / c
	        self.par['rhos'] = M / (2 * np.pi * rs**3) / c**2 * (1.0 + c)**2
	        return    

This derived class inherits all the functionality of the parent class, including other physical
quantities (enclosed mass, surface density etc), derivatives, fitting to data, and the ability to
add outer profile terms.

---------------
Module contents
---------------

The following documents describe the general functionality of inner and outer profiles:

.. toctree::
    :maxdepth: 1

    halo_profile_base
    halo_profile_outer
    halo_profile_composite

The following documents describe the specific implementations for each profile model:

.. toctree::
    :maxdepth: 1

    halo_profile_spline
    halo_profile_einasto
    halo_profile_hernquist
    halo_profile_nfw
    halo_profile_dk14
    halo_profile_diemer23
