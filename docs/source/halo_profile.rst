=====================
Halo Density Profiles
=====================

This document describes the Colossus mechanisms for dealing with halo density profiles.

------
Basics
------

The halo density profile module is based on a powerful base class, 
:class:`~halo.profile_base.HaloDensityProfile`. Some of the major design decisions are as follows:

* The halo density profile is represented in physical units.
* A halo density profile is split into two parts, an inner profile (1-halo term) and an outer 
  profile (infalling matter plus 2-halo term). The outer profile consists of the sum of a number 
  of possible terms, such as the mean density, a power law, and a 2-halo term based on the 
  matter-matter correlation function. These terms can be added to any implementation of the inner 
  profile.
* There are two fundamental aspects to a model of the inner profile: it's functional form, and the 
  values of the parameters of this form. These parameters should be independent from each other, 
  i.e., parameters should not be derivable from the other parameters.
* Other quantities, such as settings and values derived from the parameters, are stored as 
  so-called "options". Options cannot be varied in a fit, and are not generally meant to be 
  modified by the user once the profile has been instantiated.
* The functional form cannot be changed once the profile object has been
  instantiated, i.e., the user cannot change the outer profile terms, density function etc. 
* The values of the parameters can be changed either directly by the user or during fitting. After
  such changes, the update() function must be called. Otherwise, internal variables may fall out of
  sync with the profile parameters.
* Some profile forms may require knowledge of cosmological parameters and/or redshift, while some
  others do not (for example, an NFW profile without outer terms is a physical model that is independent
  of cosmology and redshift, whereas an outer term based on the mean density obviously relies on 
  cosmological information). If a profile object relies on cosmology, the user needs to set a 
  cosmology or an exception will be raised.

Almost all profile related functions are encapsulated within profile objects. For example, let us 
create an NFW profile for a halo with a particular virial mass and concentration::
    
    from colossus.halo import profile_nfw
    profile = profile_nfw.NFWProfile(M = 1E12, mdef = 'vir', z = 0.0, c = 10.0)
    R200m = profile.RDelta(0.0, '200m')
    rho = profile.density(R200m)
    Sigma = profile.surfaceDensity(R200m)
    M200m = profile.enclosedMass(R200m)

Please consult the documentation of the abstract base class
:class:`~halo.profile_base.HaloDensityProfile` for the basic functionality of profile objects,
and :doc:`halo_profile_outer` for instructions on how to add outer profile terms. For documentation 
on spherical overdensity mass definitions, please see the documentation of the :doc:`halo_mass` 
module. For more examples of how to use the Colossus profile modules, see :doc:`tutorials`.

----------------------
Density profile models
----------------------

The following functional forms for the density profile are currently implemented:

.. table::
   :widths: auto

   =================================================== =========================================== =========================
   Class                                               Explanation                                 Paper                    
   =================================================== =========================================== =========================
   :class:`~halo.profile_spline.SplineProfile`         An arbitrary density profile                --
   :class:`~halo.profile_einasto.EinastoProfile`       Einasto profile                             `Einasto 1965 <http://adsabs.harvard.edu/abs/1965TrAlm...5...87E>`_
   :class:`~halo.profile_hernquist.HernquistProfile`   Hernquist profile                           `Hernquist 1990 <http://adsabs.harvard.edu/abs/1990ApJ...356..359H>`_
   :class:`~halo.profile_nfw.NFWProfile`               Navarro-Frenk-White profile                 `Navarro et al. 1997 <http://adsabs.harvard.edu/abs/1997ApJ...490..493N>`_
   :class:`~halo.profile_dk14.DK14Profile`             Diemer & Kravtsov profile                   `Diemer & Kravtsov 2014 <http://adsabs.harvard.edu/abs/2014ApJ...789....1D>`_
   =================================================== =========================================== =========================

----------------------------
Creating a new profile class
----------------------------

It is easy to create a new form of the density profile in colossus. For example, let us create a
Hernquist profile. This profile already exists in Colossus, but it is a suitable example 
nevertheless. All we have to do is:

* Set the dictionaries for parameters and options to make our profile class "self-aware"
* Call the super class' constructor
* Set the profile parameters that were passed to the constructor
* Overwrite the density function (note that the function should be able to take either a number
  or a numpy array for the radius)

Here is the code::

    from colossus.halo import profile_base
   
    class HernquistProfile(profile_base.HaloDensityProfile):  
       
        def __init__(self, rhos, rs):
            self.par_names = ['rhos', 'rs']
            self.opt_names = []
            profile_base.HaloDensityProfile.__init__(self)
            self.par['rhos'] = rhos
            self.par['rs'] = rs
            return
        
        def densityInner(self, r):
            x = r / self.par['rs']
            density = self.par['rhos'] / x / (1.0 + x)**3
            return density
      
This derived class inherits all the functionality of the parent class, including other physical
quantities (enclosed mass, surface density etc), derivatives, fitting to data, and the ability to
add outer profile terms. In order to make this class more convenient to use and faster, we could 
improve it by letting the user pass mass and concentration to the constructor and computing 
``rhos`` and ``rs``, and overwriting more methods such as the density derivative and enclosed 
mass of the Hernquist profile.

---------------
Profile fitting
---------------

Here, fitting refers to finding the parameters of a halo density profile which best describe a
given set of data points. Each point corresponds to a radius and a particular quantity, such as 
density, enclosed mass, or surface density. Optionally, the user can pass uncertainties on the 
data points, or even a full covariance matrix. All fitting should be done using the very general 
:func:`~halo.profile_base.HaloDensityProfile.fit` routine. For example, let us fit an NFW profile 
to some density data::

    profile = NFWProfile(M = 1E12, mdef = 'vir', z = 0.0, c = 10.0)
    profile.fit(r, rho, 'rho')
    
Here, ``r`` and ``rho`` are arrays of radii and densities. Note that the current parameters of the 
profile instance are used as an initial guess for the fit, and the profile object is set to the best-fit 
parameters after the fit. Under the hood, the fit function handles multiple different fitting 
methods. By default, the above fit is performed using a least-squares minimization, but we can also 
use an MCMC sampler, for example to fit the surface density profile::

    dict = profile.fit(r, Sigma, 'Sigma', method = 'mcmc', q_cov = covariance_matrix)
    best_fit_params = dict['x_mean']
    uncertainty = dict['percentiles'][0]
    
The :func:`~halo.profile_base.HaloDensityProfile.fit` function accepts many input options, some 
specific to the fitting method used. Please see the detailed documentation for details and the 
:doc:`tutorials` for code examples.

---------------
Module contents
---------------

The following documents describe the general functionality of inner and outer profiles:

.. toctree::
    :maxdepth: 1

    halo_profile_base
    halo_profile_outer

The following documents describe the specific implementations for each profile model:

.. toctree::
    :maxdepth: 1

    halo_profile_spline
    halo_profile_einasto
    halo_profile_hernquist
    halo_profile_nfw
    halo_profile_dk14
