=====================================
Halo Density Profiles
=====================================

This document describes the Colossus mechanisms for dealing with halo density profiles.

---------------------------------------------------------------------------------------------------
General philosophy
---------------------------------------------------------------------------------------------------

The entire halo density profile module is based on a powerful base class, profile_base.HaloDensityProfile.
Some of the major design decisions regarding this class are:

* The halo density profile is represented in physical units, not some re-scaled units.
* A halo density profile is split into two parts, an inner profile (1-halo term) and an outer profile
  (2-halo term). The inner profile is what sets different profile models apart, whereas there are 
  numerous solutions for how to implement the 2-halo term.
* The outer profile is implemented as the sum of a number of possible terms, such as the mean
  density, a power law, and a 2-halo term based on the matter-matter correlation function. These
  terms are implemented in profile_base.HaloDensityProfile so that they can be used with any 
  of the derived profile classes.
* There are two fundamental aspects to a profile model: it's functional form, and the values of the
  parameters of this form. These parameters should be independent from each other, i.e., parameters
  should not be derivable from the other parameters.
* Other quantities, such as settings and values derived from the parameters, are stored as options.
  Options cannot be varied in a fit, and are not generally meant to be modified by the user once
  the profile has been instantiated.
* The functional form cannot be changed once the profile object has been
  instantiated, i.e., the user cannot change the outer profile terms, density function etc. 
* The values of the parameters can be changed either directly by the user or during fitting. After
  such changes, the update() function must be called. Otherwise, internal variables may fall out of
  sync with the profile parameters.
* Some profile forms may require knowledge of cosmological parameters and/or redshift, while some
  others do not (for example, an NFW profile without outer terms is a physical model that is independent
  of cosmology and redshift, whereas an outer term based on the mean density obviously relies on 
  cosmological information). If a profile object relies on cosmology, the user needs to set a 
  cosmology or an error will be thrown.

---------------------------------------------------------------------------------------------------
Basic usage
---------------------------------------------------------------------------------------------------

We create a density profile object which has a range of functions. For example, let us create an 
NFW profile::
    
    profile = NFWProfile(M = 1E12, mdef = 'vir', z = 0.0, c = 10.0)
    Rvir = profile.RDelta(0.0, 'vir')
    rho = profile.density(Rvir)

See the documentation of the abstract base class :class:`halo.profile_base.HaloDensityProfile` for the functionality 
of the profile objects. For documentation on spherical overdensity mass definitions, please see the 
documentation of the :doc:`halo_mass` module. The following functional forms for the density 
profile are implemented:

============================================ =============================== ========================= =============
Class                                        Explanation                     Paper                      Reference
============================================ =============================== ========================= =============
:class:`halo.profile_spline.SplineProfile`   An arbitrary density profile     ---                      ---
:class:`halo.profile_einasto.EinastoProfile` Einasto profile                 Einasto 1965              TrAlm 5, 87
:class:`halo.profile_nfw.NFWProfile`         Navarro-Frenk-White profile     Navarro et al. 1997       ApJ 490, 493
:class:`halo.profile_dk14.DK14Profile`       Diemer & Kravtsov 2014 profile  Diemer & Kravtsov 2014    ApJ 789, 1
============================================ =============================== ========================= =============

---------------------------------------------------------------------------------------------------
Example of derived profile class
---------------------------------------------------------------------------------------------------

It is easy to create a new form of the density profile in colossus. For example, let us create a
Hernquist profile. All we have to do is:

* Set the arrays of parameters and options our profile class will have
* Call the super class' constructor
* Set the profile parameters that were passed to the constructor
* Overwrite the density function (note that the function should be able to take either a number
  or a numpy array for the radius)

Here is the code::

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
improve it by letting the user pass mass and concentration to the constructor and computing rhos 
and rs, and overwriting more methods such as the density derivative and enclosed mass of the
Hernquist profile.

---------------------------------------------------------------------------------------------------
Profile fitting
---------------------------------------------------------------------------------------------------

Here, fitting refers to finding the parameters of a halo density profile which best describe a
given set of data points. Each point corresponds to a radius and a particular quantity, such as 
density, enclosed mass, or surface density. Optionally, the user can pass uncertainties on the 
data points, or even a full covariance matrix. All fitting should be done using the very general 
:func:`halo.profile_base.HaloDensityProfile.fit` routine. For example, let us fit an NFW profile 
to some density data::

    profile = NFWProfile(M = 1E12, mdef = 'vir', z = 0.0, c = 10.0)
    profile.fit(r, rho, 'rho')
    
Here, r and rho are arrays of radii and densities. Note that the current parameters of the profile 
instance are used as an initial guess for the fit, and the profile object is set to the best-fit 
parameters after the fit. Under the hood, the fit function handles multiple different fitting 
methods. By default, the above fit is performed using a least-squares minimization, but we can also 
use an MCMC sampler, for example to fit the surface density profile::

    dict = profile.fit(r, Sigma, 'Sigma', method = 'mcmc', q_cov = covariance_matrix)
    best_fit_params = dict['x_mean']
    uncertainty = dict['percentiles'][0]
    
The :func:`halo.profile_base.HaloDensityProfile.fit` function accepts many input options, some specific to the 
fitting method used. Please see the detailed documentation below.

---------------------------------------------------------------------------------------------------
Units
---------------------------------------------------------------------------------------------------

Unless otherwise noted, all functions in this module use the following units:

================ =======================================
Variable         Unit
================ =======================================
Length           Physical kpc/h
Mass             :math:`M_{\odot}/h`
Density          Physical :math:`M_{\odot} h^2 / kpc^3`
Surface density  Physical :math:`M_{\odot} h / kpc^2`
================ =======================================

---------------------------------------------------------------------------------------------------
General profile implementation
---------------------------------------------------------------------------------------------------

.. toctree::
    :maxdepth: 3

    halo_profile_base
    halo_profile_outer

---------------------------------------------------------------------------------------------------
Specific forms
---------------------------------------------------------------------------------------------------

.. toctree::
    :maxdepth: 3

    halo_profile_spline
    halo_profile_einasto
    halo_profile_nfw
    halo_profile_dk14
