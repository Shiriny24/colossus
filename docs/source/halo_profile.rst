=====================================
Density Profiles
=====================================

.. toctree::
    :maxdepth: 3

    halo_profile_base
    halo_profile_spline
    halo_profile_nfw
    halo_profile_einasto
    halo_profile_dk14
    halo_profile_utils

---------------------------------------------------------------------------------------------------
Philosophy
---------------------------------------------------------------------------------------------------

Derive from HaloDensityProfile object
A profile really means a description of the 1-halo term, but descriptions of the 2-halo term are
generic and can be added to all profiles. Thus, they are generically implemented in the base class
represents a physical halo density profile

Can outer terms be added outside the constructor?? No... _outerTerms

WithOuter
parameters can be changed, update routine
some profiles demand cosmological knowledge

---------------------------------------------------------------------------------------------------
Basic usage
---------------------------------------------------------------------------------------------------

We create a density profile object which has a range of functions. For example, let us create an 
NFW profile::
    
    profile = NFWProfile(M = 1E12, mdef = 'vir', z = 0.0, c = 10.0)
    Rvir = profile.RDelta(0.0, 'vir')
    rho = profile.density(Rvir)

See the documentation of the abstract base class :class:`profile_base.HaloDensityProfile` for the functionality 
of the profile objects. For documentation on spherical overdensity mass definitions, please see the 
documentation of the :mod:`halo.basics` module. The following functional forms for the density 
profile are implemented:

============================ =============================== ========================== =============
Class                        Explanation                     Paper                      Reference
============================ =============================== ========================== =============
:func:`SplineDensityProfile` A arbitrary density profile     ---                        ---
:func:`EinastoProfile`       Einasto profile                 Einasto 1965               TrAlm 5, 87
:func:`NFWProfile`           Navarro-Frenk-White profile     Navarro et al. 1997        ApJ 490, 493
:func:`DK14Profile`          Diemer & Kravtsov 2014 profile  Diemer & Kravtsov 2014     ApJ 789, 1
============================ =============================== ========================== =============


---------------------------------------------------------------------------------------------------
Profile fitting
---------------------------------------------------------------------------------------------------

Here, fitting refers to finding the parameters of a halo density profile which best describe a
given set of data points. Each point corresponds to a radius and a particular quantity, such as 
density, enclosed mass, or surface density. Optionally, the user can pass uncertainties on the 
data points, or even a full covariance matrix. All fitting should be done using the very general 
:func:`profile_base.HaloDensityProfile.fit` routine. For example, let us fit an NFW profile to some density 
data::

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
    
The :func:`profile_base.HaloDensityProfile.fit` function accepts many input options, some specific to the 
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
Units
---------------------------------------------------------------------------------------------------

.. automodule:: halo.profile_base
    :members:

.. automodule:: halo.profile
    :members:
