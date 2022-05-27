=====================
Halo Mass Definitions
=====================

This document describes the Colossus mechanisms for dealing with halo mass and radius definitions.

-----------------------------------
Allowed mass and radius definitions
-----------------------------------

Halo masses and radii are most commonly defined using spherical overdensity mass definitions. A
spherical overdensity radius is the radius within which the halo has an overdensity :math:`\Delta`
with respect to some reference density, usually the mean or critical density of the universe:

.. math::
    M_{\rm \Delta} \equiv \frac{4 \pi}{3} \rho_{\rm mdef} R_{\rm \Delta}^3

where the ``mdef`` parameter determines the density threshold :math:`\rho_{\rm mdef}`. Another 
common mass definition is via the friends-of-friends algorithm 
(`Davis et al. 1985 <http://adsabs.harvard.edu/abs/1985ApJ...292..371D>`__). Some Colossus
functions can also handle splashback radii and masses (see :doc:`halo_splashback`). 
The ``mdef`` parameter can take on the following values in Colossus:

.. table::
    :widths: auto

    ======================== =============== ============================== ================================================================
    Type                     mdef            Examples                       Explanation
    ======================== =============== ============================== ================================================================
    Matter                   ``<int>m``      ``178m``, ``200m``             Integer times the mean matter density of the universe
    Critical                 ``<int>c``      ``200c``, ``500c``, ``2500c``  Integer times the critical density of the universe
    Virial                   ``vir``         ``vir``                        An overdensity that varies with redshift (`Bryan & Norman 1998 <http://adsabs.harvard.edu/abs/1998ApJ...495...80B>`__)
    Any SO                   ``*``           ``*``                          Any spherical overdensity mass definition
    Friends-of-friends       ``fof``         ``fof``                        Friends-of-friends mass (any linking length)
    Splashback (mean)        ``sp-apr-mn``   ``sp-apr-mn``                  Splashback computed from the mean of the particle apocenter distribution
    Splashback (percentile)  ``sp-apr-p*``   ``sp-apr-p75``, ``sp-apr-p90`` Splashback computed from percentiles (50-90) of the particle apocenter distribution
    ======================== =============== ============================== ================================================================

Some functions furthermore use an ``rmdef`` parameter which denotes spherical overdensity radii 
by ``R<mdef>`` (e.g., ``R200m``) and masses by ``M<mdef>``, e.g. ``M500c``. Not all mass 
definitions are allowed in all functions. For example, functions dealing with SO masses will 
typically not accept ``*`` or ``fof``.

-------
Modules
-------

The functions in the halo.mass module are split into various sub-modules:

* The :mod:`~halo.mass_so` module contains functions that are purely related to spherical overdensity, 
  i.e., that do not depend on the particular form of the density profile. This includes density 
  thresholds, converting mass to radius and vice versa, and the dynamical time.
* The :mod:`~halo.mass_defs` module contains functions related to converting SO mass definitions into 
  one another, as well as computing pseudo-evolution. 
* The :mod:`~halo.mass_adv` module contains advanced functions that rely on other modules such as 
  concentration, as well as some non-SO mass definitions.
* The :mod:`~halo.splashback` module contains functions related to the splashback radius and mass. 

Please see the following documentation pages for more information:

.. toctree::
    :maxdepth: 3

    halo_mass_so
    halo_mass_defs
    halo_mass_adv
    halo_splashback
