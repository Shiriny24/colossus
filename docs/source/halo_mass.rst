=====================================
Halo Mass Definitions
=====================================

This document describes the Colossus mechanisms for dealing with halo mass and radius definitions.

---------------------------------------------------------------------------------------------------
Spherical overdensity basics
---------------------------------------------------------------------------------------------------

Halo masses and radii are most commonly defined using spherical overdensity mass definitions. A
spherical overdensity radius is the radius within which the halo has an overdensity :math:`\Delta`
with respect to some reference density, usually the mean or critical density of the universe:

.. math::
    M_{\rm \Delta} \equiv \frac{4 \pi}{3} \rho_{\rm mdef} R_{\rm \Delta}^3

where the ``mdef`` parameter that determines the density threshold :math:`\rho_{\rm mdef}` can take on the 
following values in colossus:

========== ========== ==================== ================================================================
Type       mdef       Examples             Explanation
========== ========== ==================== ================================================================
Matter     '<int>m'   178m, 200m           An integer number times the mean matter density of the universe
Critical   '<int>c'   200c, 500c, 2500c    An integer number times the critical density of the universe
Virial     'vir'      vir                  An overdensity that varies with redshift (Bryan & Norman 1998)
========== ========== ==================== ================================================================

Some functions furthermore use an ``rmdef`` parameter which denotes spherical overdensity radii 
by R<mdef> (e.g., R200m) and masses by M<mdef>, e.g. M500c.

---------------------------------------------------------------------------------------------------
Modules
---------------------------------------------------------------------------------------------------

The functions in the halo.mass module are split into various sub-modules:

* The :mod:`halo.mass_so` module contains functions that are purely related to spherical overdensity, 
  i.e., that do not depend on the particular form of the density profile, such as 
  computing density thresholds and converting mass to radius and vice versa. 
* The :mod:`halo.mass_defs` module contains functions related to converting SO mass definitions into 
  one another, as well as computing pseudo-evolution. 
* The :mod:`halo.mass_adv` module contains functions describing mass definitions beyond spherical 
  overdensity, such as the mass within four scale radii.
* The :mod:`halo.splashback` module contains functions related to the splashback radius and mass. 

Please see the following pages for detailed information:

.. toctree::
    :maxdepth: 3

    halo_mass_so
    halo_mass_defs
    halo_mass_adv
    halo_splashback
