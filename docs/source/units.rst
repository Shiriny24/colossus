*****
Units
*****

As much as possible, all functions in colossus use a consistent unit system. However, the unit 
systems for the three top-level modules vary slightly:

=================== ======================================= ======================================= =======================================
``Variable``        ``Cosmology``                           ``Large-scale structure``               ``Dark matter halos``
=================== ======================================= ======================================= =======================================                            
``Length``          Comoving Mpc/h                          Comoving Mpc/h                          Physical kpc/h
``Wavenumber``      Comoving h/Mpc                          Comoving h/Mpc                          Comoving h/Mpc
``Time``            Gigayears                               ---                                     Gigayears
``Mass``            :math:`M_{\odot}/h`                     :math:`M_{\odot}/h`                     :math:`M_{\odot}/h`
``Density``         Physical :math:`M_{\odot} h^2 / kpc^3`  Physical :math:`M_{\odot} h^2 / kpc^3`  Physical :math:`M_{\odot} h^2 / kpc^3`
``Surface density`` ---                                     ---                                     Physical :math:`M_{\odot} h / kpc^2`
=================== ======================================= ======================================= =======================================

The only quantity represented by two different units in colossus is length: distances in the halo 
module are represented as physical kpc, whereas distances in the cosmology module are 
represented as comoving Mpc. This distinction is necessary because halos are collapsed objects 
that do not "feel" the expansion of the universe any more, whereas cosmology is most naturally 
represented in comoving units.

Note that, even in the cosmology module, density units are based on physical kpc. Thus, quantities
such as the mean density of the universe are compatible with the densities computed in the halo 
module.
