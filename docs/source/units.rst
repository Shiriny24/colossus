*****
Units
*****

As much as possible, all functions in Colossus use a consistent unit system.
The only quantity represented by two different units in Colossus is length: distances in the halo 
module are represented as physical kpc, whereas distances in the cosmology module are 
represented as comoving Mpc. This distinction is necessary because halos are collapsed objects 
that do not "feel" the expansion of the universe any more, whereas cosmology is most naturally 
represented in comoving units. For completeness, the following table lists all units used by
each top-level module:

.. table::
   :widths: auto

   =================== ============================================== ================================================= =======================================
   ``Variable``        ``Cosmology``                                  ``Large-scale str.``                              ``Dark matter halos``
   =================== ============================================== ================================================= =======================================                            
   ``Length``          Comoving :math:`{\rm Mpc}/h`                   Comoving :math:`{\rm Mpc}/h`                      Physical :math:`{\rm kpc}/h`
   ``Wavenumber``      Comoving :math:`h/{\rm Mpc}`                   Comoving :math:`h/{\rm Mpc}`                      Comoving :math:`h/{\rm Mpc}`
   ``Time``            Gigayears                                      ---                                               Gigayears
   ``Mass``            :math:`M_{\odot}/h`                            :math:`M_{\odot}/h`                               :math:`M_{\odot}/h`
   ``Density``         Physical :math:`M_{\odot} h^2 / {\rm kpc}^3`   Physical :math:`M_{\odot} h^2 / {\rm kpc}^3`      Physical :math:`M_{\odot} h^2 / {\rm kpc}^3`
   ``Surface density`` ---                                            ---                                               Physical :math:`M_{\odot} h / {\rm kpc}^2`
   =================== ============================================== ================================================= =======================================

Note that, even in the cosmology module, density units are based on physical kpc. Thus, quantities
such as the mean density of the universe are compatible with the densities computed in the halo 
module.
