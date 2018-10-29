=====================
Large-scale structure
=====================

This module implements functions related to the large-scale structure of matter in the universe. The scope includes the non-linear collapse of peaks in Gaussian random fields, i.e. density peaks, peak height and peak curvature, as well as the statistics of collapsed peaks, i.e., the halo mass function and halo bias. 

The module does not deal with the power spectrum and correlation function, or other quantities that are not related to collapsed peaks; those are based in the :doc:`cosmology_cosmology` module. Any functions concerned with the properties of individual halos are based in the :doc:`halo` module. See the :doc:`tutorials` for LSS code examples.

----------------
Module reference
----------------

.. toctree::
    :maxdepth: 3

    lss_peaks
    lss_mass_function
    lss_bias
