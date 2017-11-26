=====================
Large-scale structure
=====================

This module implements functions related to large-scale structure. The scope includes the non-
linear collapse of Gaussian random fields, i.e. density peaks, peak height and such, as well as the
abundance of collapsed peaks (the halo mass function) and bias. 

The module does not deal with the power spectrum and correlation function of matter, or other 
quantities that are not necessarily related to collapsed peaks; those are based in the 
:doc:`cosmology_cosmology` module. Any functions concerned with the shape of collapsed peaks are 
based in the :doc:`halo` module.

----------------
Module reference
----------------

.. toctree::
    :maxdepth: 3

    lss_peaks
    lss_mass_function
    lss_bias
