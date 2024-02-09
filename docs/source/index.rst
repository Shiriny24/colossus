======================
Colossus Documentation
======================

Colossus is a python toolkit for calculations pertaining to cosmology, the large-scale 
structure of the universe, and the properties of dark matter halos. The name is an acronym 
for **CO**\ smology, ha\ **LO** and large-\ **S**\ cale **S**\ tr\ **U**\ cture tool\ **S**\ . 
Correspondingly, Colossus consists of three top-level modules:

* :doc:`cosmology_cosmology`: Implements LCDM cosmologies with curvature, relativistic species, 
  and different dark energy equations of state. Includes standard calculations such as densities and 
  times, but also more advanced computations such as the power spectrum, variance, and correlation 
  function.
* :doc:`lss`: Deals with peaks in Gaussian random fields and the statistical properties of halos 
  such as peak height, peak curvature, halo bias, and the mass function.
* :doc:`halo`: Deals with halo masses and radii in arbitrary spherical overdensity definitions, 
  pseudo-evolution, implements general and specific halo density profiles (Einasto, Hernquist, NFW, 
  DK14), computes models for halo concentration and the splashback radius.

Colossus is developed with the following chief design goals in mind:

* *Intuitive use:* The fundamental philosophy of Colossus is to make it easy to evaluate complex 
  astrophysical quantities in a single or in a few lines of code. For this purpose, numerous fitting 
  functions have been pre-programmed.
* *Stand-alone, pure python:* No dependencies beyond numpy and scipy, no C modules to be compiled. 
  You can install Colossus either as a python package using pip or clone the repository. 
  Optionally, external Boltzmann solvers can be used.
* *Performance:* Computationally intensive routines have been optimized for speed, often using 
  interpolation tables. Virtually all functions accept either numbers or numpy arrays as input.
  
The easiest way to learn how to use Colossus is to follow the examples in the :doc:`tutorials`. 
The :ref:`search` is useful when looking for specific functions. While Colossus has been tested 
extensively, there is no guarantee that it is free of bugs. Use it at your own risk, and please 
report any errors, inconveniences and unclear documentation to the 
`author <http://www.benediktdiemer.com/>`__.

****************
License & Citing
****************

Main Developer: Benedikt Diemer (diemer@umd.edu)

Contributors: Matt Becker, Michael Joyce, Andrey Kravtsov, Steven Murray

License: MIT. Copyright (c) 2014-2024

If you use Colossus for a publication, please cite the code paper 
(`Diemer 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJS..239...35D/abstract>`__). Many Colossus 
routines implement the results of other papers. If you use such routines, please take care to 
cite the relevant papers as well (they will be mentioned in the function and/or module 
documentation).

********
Contents
********

.. toctree::
    :maxdepth: 2

    installation
    versions
    tutorials
    faq
    modules
    utils
    global

******
Search
******

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
