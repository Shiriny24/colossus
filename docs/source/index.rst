===================================================================================================
Colossus Documentation
===================================================================================================

Colossus is an acronym for **CO**\ smology, ha\ **LO** and large-\ **S**\ cale **S**\ tr\ **U**\ cture 
tool\ **S**\ . As the name suggests, Colossus constitutes a collection of modules pertaining to 
cosmology and dark matter halos, including:

* Cosmological calculations with an emphasis on structure formation applications
  (power spectrum, variance, correlation function, peaks in Gaussian random fields and more).
* General and specific halo density profiles (including the NFW, Einasto, and Diemer & Kravtsov 2014 
  profiles).
* Spherical overdensity halo masses, conversion between mass definitions, pseudo-evolution, and
  alternative radius and mass definitions such as the splashback radius.
* A large range of models for the concentration-mass relation, including a conversion to arbitrary 
  mass definitions.

Colossus is developed with the following design goals in mind:

* \ **Performance**\ : Computationally intensive routines have been optimized for fast execution, 
  often using smart interpolation. Virtually all functions accept both numbers and numpy arrays 
  as input.
* \ **Pure Python**\ : No C modules that need to be compiled, and no dependencies beyond the 
  standard numpy and scipy libraries.
* \ **Object-orientation**\ : Class-based implementations wherever it makes sense and does not 
  hurt performance.
* \ **Easy (or no) installation**\ : You can either install Colossus as a python package with 
  pip/easy_install, or clone the repository and develop the code yourself.

While Colossus has been tested against various other codes, there is no guarantee that it is free 
of bugs. Use it at your own risk, and please report any errors, inconveniences and unclear 
documentation to the developer.

***************************************************************************************************
Citing Colossus
***************************************************************************************************

If you use Colossus for a publication, please cite Diemer & Kravtsov 2015 
[`ApJ 799, 108 <http://adsabs.harvard.edu/abs/2015ApJ...799..108D>`_] and/or the
[`ASCL entry <http://adsabs.harvard.edu/abs/2015ascl.soft01016D>`_]. Many Colossus routines are 
based on other papers that proposed, for example, density profile or concentration-mass models. 
If you use such routines, please cite the paper(s) mentioned in the function and/or module
documentation.

***************************************************************************************************
Contents
***************************************************************************************************

.. toctree::
    :maxdepth: 5

    versions
    installation
    units
    global
    modules
    demos

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
