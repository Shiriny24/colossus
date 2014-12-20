===================================================================================================
Colossus Documentation
===================================================================================================

Colossus is an acronym for **CO**\ smology, ha\ **LO** and large-\ **S**\ cale **S**\ tr\ **U**\ cture 
tool\ **S**\ . As the name suggests, Colossus constitutes a collection of modules for calculations
pertaining to cosmology and dark matter halos. It currently contains the following modules:

* \ **Cosmology**\ : Cosmological calculations with an emphasis on structure formation applications
  (power spectrum, variance, correlation function, peaks in Gaussian random fields etc.)
* \ **HaloDensityProfile**\ : Implementation of general and specific density profiles, including
  the Navarro-Frenk-White and Diemer & Kravtsov 2014 profiles
* \ **HaloConcentration**\ : A large range of models for the concentration-mass relation, including
  a conversion to arbitrary mass definitions

Colossus is developed with the following design goals in mind:

* \ **Performance**\ : Computationally intensive routines have been optimized for fast execution,
  often using smart interpolation. Virtually all functions accept numbers or numpy arrays as input.
* \ **Object-orientation**\ : Class-based implementations wherever it makes sense and does not
  hurt performance
* \ **Pure Python**\ : no C modules that need to be compiled, and no dependencies beyond the 
  standard numpy and scipy libraries
* \ **No installation necessary**\ : simply clone the repository, make sure its path is in the 
  python path, and import the desired Colossus modules

While Colossus has been tested against various other codes, there is no guarantee that it is free 
of bugs. Use it at your own risk, and please report any errors, inconveniences and unclear 
documentation to the developer.

***************************************************************************************************
Citing Colossus
***************************************************************************************************

If you use Colossus for a publication, please cite Diemer & Kravtsov 2014, 
`arXiv:1407.4730 <http://arxiv.org/abs/1407.4730>`_, as well as any other papers that 
form the basis for the Colossus routines you are using. Such references are mentioned 
in the documentation.

***************************************************************************************************
Contents
***************************************************************************************************

.. toctree::
    :maxdepth: 3

    Cosmology
    HaloDensityProfile
    HaloConcentration
    Utilities

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
