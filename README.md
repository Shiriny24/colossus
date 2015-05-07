Colossus
========

Colossus is an acronym for **CO**smology, ha**LO** and large-**S**cale **S**tr**U**cture 
tool**S**. As the name suggests, Colossus constitutes a collection of modules pertaining to 
cosmology and dark matter halos, the most important of which are:

* **Cosmology**: Cosmological calculations with an emphasis on structure formation applications 
  (power spectrum, variance, correlation function, peaks in Gaussian random fields and such).
* **HaloDensityProfile**: Implementation of general and specific density profiles (including the 
  NFW and Diemer & Kravtsov 2014 profiles), pseudo-evolution, and alternative radius and mass 
  definitions such as the splashback radius.
* **HaloConcentration**: A large range of models for the concentration-mass relation (including 
  the Diemer & Kravtsov 2015 model), with a conversion to arbitrary mass definitions.

Colossus is developed with the following design goals in mind:

* **Performance**: Computationally intensive routines have been optimized for fast execution, 
  often using smart interpolation. Virtually all functions accept both numbers and numpy arrays 
  as input.
* **Pure Python**: No C modules that need to be compiled, and no dependencies beyond the 
  standard numpy and scipy libraries.
* **Object-orientation**: Class-based implementations wherever it makes sense and does not 
  hurt performance.
* **Easy (or no) installation**: You can either install Colossus as a python package with 
  pip/easy_install, or clone the repository and develop the code yourself.

While Colossus has been tested against various other codes, there is no guarantee that it is free 
of bugs. Use it at your own risk, and please report any errors, inconveniences and unclear 
documentation to the developer.

Documentation
-------------

Please see the [[Online Documentation](http://bdiemer.bitbucket.org/)] for details.

Installation
------------

You can install Colossus as a python package by executing one of the following commands, depending 
on your preferred installer software. You might need to prefix these commands with 'sudo'::

    pip install https://bitbucket.org/bdiemer/colossus/get/tip.tar.gz
    easy_install https://bitbucket.org/bdiemer/colossus/get/tip.tar.gz

Alternatively, you can clone this BitBucket repository by executing::

    hg clone https://bitbucket.org/bdiemer/colossus

For the latter method, you will need the version control system Mercurial (hg), which you can 
download [[here](http://mercurial.selenic.com/)].

Citing Colossus
---------------

If you use Colossus for a publication, please cite Diemer & Kravtsov 2015 [[ApJ 799, 108](http://adsabs.harvard.edu/abs/2015ApJ...799..108D)]. Many Colossus routines are 
based on other papers that proposed, for example, density profile or concentration-mass models. 
If you use such routines, please cite the paper(s) mentioned in the function and/or module
documentation.

License
-------

MIT

Copyright (c) 2014-2015, Benedikt Diemer, University of Chicago

bdiemer@oddjob.uchicago.edu