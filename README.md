Colossus
========

Colossus is an acronym for **CO**smology, ha**LO** and large-**S**cale **S**tr**U**cture 
tool**S**. Please see the [[Online Documentation](https://bdiemer.bitbucket.io/)] for details.

Installation
------------

You can install Colossus as a python package by executing one of the following commands, depending 
on your preferred installer software. You might need to prefix these commands with 'sudo':

    pip install https://bitbucket.org/bdiemer/colossus/get/tip.tar.gz
    easy_install https://bitbucket.org/bdiemer/colossus/get/tip.tar.gz

Alternatively, you can clone this BitBucket repository by executing:

    hg clone https://bitbucket.org/bdiemer/colossus

For the latter method, you will need the version control system Mercurial (hg), which you can 
download [[here](http://mercurial.selenic.com/)]. After installing colossus, you should run a
suite of unit tests to ensure the code works as expected:

    cd colossus/tests
    python run_tests.py
    
The output should look something like this:

    test_Ez (colossus.tests.test_cosmology.TCComp) ... ok
    test_Hz (colossus.tests.test_cosmology.TCComp) ... ok
    ...
    test_pdf (colossus.tests.test_halo_profile.TCNFW) ... ok
    test_update (colossus.tests.test_halo_profile.TCDK14) ... ok
    
    ----------------------------------------------------------------------
    Ran 72 tests in 3.327s
    
    OK

If any errors occur, please send the output to the developer (benedikt.diemer@cfa.harvard.edu).

License & Citing
----------------

Author:        Benedikt Diemer (benedikt.diemer@cfa.harvard.edu)

Contributors:  Andrey Kravtsov (MCMC)

License:       MIT. Copyright (c) 2014-2015

If you use Colossus for a publication, please cite Diemer & Kravtsov 2015 [[ApJ 799, 108](http://adsabs.harvard.edu/abs/2015ApJ...799..108D)] and/or the [[ASCL entry](http://adsabs.harvard.edu/abs/2015ascl.soft01016D)]. Many Colossus routines are 
based on other papers that proposed, for example, density profile or concentration-mass models. 
If you use such routines, please cite the paper(s) mentioned in the function and/or module
documentation.
