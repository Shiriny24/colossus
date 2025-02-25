Colossus
========

Colossus is an astrophysics toolkit, the name is an acronym for **CO**smology, ha**LO**, 
and large-**S**cale **S**tr**U**cture tool**S**. Please consult the [Online Documentation](https://bdiemer.bitbucket.io/colossus/) for details.

Installation
------------

The easiest way to install Colossus is by executing:

    pip install colossus

You might need to prefix this command with 'sudo'. Alternatively, you can clone the BitBucket 
repository by executing:

    git clone git@bitbucket.org:bdiemer/colossus.git

After installing colossus, you should run its unit test suite to ensure that the code works as 
expected. In python, execute:

    from colossus.tests import run_tests

The output should look something like this:

    test_home_dir (colossus.tests.test_utils.TCGen) ... ok
    test_Ez (colossus.tests.test_cosmology.TCComp) ... ok
    ...
    test_DK14ConstructorOuter (colossus.tests.test_halo_profile.TCDK14) ... ok
    test_DK14ConstructorWrapper (colossus.tests.test_halo_profile.TCDK14) ... ok
    
    ----------------------------------------------------------------------
    Ran 86 tests in 7.026s
    
    OK

If any errors occur, please send the output to the author.

License & Citing
----------------

Main Developer: Benedikt Diemer (diemer@umd.edu)

Contributors:   Matt Becker, Michael Joyce, Andrey Kravtsov, Steven Murray, Riccardo Seppi

License:        MIT. Copyright (c) 2014-2024

If you use Colossus for a publication, please cite the code paper 
([Diemer 2018](https://ui.adsabs.harvard.edu/abs/2018ApJS..239...35D/abstract)). 
Many Colossus routines implement the results of other papers. If you use such routines, please 
take care to cite the relevant papers as well (they will be mentioned in the function and/or 
module documentation).
