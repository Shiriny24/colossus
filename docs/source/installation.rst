===================================================================================================
Installation
===================================================================================================

You can install Colossus either using one of the common python package managers, or by downloading
the code directly.

Colossus is compatible with both Python 2.7 and Python 3.x. However, the code is developed and
mostly tested in Python 3 which is thus the recommended version.

***************************************************************************************************
Using setuptools
***************************************************************************************************

To install Colossus as a python package, execute one of the following commands, depending 
on your preferred installer software. You might need to prefix these commands with 'sudo'::

    pip install https://bitbucket.org/bdiemer/colossus/get/tip.tar.gz
    easy_install https://bitbucket.org/bdiemer/colossus/get/tip.tar.gz

Colossus does not yet have a PyPi registry entry due to a naming conflict which will hopefully be
resolved in the future. Once the code is installed, please test its functionality on your 
machine by running the test suite::

    from colossus.tests import run_tests
    run_tests
    
The output should look something like this::

    test_Ez (colossus.tests.test_cosmology.TCComp) ... ok
    test_Hz (colossus.tests.test_cosmology.TCComp) ... ok
    ...
    test_pdf (colossus.tests.test_halo_profile.TCNFW) ... ok
    test_update (colossus.tests.test_halo_profile.TCDK14) ... ok
    
    ----------------------------------------------------------------------
    Ran 72 tests in 3.327s
    
    OK

If any errors occur, please send the console output to the developer.

***************************************************************************************************
Manual installation
***************************************************************************************************

Alternatively, you can clone the public BitBucket repository [https://bitbucket.org/bdiemer/colossus] 
by executing::

    hg clone https://bitbucket.org/bdiemer/colossus

For this method, you will need the version control system Mercurial (hg), which you can 
download [`here <http://mercurial.selenic.com/>`_]. After installing colossus, you should run a
suite of unit tests to ensure the code works as expected::

    cd colossus/tests
    python run_tests.py
    
The output should look something like this::

    test_Ez (colossus.tests.test_cosmology.TCComp) ... ok
    test_Hz (colossus.tests.test_cosmology.TCComp) ... ok
    ...
    test_pdf (colossus.tests.test_halo_profile.TCNFW) ... ok
    test_update (colossus.tests.test_halo_profile.TCDK14) ... ok
    
    ----------------------------------------------------------------------
    Ran 72 tests in 3.327s
    
    OK
        
If any errors occur, please send the output to the developer.
