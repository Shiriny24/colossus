***************************************************************************************************
Installation
***************************************************************************************************

You can install Colossus as a python package by executing one of the following commands, depending 
on your preferred installer software. You might need to prefix these commands with 'sudo'::

    pip install https://bitbucket.org/bdiemer/colossus/get/tip.tar.gz
    easy_install https://bitbucket.org/bdiemer/colossus/get/tip.tar.gz

Alternatively, you can clone the public BitBucket repository [https://bitbucket.org/bdiemer/colossus] 
by executing::

    hg clone https://bitbucket.org/bdiemer/colossus

For the latter method, you will need the version control system Mercurial (hg), which you can 
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
