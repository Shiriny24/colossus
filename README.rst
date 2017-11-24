Colossus
========

Colossus is an astrophysics toolkit, the name is an acronym for **CO**\ smology, ha\ **LO**\ , 
and large-**S**\ cale **S**\ tr\ **U**\ cture tool\ **S**\ . Please consult the 
`Online Documentation <https://bdiemer.bitbucket.io/>`_ for details.

Installation
------------

The easiest way to install Colossus is by executing:

.. code:: shell

    pip install colossus

You might need to prefix this command with ``sudo``. Alternatively, you can clone the BitBucket 
repository by executing:

.. code:: shell

    hg clone https://bitbucket.org/bdiemer/colossus

After installing colossus, you should run its unit test suite to ensure that the code works as 
expected. In python, execute:

.. code:: shell

    from colossus.tests import run_tests

The output should look something like this:

.. code:: shell

    test_Ez (colossus.tests.test_cosmology.TCComp) ... ok
    test_Hz (colossus.tests.test_cosmology.TCComp) ... ok
    ...
    test_pdf (colossus.tests.test_halo_profile.TCNFW) ... ok
    test_update (colossus.tests.test_halo_profile.TCDK14) ... ok
    
    ----------------------------------------------------------------------
    Ran 72 tests in 3.327s
    
    OK

If any errors occur, please send the output to the author.

License & Citing
----------------

Author:        Benedikt Diemer (benedikt.diemer@cfa.harvard.edu)

Contributors:  Andrey Kravtsov (MCMC)

License:       MIT. Copyright (c) 2014-2017

If you use Colossus for a publication, please cite Diemer & Kravtsov 2015 
`ApJ 799, 108 <http://adsabs.harvard.edu/abs/2015ApJ...799..108D>`_ 
and/or the `ASCL entry <http://adsabs.harvard.edu/abs/2015ascl.soft01016D>`_. Many Colossus 
routines implement the results of other papers. If you use such
routines, please take care to cite the relevant papers as well (they will be mentioned in the 
function and/or module documentation).
    