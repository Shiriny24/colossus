============
Installation
============

You can install Colossus either using one of the common python package managers, or by downloading
the code directly. Colossus is compatible with both Python 2.7 and Python 3.x. However, the code is 
developed and mostly tested in Python 3 which is thus the recommended version.

.. rubric:: Package installation

The easiest way to install Colossus is by executing

.. code:: shell

    pip install colossus

You might need to prefix this command with ``sudo``. To update the code, execute

.. code:: shell

    pip install --upgrade colossus

For more information, please see the 
`pip documentation <https://packaging.python.org/tutorials/installing-packages/>`_.

.. rubric:: Repository installation

If you want to edit the code, you might prefer to clone the public BitBucket repository 
https://bitbucket.org/bdiemer/colossus by executing

.. code:: shell

   hg clone https://bitbucket.org/bdiemer/colossus

For this method, you will need the version control system Mercurial (hg), which you can 
download `here <http://mercurial.selenic.com/>`_. You need to manually add colossus
to your $PYTHONPATH variable. You can can update the code by pulling changes from the 
repository,

.. code:: shell

   hg pull
   hg up

.. rubric:: Running unit tests

After installing colossus, you should run a suite of unit tests to ensure the code works
as expected. In python, execute::

    from colossus.tests import run_tests
    
The output should look something like this::

    test_Ez (colossus.tests.test_cosmology.TCComp) ... ok
    test_Hz (colossus.tests.test_cosmology.TCComp) ... ok
    ...
    test_pdf (colossus.tests.test_halo_profile.TCNFW) ... ok
    test_update (colossus.tests.test_halo_profile.TCDK14) ... ok
    
    ----------------------------------------------------------------------
    Ran 72 tests in 3.327s
    
    OK
        
If any errors occur, please send the output to the author.
