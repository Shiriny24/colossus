============
Installation
============

You can install Colossus either using the common python package manager pip or by downloading the 
code directly. 

.. rubric:: Requirements

Colossus is compatible with both Python 2.7 and Python 3.x. However, the code is developed and 
mostly tested in Python 3, which is thus the recommended version. Colossus requires the following 
standard packages:

* `numpy <http://www.numpy.org/>`__
* `scipy <https://www.scipy.org/>`__
* `six <https://pypi.org/project/six/>`__

Some tutorial notebooks also use other packages, for example matplotlib for plotting. In addition,
the following dependencies are optional:

* `astropy <https://www.astropy.org/>`__ (only if the astropy cosmology converter is used)
* `camb <https://camb.readthedocs.io/en/latest/index.html>`__ (only if a CAMB power spectrum is requested)

.. rubric:: Package installation

The easiest way to install Colossus is by executing

.. code:: shell

    pip install colossus

You might need to prefix this command with ``sudo``. To update the code, execute

.. code:: shell

    pip install --upgrade colossus

If the numpy and scipy packages are not already installed, you can similarly install them with the 
pip command. For more information, please see the 
`pip documentation <https://packaging.python.org/tutorials/installing-packages/>`__.

.. rubric:: Repository installation

If you want to edit the code, you might prefer to clone the public BitBucket repository 
https://bitbucket.org/bdiemer/colossus/src/master/ by executing

.. code:: shell

   git clone git@bitbucket.org:bdiemer/colossus.git

You will also need to manually include Colossus in your ``$PYTHONPATH`` variable, for example 
by adding this command to your shell's initialization script (e.g., ``bashrc``):

.. code:: shell
   
   export PYTHONPATH=$PYTHONPATH:/users/me/code/colossus

where the path is, of course, replaced with the location of Colossus on your system. 

If, for some reason, you wish to avoid using mercurial or pip, you can install Colossus manually 
by downloading the current repository contents from the
`repository site <https://bitbucket.org/bdiemer/colossus/src/master/>`__. The disadvantage of this method 
is that it makes updating the code relatively cumbersome.

.. rubric:: Running unit tests

After installing Colossus, you should run its suite of unit tests to ensure the code works as 
expected. In python, execute::

    from colossus.tests import run_tests
    
The output should look something like this::

   test_get_version (colossus.tests.test_utils.TCVersions) ... ok
   test_versions (colossus.tests.test_utils.TCVersions) ... ok
   ...
   test_DK14ConstructorOuter (colossus.tests.test_halo_profile.TCDK14) ... ok
   test_DK14ConstructorWrapper (colossus.tests.test_halo_profile.TCDK14) ... ok
   
   ----------------------------------------------------------------------
   Ran 97 tests in 16.697s
   
   OK
       
If any errors occur, please send the output to the 
`author <http://www.benediktdiemer.com/contact/>`__.
