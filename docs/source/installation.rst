============
Installation
============

You can install Colossus either using the common python package manager pip or by downloading the code directly. 

.. rubric:: Requirements

Colossus is compatible with both Python 2.7 and Python 3.x. However, the code is developed and mostly tested in Python 3, which is thus the recommended version. Colossus requires the following standard packages:

* `numpy <http://www.numpy.org/>`_
* `scipy <https://www.scipy.org/>`_
* `six <https://pypi.org/project/six/>`_

Some tutorial notebooks also use other packages, for example matplotlib for plotting.

.. rubric:: Package installation

The easiest way to install Colossus is by executing

.. code:: shell

    pip install colossus

You might need to prefix this command with ``sudo``. To update the code, execute

.. code:: shell

    pip install --upgrade colossus

If the numpy and scipy packages are not already installed, you can similarly install them with the pip command. For more information, please see the `pip documentation <https://packaging.python.org/tutorials/installing-packages/>`_.

.. rubric:: Repository installation

If you want to edit the code, you might prefer to clone the public BitBucket repository https://bitbucket.org/bdiemer/colossus by executing

.. code:: shell

   hg clone https://bitbucket.org/bdiemer/colossus

For this method, you will need the version control system Mercurial (hg), which you can download `here <http://mercurial.selenic.com/>`_. You can update the code by pulling changes from the repository,

.. code:: shell

   hg pull
   hg up

You will also need to manually include Colossus in your ``$PYTHONPATH`` variable, for example by adding this command to your shell's initialization script (e.g., ``bashrc``):

.. code:: shell
   
   export PYTHONPATH=$PYTHONPATH:/Users/me/code/colossus

where the path is, of course, replaced with the location of Colossus on your system. 

If, for some reason, you wish to avoid using mercurial or pip, you can install Colossus manually by downloading the current repository contents as a `zip file <https://bitbucket.org/bdiemer/colossus/downloads/>`_. The disadvantage of this method is that it makes updating the code relatively cumbersome.

.. rubric:: Running unit tests

After installing Colossus, you should run its suite of unit tests to ensure the code works as expected. In python, execute::

    from colossus.tests import run_tests
    
The output should look something like this::

   test_home_dir (colossus.tests.test_utils.TCGen) ... ok
   test_Ez (colossus.tests.test_cosmology.TCComp) ... ok
   ...
   test_DK14ConstructorOuter (colossus.tests.test_halo_profile.TCDK14) ... ok
   test_DK14ConstructorWrapper (colossus.tests.test_halo_profile.TCDK14) ... ok
   
   ----------------------------------------------------------------------
   Ran 85 tests in 6.788s
   
   OK
           
If any errors occur, please send the output to the `author <http://www.benediktdiemer.com/contact/>`_.
