================================= 
FAQ and troubleshooting
=================================

Below are some commonly asked questions and explanations.

.. rubric:: I cannot import colossus.

If commands such as ``from colossus.tests import run_tests`` fail, your installation probably did 
not succeed in the first place. If you have cloned the repository directly, make sure Colossus is
included in your ``$PYTHONPATH``. Please follow the steps described on the :doc:`installation` 
page.

.. rubric:: While running the unit tests, I get "no attribute 'solve_ivp'" error

If you get an error like this while running the unit tests::

    dic = scipy.integrate.solve_ivp(derivatives_G, (a_min, a_max), [1.0, 0.0],
    AttributeError: 'module' object has no attribute 'solve_ivp'
    
then you almost certainly need to update your scipy distribution (and probably numpy, while you're
at it).

.. rubric:: I am getting errors because the cosmology is not set

A common problem can be error messages like these::

    File "/some/dir/colossus/cosmology/cosmology.py", line 2706, in getCurrent
        raise Exception('Cosmology is not set.')
    Exception: Cosmology is not set.

This message means that you need to set a cosmology before executing a given function. Colossus
deliberately does not set a default cosmology to make sure the user is aware of the cosmological
parameters that are being used (which influence virtually all Colossus functions).