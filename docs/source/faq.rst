================================= 
FAQ and troubleshooting
=================================

Below are some commonly asked questions and explanations.

.. rubric:: I cannot import Colossus.

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

.. rubric:: How can I tell what version of Colossus I have?

Just execute this code::

    from colossus import version
    print(version.__version__)

If this function fails, you are using a very old version and an update would probably be in order!   

.. rubric:: I am getting errors because the cosmology is not set

A common problem can be error messages like these::

    File "/some/dir/colossus/cosmology/cosmology.py", line 2706, in getCurrent
        raise Exception('Cosmology is not set.')
    Exception: Cosmology is not set.

This message means that you need to set a cosmology before executing a given function. Colossus
deliberately does not set a default cosmology to make sure the user is aware of the cosmological
parameters that are being used (which influence virtually all Colossus functions).

.. rubric:: Can I use Colossus to analyze my simulation data?

Probably not. Colossus may help with your analysis as it pertains to cosmology, halos, and 
large-scale structure, but it is not designed to work with any specific type of simulation data.

.. rubric:: How accurate is Colossus?

It depends: there is no set accuracy that applies to all functions in Colossus. Many calculations
are, in principle, accurate to machine precision. For most functions, there are no analytical 
solutions, so it is hard to give an absolute accuracy. 

However, those modules of Colossus that are also implemented in other codes (especially the 
cosmology module) have been compared in great detail to codes such as 
`AstroPy <https://www.astropy.org/index.html>`__ and `CCL <https://github.com/LSSTDESC/CCL>`__. They 
agree to high precision for the basic cosmology functions. Numerous other functions have been 
validated against other codes and/or data in published papers.

An important consideration is whether a function returns the most exact result or an interpolated
version for performance. The latter happens frequently in the cosmology module, for example. You 
can turn this type of interpolation off if you are interested in accuracy over perfomance (see the
respective module documentations).

Finally, the accuracy of many functions is tested and stated explicitly in the 
`code paper <https://ui.adsabs.harvard.edu/abs/2018ApJS..239...35D/abstract>`__.

.. rubric:: Using a tabulated power spectrum causes weird errors

When using a tabulated power spectrum, you may get errors such as::

    ValueError: x must be strictly increasing
    
These errors occur when colossus tries to set up an interpolation table where the x-dimension is
not monotonic. For example, if a tabulated power spectrum cuts out at some wavenumber, the variance
will approach a constant which can throw the interpolator. In such cases, one can change the extent
of the cosmological interpolation tables, for example::

    R_min_sigma = 1E-3
    
means that Colossus will not try to interpolate sigma at smaller radii, which it normally would. 
When changing the interpolation tables in this way, make sure to delete old persistence files.
Such manipulations require some knowledge of the internal workings of Colossus. Feel free
to contact the developer.

.. rubric:: I am using non-LCDM cosmologies and get weird errors

When using ``de_model = 'w0wa'`` or ``de_model = 'user'``, the calculation of the growth factor
can fail due to overflow errors. These are typically caused because the dark energy component
grows exponentially in the future. For this reason, the default redshift range of all interpolation
tables is reduced in these cases. If such errors still occur, you can set ``interpolation = False``, 
but that reduces the functionality of the cosmology module somewhat. Instead, you can reduce the 
redshift range of the interpolation table, for example::

    cosmo = cosmology.setCosmology('myCosmo', params)
    cosmo.z_min = -0.1
    cosmo.z_min_compute = -0.2
    
Note that changes in the interpolation tables will cause untested changes in the behavior of the
interpolation.
    
.. rubric:: Can I reset all previously stored cosmological data?

Yes! Just go to the cache directory::

    ~/.colossus/cache/cosmology/

and delete all files. The next time you create a cosmology, a new file will be stored there.
