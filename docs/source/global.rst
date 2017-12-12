============================
Global settings and defaults
============================

The global settings used in Colossus are split into two modules. The settings module contains
user-defined settings that control how Colossus runs and interacts with the system. In contrast,
the defaults module contains default values for physical parameters that influence the output of
numerous Colossus routines, as well as standard models used for physical quantities (such as the 
default concentration module). For physical constants, please see the :mod:`~utils.constants`
module.

--------
Settings
--------

.. automodule:: settings
    :members:

--------
Defaults
--------

.. automodule:: defaults
    :members:
