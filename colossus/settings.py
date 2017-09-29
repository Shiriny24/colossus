###################################################################################################
#
# settings.py 	        (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
Global settings used across colossus.
"""

###################################################################################################
# PATH TO COLOSSUS FILES
###################################################################################################

BASE_DIR = None
"""The directory in which colossus stores files, e.g. cache files (without a backslash at the end). 
If None, the home directory is used. If a home directory cannot be identified (an extremely rare 
case), the code directory is used."""

###################################################################################################
# PERSISTENT STORAGE
###################################################################################################

PERSISTENCE = 'rw'
"""This parameter determines whether colossus stores persistent files such as the cosmology cache.
The parameter can take on any combination of read ('r') and write ('w'), such as 'rw' (read and 
write, the default), 'r' (read only), 'w' (write only), or '' (no persistence). Note that this 
parameter is used as a default, but can still be changed for individual colossus modules or 
objects, such as cosmology objects."""
