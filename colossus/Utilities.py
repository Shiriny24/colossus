###################################################################################################
#
# Utilities.py 		        (c) Benedikt Diemer
#							University of Chicago
#     				    	bdiemer@oddjob.uchicago.edu
#
###################################################################################################

"""
Common routines for Colossus modules.
"""

import os
import numpy

###################################################################################################

def printLine():
	"""
	Print a line to the console.
	"""

	print('-------------------------------------------------------------------------------------')

	return

###################################################################################################

def getCacheDir():
	"""
	Get a directory for the persistent caching of data. Here, this directory is chosen to be the
	directory where this file is located. This directory obviously already exists.
	
	Returns
	-------
	path : string
		The cache directory.
	"""
	
	path = getCodeDir() + '/cache/'
	
	return path

###################################################################################################

def getCodeDir():
	"""
	Returns the path to this code file.
	
	Returns
	-------
	path : string
		The code directory.
	"""
	
	path = os.path.dirname(os.path.realpath(__file__))

	return path

###################################################################################################

def isArray(var):
	"""
	Tests whether a variable var is iterable or not.

	Parameters
	---------------------------
	var : array_like
		Variable to be tested.
	
	Returns
	-------
	is_array : boolean
		Whether var is a numpy array or not.
	"""
	
	try:
		dummy = iter(var)
	except TypeError:
		is_array = False
	else:
		is_array = True
		
	return is_array

###################################################################################################

def getArray(var):
	"""
	Convert a variable to a numpy array, whether it already is one or not.

	Parameters
	-----------------------------------------------------------------------------------------------
	var: array_like
		Variable to be converted to an array.
	
	Returns
	-----------------------------------------------------------------------------------------------
	var_ret: numpy array
		A numpy array with one or more entries.
	is_array: boolean
		Whether var is a numpy array or not.
	"""
		
	is_array = isArray(var)
	if is_array:
		var_ret = var
	else:
		var_ret = numpy.array([var])
		
	return var_ret, is_array 

###################################################################################################
