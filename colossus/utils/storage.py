###################################################################################################
#
# storage.py 	        (c) Benedikt Diemer
#     				    	benedikt.diemer@cfa.harvard.edu
#
###################################################################################################

"""
This module provides both non-persistent and persistent storage and interpolation to be used by any
module in colossus. Both reading and writing can be turned on and off by the user. Each user of 
the storage module receives their own storage space and a uniquely identifying hash code that can
be used to detect changes that make it necessary to reset the storage, for example changes in 
physical parameters to a model. 

---------------------------------------------------------------------------------------------------
Basic usage
---------------------------------------------------------------------------------------------------

The most important step is to set up this storage user class::

	storageUser = storage_unit.StorageUser('myModule', 'rw', self.getName, 
					self.getHashableString, self.reportChanges)
									
where 'rw' indicates that the storage should be both written and read from persistent files, and 
the rest of the parameters are functions that return a name for the user class (e.g. planck15 for
a cosmology), getHashableString provides a uniquely identifying string for the user class, and
reportChanges can be used to react when a change in the hash is detected. Once the user is set up,
we can add objects easily::

	norm = 5.2
	storageUser.storeObject('normalization', norm, persistent = True)

where the persistent parameter determines that this object will be written to disk as part of a 
pickle and loaded next time the same user class (same name and same hash code) is instantiated. 
Objects are retrieved similarly::

	my_norm = storageUser.getStoredObject('normalization')
	
The storage_unit module offers native support for interpolation tables. For example, if we have 
stored a table of variables x and y, we can get a spline interpolator for y(x) or even a reverse
interpolator for x(y) by calling::

	interp_y_of_x = storageUser.getStoredObject('xy', interpolator = True)
	interp_x_of_y = storageUser.getStoredObject('xy', interpolator = True, inverse = True)

The getStoredObject() returns None if no object is found.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

import os
import hashlib
import pickle
import warnings
import numpy as np
import scipy.interpolate

from colossus import settings
from colossus.utils import utilities

###################################################################################################

# name = name for this user, doesn't have to be unique but makes file id easier
# func_hashstring = return string that identifies the user uniquely
# func_changed = execute when change in hash detected

class StorageUser():
	"""
	A storage user object allows access to persistent and non-persistent storage.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	module: str
		The name of the module to which this user belongs. This name determines the cache sub-
		directory where files will be stored.
	storage: str
		A combination of 'r' and 'w', e.g. 'rw' or '', indicating whether the storage is read 
		and/or written from and to disk.
		
	"""
	
	def __init__(self, module, persistence, func_name, func_hashstring, func_changed):

		self.module = module
		self.func_name = func_name
		self.func_hashstring = func_hashstring
		self.func_changed = func_changed

		if persistence in [True, False]:
			raise DeprecationWarning('The persistence parameter is not boolean but a combination of r and w, such as "rw".')

		for l in persistence:
			if not l in ['r', 'w']:
				raise Exception('The persistence parameter contains an unknown letter %c.' % l)
		self.persistence_read = ('r' in persistence)
		self.persistence_write = ('w' in persistence)

		if self.persistence_read or self.persistence_write:
			self.cache_dir = getCacheDir(module = self.module)
		
		self.resetStorage()
	
		return

	###############################################################################################

	def getHash(self):
		
		hashable_string = self.func_hashstring()
		hash = hashlib.md5(hashable_string.encode()).hexdigest()
		
		return hash

	###############################################################################################

	# Create a file name that is unique to this StorageUser. The hash must encode all necessary
	# information, but a name for the user is added to make it easier to identify the files with a 
	# user.

	def getUniqueFilename(self):
		
		return self.cache_dir + self.func_name() + '_' + self.getHash()
	
	###############################################################################################
	
	def checkForChangedHash(self):

		hash_new = self.getHash()
		has_changed = (hash_new != self.hash_current)
		
		return has_changed
	
	###############################################################################################

	# Load stored objects. This function is called during the __init__() routine, and if a change
	# in the storage user is detected.

	def resetStorage(self):

		# Reset the test hash and storage containers. There are two containes, one for objects
		# that are stored in a pickle file, and one for those that will be discarded when the 
		# class is destroyed.
		self.hash_current = self.getHash()
		self.storage_pers = {}
		self.storage_temp = {}
		
		# Check if there is a persistent object storage file. If so, load its contents into the
		# storage dictionary. We only load from file if the user has not switched of storage, and
		# if the user has not switched off interpolation.
		#
		# Loading the pickle can go wrong due to python version differences, so we generously
		# catch any exceptions that may occur and simply delete the file in that case.
		if self.persistence_read:
			filename_pickle = self.getUniqueFilename()
			if os.path.exists(filename_pickle):
				try:
					input_file = open(filename_pickle, "rb")
					self.storage_pers = pickle.load(input_file)
					input_file.close()
				except Exception:
					warnings.warn('Encountered file error while reading cache file. This usually \
						happens when switching between python 2 and 3. Deleting cache file.')
					try:
						os.remove(filename_pickle)
					except Exception:
						pass
		
		return
	
	###############################################################################################

	# Permanent storage system for objects such as 2-dimensional data tables. If an object is 
	# already stored in memory, return it. If not, try to load it from file, otherwise return None.
	# Certain operations can already be performed on certain objects, so that they do not need to 
	# be repeated unnecessarily, for example:
	#
	# interpolator = True	Instead of a 2-dimensional table, return a spline interpolator that can
	#                       be used to evaluate the table.
	# inverse = True        Return an interpolator that gives x(y) instead of y(x)
	
	def getStoredObject(self, object_name, interpolator = False, inverse = False):

		# First, check for changes in the hash. If changes are detected, first call the user's 
		# change callback function and then reset the storage.
		if self.checkForChangedHash():
			self.func_changed()
			self.resetStorage()
			
		# Compute object name
		object_id = object_name
		if interpolator:
			object_id += '_interpolator'
		if inverse:
			object_id += '_inverse'

		# Find the object. There are multiple possibilities:
		# - Check for the exact object the user requested (the object_id)
		#   - Check in persistent storage
		#   - Check in temporary storage (where interpolator / inverse objects live)
		#   - Check in user text files
		# - Check for the raw object (the object_name)
		#   - Check in persistent storage
		#   - Check in user text files
		#   - Convert to the exact object, store in temporary storage
		# - If all fail, return None

		if object_id in self.storage_pers:	
			object_data = self.storage_pers[object_id]
		
		elif object_id in self.storage_temp:	
			object_data = self.storage_temp[object_id]

		elif self.persistence_read and os.path.exists(self.cache_dir + object_id):
			object_data = np.loadtxt(self.cache_dir + object_id, usecols = (0, 1),
									skiprows = 0, unpack = True)
			self.storage_temp[object_id] = object_data
			
		else:

			# We could not find the object ID anywhere. This can have two reasons: the object does
			# not exist, or we must transform an existing object.
			
			if interpolator:
				
				# Try to find the object to transform. This object CANNOT be in temporary storage,
				# but it can be in persistent or user storage.
				object_raw = None
				
				if object_name in self.storage_pers:	
					object_raw = self.storage_pers[object_name]
		
				elif self.persistence_read and os.path.exists(self.cache_dir + object_name):
					object_raw = np.loadtxt(self.cache_dir + object_name, usecols = (0, 1),
									skiprows = 0, unpack = True)

				if object_raw is None:
					
					# We cannot find an object to convert, return none.
					object_data = None
				
				else:
					
					# Convert and store in temporary storage.
					if inverse: 
						
						# There is a subtlety: the spline interpolator can't deal with decreasing 
						# x-values, so if the y-values are decreasing, we reverse their order.
						if object_raw[1][-1] < object_raw[1][0]:
							object_raw = object_raw[:, ::-1]
						
						object_data = scipy.interpolate.InterpolatedUnivariateSpline(object_raw[1],
																					object_raw[0])
					else:
						object_data = scipy.interpolate.InterpolatedUnivariateSpline(object_raw[0],
																					object_raw[1])
					self.storage_temp[object_id] = object_data
						
			else:
							
				# The object is not in storage at all, and cannot be generated; return none.
				object_data = None
				
		return object_data
	
	###############################################################################################

	# Save an object in memory and file storage. If persistent == True, this object is written to 
	# file storage (unless persistence != 'w'), and will be loaded the next time the same cosmology
	# is loaded. If persistent == False, the object is stored non-persistently.
	#
	# Note that all objects are reset if the cosmology changes. Thus, this function should be used
	# for ALL data that depend on cosmological parameters.
	
	def storeObject(self, object_name, object_data, persistent = True):

		if persistent:
			self.storage_pers[object_name] = object_data
			
			if self.persistence_write:
				# Store in file. We do not wish to save the entire storage dictionary, as there might be
				# user-defined objects in it.
				filename_pickle = self.getUniqueFilename()
				output_file = open(filename_pickle, "wb")
				pickle.dump(self.storage_pers, output_file, pickle.HIGHEST_PROTOCOL)
				output_file.close()  

		else:
			self.storage_temp[object_name] = object_data

		return
	
###################################################################################################

def getCacheDir(module = None):
	"""
	Get a directory for the persistent caching of data. The function attempts to locate the home 
	directory and (if necessary) create a .colossus sub-directory. In the rare case where that 
	fails, the location of this code file is used as a base directory.

	Parameters
	---------------------------
	module: string
		The name of the module that is requesting this cache directory. Each module has its own
		directory in order to avoid name conflicts.
	
	Returns
	-------
	path : string
		The cache directory.
	"""
	
	if settings.BASE_DIR is None:
		base_dir = utilities.getHomeDir()
		if base_dir is None:
			base_dir = utilities.getCodeDir()
	else:
		base_dir = settings.BASE_DIR
		
	cache_dir = base_dir + '/.colossus/cache/'
	
	if module is not None:
		cache_dir += module + '/'

	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)
	
	return cache_dir

###################################################################################################
