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

class StorageUser():
	"""
	A storage user object allows access to persistent and non-persistent storage.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	module: str
		The name of the module to which this user belongs. This name determines the cache sub-
		directory where files will be stored.
	persistence: str
		A combination of 'r' and 'w', e.g. 'rw' or '', indicating whether the storage is read 
		and/or written from and to disk.
	func_name: function
		A function that takes no parameters and returns the name of the user class.
	func_hashstring: function
		A function that takes no parameters and returns a unique string identifying the user class
		and any of its properties that, if changed, should trigger a resetting of the storage. If 
		the hash string changes, the storage is emptied.
	func_changed: function
		A function that takes no parameters and will be called if the hash string has been found to
		have changed (see above).
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
		"""
		Get a unique string from the user class and convert it to a hash.
		
		Returns
		-------------------------------------------------------------------------------------------
		hash: str
			A string that changes if the input string is changed, but can be much shorter than the 
			input string.
		"""
			
		hashable_string = self.func_hashstring()
		hash = hashlib.md5(hashable_string.encode()).hexdigest()
		
		return hash

	###############################################################################################

	def getUniqueFilename(self):
		"""
		Create a unique filename for this storage user.
		
		Returns
		-------------------------------------------------------------------------------------------
		filename: str
			A filename that is unique to this module, storage user name, and the properties of the 
			user as encapsulated in its hashable string.
		"""
					
		return self.cache_dir + self.func_name() + '_' + self.getHash()
	
	###############################################################################################
	
	def checkForChangedHash(self):
		"""
		Check whether the properties of the user class have changed.
		
		Returns
		-------------------------------------------------------------------------------------------
		has_changed: bool
			Returns True if the hash has changed compared to the last stored hash.
		"""
			
		hash_new = self.getHash()
		has_changed = (hash_new != self.hash_current)
		
		return has_changed
	
	###############################################################################################

	def resetStorage(self):
		"""
		Reset the storage arrays and load persistent storage from file.
		"""
			
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

	def storeObject(self, object_name, object_data, persistent = True):
		"""
		Save an object in memory and/or file storage.

		The object is written to a dictionary in memory, and also to file if ``persistent == True``
		(unless persistence does not contain 'w'). 

		Parameters
		-------------------------------------------------------------------------------------------
		object_name: str
			The name of the object by which it can be retrieved later.
		object_data: any
			The object; can be any picklable data type.
		persistent: bool
			If true, store this object on disk (if persistence is activated globally).
		"""
	
		if persistent:
			self.storage_pers[object_name] = object_data
			
			if self.persistence_write:
				filename_pickle = self.getUniqueFilename()
				output_file = open(filename_pickle, "wb")
				pickle.dump(self.storage_pers, output_file, pickle.HIGHEST_PROTOCOL)
				output_file.close()  

		else:
			self.storage_temp[object_name] = object_data

		return
		
	###############################################################################################

	def getStoredObject(self, object_name, interpolator = False, inverse = False, path = None,
					store_interpolator = True, store_path_data = True):
		"""
		Retrieve a stored object from memory or file.

		If an object is already stored in memory, return it. If not, try to load it from file, 
		otherwise return None. If the object is a 2-dimensional table, this function can also 
		return an interpolator.
		
		If the ``path`` parameter is passed, the file is loaded from that file path.
		
		Parameters
		-------------------------------------------------------------------------------------------
		object_name: str
			The name of the object to be loaded.
		interpolator: bool
			If True, return a spline interpolator instead of the underlying table.
		inverse: bool
			Return an interpolator that gives x(y) instead of y(x).
		path: str
			If not None, data is loaded from this file path (unless it has already been loaded, in
			which case it is found in memory).
		store_interpolator: bool
			If True (the default), an interpolator that has been created is temporarily stored so
			that it does not need to be created again.
		store_path_data: bool
			If True (the default), data loaded from a file defined by path is stored temporarily
			so that it does not need to be loaded again.
	
		Returns
		-------------------------------------------------------------------------------------------
		object_data: any
			Returns the loaded object (any pickleable data type), or a 
			scipy.interpolate.InterpolatedUnivariateSpline interpolator object, or None if no 
			object was found.
		"""
		
		# -----------------------------------------------------------------------------------------
		
		def tryTxtLoad(self, read_path):
			
			object_data = None
			if not self.persistence_read:
				return None
	
			if read_path is not None:
				if os.path.exists(read_path):
					object_data = np.loadtxt(read_path, usecols = (0, 1),
										skiprows = 0, comments = '#', unpack = True)
				else:
					raise Exception('File %s not found.' % (read_path))
							
			return object_data

		# -----------------------------------------------------------------------------------------
		
		# First, check for changes in the hash. If changes are detected, first call the user's 
		# change callback function and then reset the storage.
		if self.checkForChangedHash():
			self.func_changed()
			self.resetStorage()
			
		# Compute object name. If the object contains a file path, we need to isolate 
		object_id = object_name
		if interpolator:
			object_id += '_interpolator'
		if inverse:
			object_id += '_inverse'

		# Find the object. There are multiple possibilities:
		# - Check for the exact object the user requested (the object_id)
		#   - Check in persistent storage
		#   - Check in temporary storage (where interpolator / inverse objects live)
		#   - Check in user text file (where the path was given)
		# - Check for the raw object (the object_name)
		#   - Check in persistent storage
		#   - Check in temporary storage (where user-defined, pre-loaded objects live)
		#   - Check in user text files (where the path was given)
		#  - Convert to the exact object, store in temporary storage
		# - If all fail, return None

		object_data = None
		if object_id in self.storage_pers:	
			object_data = self.storage_pers[object_id]
		
		elif object_id in self.storage_temp:
			object_data = self.storage_temp[object_id]

		elif not interpolator:
			object_data = tryTxtLoad(self, path)
			if (object_data is not None) and store_path_data:
				self.storage_temp[object_id] = object_data

		# We could not find the object ID anywhere. This can have two reasons: the object does
		# not exist, or we must transform an existing object.
		if interpolator and object_data is None:
			
			# Try to find the object to transform.
			object_raw = None
			
			if object_name in self.storage_pers:	
				object_raw = self.storage_pers[object_name]
			elif object_name in self.storage_temp:
				object_raw = self.storage_temp[object_name]
			else:
				object_raw = tryTxtLoad(self, path)

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
				if store_interpolator:
					self.storage_temp[object_id] = object_data
		
		return object_data

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
