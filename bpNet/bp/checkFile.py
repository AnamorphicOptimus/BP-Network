
# Check if there is a data folder
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import subprocess
import os
import numpy
from six.moves import urllib

def maybe_download(filename, data_dir, SOURCE_URL):
	"""Download the data from Yann's website, unless it's already here."""
	filepath = os.path.join(data_dir, filename)
	if not os.path.exists(filepath):
		filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
		statinfo = os.stat(filepath)
		print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

def uzip_data(target_path):
	# uzip mnist data
	cmd = ['gzip', '-d', target_path]
	print('Unzip ', target_path)
	subprocess.call(cmd)1

def checkFile(testFilename):
    data_dir = os.path.join(os.path.abspath(os.getcwd()),testFilename)
    print(data_dir)
    if not os.path.exists(data_dir):
        print("You don't have this file")
        os.mkdir(data_dir)
        SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
        data_keys = ['train-images-idx3-ubyte.gz', 
    				'train-labels-idx1-ubyte.gz', 
    				't10k-images-idx3-ubyte.gz', 
    				't10k-labels-idx1-ubyte.gz']
        for key in data_keys:
            maybe_download(key, data_dir, SOURCE_URL)
    		
            uziped_data_keys = ['train-images-idx3-ubyte', 
    						'train-labels-idx1-ubyte', 
    						't10k-images-idx3-ubyte', 
    						't10k-labels-idx1-ubyte']
        for key in uziped_data_keys:
            if os.path.isfile(os.path.join(data_dir, key)):
                print("[warning...]", key, "already exist.")
            else:
                target_path = os.path.join(data_dir, key)
                uzip_data(target_path)
    else:
        print("You already have this File!")
			
		
			
	
		








	