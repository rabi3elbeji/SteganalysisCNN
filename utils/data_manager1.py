##########################################################
############### Dataset management class #################
##########################################################

import cv2
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils


class DataManager(object):
	"""docstring for DataManager"""
	def __init__(self, data_path, image_height, image_width, nbr_channels, nbr_classes, test_size):
		super(DataManager, self).__init__()
		self.data_path 		= data_path
		self.image_height 	= image_height
		self.image_width 	= image_width
		self.nbr_channels 	= nbr_channels
		self.nbr_classes 	= nbr_classes
		self.test_size 	= test_size

	def get_data(self):
		# Tab fo store data
		img_data_list 	= []
		# Tab to store size of each class
		img_class_size 	= []
		# Tab to store name of each class
		img_class_names = []
		# List floders in dir 
		data_dir_list 	= os.listdir(self.data_path)
		# Loop folders
		for data_folder in data_dir_list:
			img_list = os.listdir(self.data_path+'/'+data_folder)
			img_class_size.append(len(img_list))
			img_class_names.append(data_folder)
			print ('Loaded the images of dataset-{}\n'.format(data_folder))
			# Loop images in folder and store the in array
			for img in img_list:
				input_img = cv2.imread(self.data_path+'/'+data_folder+'/'+img)
				input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
				input_img = cv2.resize(input_img, (self.image_height, self.image_width))
				img_data_list.append(input_img)
		# convert python array to a numpy array
		img_data = np.array(img_data_list)
		# Integer pixels to float
		img_data = img_data.astype('float32')
		img_data /= 255
		if self.nbr_channels == 1:
			img_data = np.expand_dims(img_data, axis=4)
		# Get the labels id for each image
		labels, class_label = self.prepare_labels(img_data, img_class_names, img_class_size)
		# Convert class labels to on-hot encoding
		labels_one_hot_encoded = np_utils.to_categorical(labels, self.nbr_classes)
		# Shuffle the data
		data, labels = shuffle(img_data, labels_one_hot_encoded, random_state=2)
		# Split the data (train & test) based on the percent of the test data
		data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=self.test_size, random_state=2)

		input_shape =img_data[0].shape
		return data_train, data_test, labels_train, labels_test, class_label


	def prepare_labels(self, img_data, img_class_names, img_class_size):
		# HashMap array to identify class name from label
		class_label = {}
		# declare an oray of one size=length-of-data
		num_of_samples = img_data.shape[0]
		labels = np.ones((num_of_samples,), dtype='int64')
		for x in xrange(0, len(img_class_size)):
			class_name = img_class_names[x]
			imgs_range_min  = img_class_size[x] * (x)
			imgs_range_max = img_class_size[x] * (x+1)
			#print(imgs_range_min, imgs_range_max, str(class_name) + " ==> label "+str(x))
			labels[imgs_range_min:imgs_range_max]=x
			class_label[x] = class_name

		return labels, class_label
		