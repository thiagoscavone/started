# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:27:39 2023


"""

import pandas as pd
import pickle
from PIL import Image
#from google.colab import drive
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import zipfile
import os
import warnings
import shutil
import seaborn as sns
import random
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing




def unzip_data(filename):
  """
  Unzips filename into the current working directory.
  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()

def plot_value_count(df, column):
	"""
	plots value count of dataframe column
	"""
	sns.set(style='darkgrid')
	plt.figure(figsize= (20,10))
	sns.countplot(x=column, data=df, order = df[column].value_counts().index)
	plt.xticks(rotation=90)
	plt.show()
    
    
    
#data is avalaible here - https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset
unzip_data = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"

#1. Preprocessing
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #Splitting dataset for validation
    validation_split = 0.2,
    #normalising pixel values
    rescale = 1./255., horizontal_flip=True
)

testDatagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #normalising pixel values
    rescale = 1./255.
)


train_dataset = datagen.flow_from_directory(
    data_path,
    # Specify a size that all images should be in
    target_size = (224,224),
    # Number of Batches
    batch_size = 8,
    # Specifying subsets to be implemented upon
    subset = 'training',
    # Shuffle the dataset before batching to avoid locality bias
    shuffle = True,
)
