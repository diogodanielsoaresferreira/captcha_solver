import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

import numpy as np
import cv2

output_dir = "emnist_dataset"
new_width, new_height = 21, 38

important_numbers = [2, 3, 4, 5, 6, 7, 8]
important_chars = ['b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y']
important_chars_numbers = [37, 12, 38, 39, 40, 41, 22, 43, 25, 32, 33, 34]


def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image


train = pd.read_csv('emnist_raw_dataset/emnist-balanced-train.csv', header=None)
test = pd.read_csv('emnist_raw_dataset/emnist-balanced-test.csv', header=None)


train_data = train.iloc[:, 1:]
train_labels = train.iloc[:, 0]
test_data = test.iloc[:, 1:]
test_labels = test.iloc[:, 0]

train_data = train_data.values
train_labels = train_labels.values
test_data = test_data.values
test_labels = test_labels.values
del train, test

train_data = np.apply_along_axis(rotate, 1, train_data)
test_data = np.apply_along_axis(rotate, 1, test_data)
i = 1
for image, label in zip(train_data, train_labels):
	if label in important_numbers or int(label) in important_chars_numbers:
		im = Image.fromarray(image.astype('uint8'))
		im = im.resize((new_width,new_height))
		im = ImageOps.invert(im)
		if label in important_numbers:
			if not os.path.exists(output_dir+"/train/"+str(label)):
				os.makedirs(output_dir+"/train/"+str(label))
			im.save(output_dir+"/train/"+str(label)+"/"+str(i)+".png")
		else:
			character = str(important_chars[important_chars_numbers.index(int(label))])
			if not os.path.exists(output_dir+"/train/"+character):
				os.makedirs(output_dir+"/train/"+character)
			im.save(output_dir+"/train/"+character+"/"+str(i)+".png")

		i += 1

for image, label in zip(test_data, test_labels):
	if label in important_numbers or int(label) in important_chars_numbers:
		im = Image.fromarray(image.astype('uint8'))
		im = im.resize((new_width,new_height))
		im = ImageOps.invert(im)
		if label in important_numbers:
			if not os.path.exists(output_dir+"/test/"+str(label)):
				os.makedirs(output_dir+"/test/"+str(label))
			im.save(output_dir+"/test/"+str(label)+"/"+str(i)+".png")
		else:
			character = str(important_chars[important_chars_numbers.index(int(label))])
			if not os.path.exists(output_dir+"/test/"+character):
				os.makedirs(output_dir+"/test/"+character)
			im.save(output_dir+"/test/"+character+"/"+str(i)+".png")

		i += 1