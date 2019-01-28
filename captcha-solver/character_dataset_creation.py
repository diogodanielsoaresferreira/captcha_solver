import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

input_dir = "captchas"
output_dir = "character_extraction/all/"

files = os.listdir(input_dir)

file_number = {}

for filename in files:
	print(filename)

	img = cv2.imread(input_dir+"/"+filename, cv2.IMREAD_GRAYSCALE)

	# Otsu thresholding
	ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	kernel = np.ones((3,3), np.uint8)
	dilation2 = cv2.dilate(th2, kernel, iterations=1)
	erosion2 = cv2.erode(dilation2, kernel, iterations=1)

	kernel = np.ones((3,1), np.uint8)
	dilation2 = cv2.dilate(erosion2, kernel, iterations=1)

	#Get the individual letters.
	x, y, w, h = 30, 12, 21, 38
	for i in range(5):
		crop_img = dilation2[y:y+h, x:x+w]
		x += w

		if not os.path.exists(output_dir+"/"+filename[i]):
			os.makedirs(output_dir+"/"+filename[i])
		
		if filename[i] in file_number:
			file_number[filename[i]] += 1
		else:
			file_number[filename[i]] = 1

		cv2.imwrite(output_dir+"/"+filename[i]+"/"+str(file_number[filename[i]])+".png", crop_img)
		
