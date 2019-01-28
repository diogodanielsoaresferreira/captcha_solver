import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

input_dir = "character_extraction/train/"
output_dir = "character_extraction/train/generated_images"

rotations = [-10, 0, 10]
shift_ammount_x = [-3, 0, 3]
shift_ammount_y = [-3, 0, 3]

files = os.listdir(input_dir)

for character in files:
	for filename in os.listdir(input_dir+"/"+character):
		images = []

		print(filename)

		img = cv2.imread(input_dir+"/"+character+"/"+filename, cv2.IMREAD_GRAYSCALE)

		# Shape of image in terms of pixels. 
		(rows, cols) = img.shape[:2] 
	  	
		fig = plt.figure()
		i=1
		for rotation in rotations:
			for shift_x in shift_ammount_x:
				for shift_y in shift_ammount_y:
					rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
					res = cv2.warpAffine(img, rotation_matrix, (cols, rows), borderValue=(255,255,255))
					translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
					res = cv2.warpAffine(res, translation_matrix, (cols, rows), borderValue=(255,255,255))
					images.append(res)
					plt.subplot(9, 3, i)
					plt.axis('off')
					i+=1
					plt.imshow(res, 'gray')

		plt.show()


		exit(0)
		if not os.path.exists(output_dir+"/"+character):
			os.makedirs(output_dir+"/"+character)


		for i in range(len(images)):
			cv2.imwrite(output_dir+"/"+character+"/"+str(filename)+"_"+str(rotations[((i//len(shift_ammount_y))//len(shift_ammount_x))%len(rotations)])+"_"+str(shift_ammount_x[(i//len(shift_ammount_y))%len(shift_ammount_x)])+"_"+str(shift_ammount_y[i%len(shift_ammount_y)])+".png", images[i])
