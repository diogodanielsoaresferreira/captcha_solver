import os
import shutil


def split_test_validation_train_data(data, validation_percentage, test_percentage):
	n_validation = int(len(data)*validation_percentage)
	n_test = int(len(data)*test_percentage)

	test_data = data[:n_test]
	validation_data = data[n_test:n_test+n_validation]
	train_data = data[n_test+n_validation:]

	return train_data, validation_data, test_data


def copy_images(old_path, new_path, images):
	for image in images:

		if not os.path.exists(new_path):
			os.makedirs(new_path)
		
		shutil.copy(old_path+"/"+image, new_path+"/"+image)


input_dir = "character_extraction/all"
output_dir = "character_extraction"

validation_percentage = 0.2
test_percentage = 0.2

folders = os.listdir(input_dir)


for folder in folders:
	class_path = input_dir+"/"+folder

	class_files = os.listdir(class_path)
	train_data, validation_data, test_data = split_test_validation_train_data(class_files, validation_percentage, test_percentage)
	copy_images(class_path, output_dir+"/train/"+folder, train_data)
	copy_images(class_path, output_dir+"/validation/"+folder, validation_data)
	copy_images(class_path, output_dir+"/test/"+folder, test_data)
