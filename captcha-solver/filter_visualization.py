from keras.models import load_model
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import cv2

model = load_model('model.h5')
model.summary()


def deprocess_image(x):
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	x += 0.5
	x = np.clip(x, 0, 1)

	x *= 255
	x = np.clip(x, 0, 255).astype('uint8')
	return x

def generate_pattern(layer_name, filter_index, height, width):
	layer_output = model.get_layer(layer_name).output
	loss = K.mean(layer_output[:, :, :, filter_index])
	grads = K.gradients(loss, model.input)[0]

	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
	
	iterate = K.function([model.input], [loss, grads])
	
	input_img_data = np.random.random((1, width, height, 1)) * 20 + 128.
	
	step = 1.
	for i in range(40):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * step
	
	img = input_img_data[0]
	return deprocess_image(img)



layer_name = 'conv2d_1'
height = 38
width = 21
margin = 5

results = np.zeros((8 * width + 7 * margin, 8 * height + 7 * margin, 3))
for i in range(8):
	for j in range(8):
		filter_img = generate_pattern(layer_name, i + (j * 8), height, width)
		horizontal_start = i * width + i * margin
		horizontal_end = horizontal_start + width
		vertical_start = j * height + j * margin
		vertical_end = vertical_start + height
		results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

big = cv2.resize(results, (0,0), fx=2, fy=2) 
cv2.imwrite(layer_name+"_filter_activation.png", big)

