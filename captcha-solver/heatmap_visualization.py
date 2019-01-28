from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
import cv2


model = load_model('model.h5')


important_numbers = ['2', '3', '4', '5', '6', '7', '8']
important_chars = ['b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y']

images = []

ch = 0

for character in important_numbers+important_chars:
	img_path = './character_extraction/test/' + character + '/4.png'
	img = image.load_img(img_path, target_size=(38, 21))
	x = image.img_to_array(img)

	x = np.expand_dims(x, axis=0)
	x = np.mean(x, axis=3)
	x = np.expand_dims(x, axis=3)

	preds = model.predict(x)

	neuron = np.argmax(preds[0])

	output = model.output[:, neuron]
	last_conv_layer = model.get_layer('conv2d_4')

	grads = K.gradients(output, last_conv_layer.output)[0]
	pooled_grads = K.mean(grads, axis=(0, 1, 2))

	iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
	pooled_grads_value, conv_layer_output_value = iterate([x])

	for i in range(512):
		conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

	heatmap = np.mean(conv_layer_output_value, axis=-1)

	heatmap = np.maximum(heatmap, 0)
	heatmap /= np.max(heatmap)

	img = cv2.imread(img_path)
	heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
	heatmap = np.uint8(255 * heatmap)
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	superimposed_img = heatmap * 0.5 + img * 0.5
	superimposed_img = cv2.resize(superimposed_img, (img.shape[1]*10, img.shape[0]*10))

	if ch%5 == 0:
		images.append(superimposed_img)
	else:
		images[ch//5] = np.hstack((images[ch//5], superimposed_img))

	ch += 1

images[3] = np.hstack((images[3], np.zeros((380, 210, 3), np.uint8)))
images = np.vstack((images[0], images[1], images[2], images[3]))
cv2.imwrite('heatmaps.jpg', images)

