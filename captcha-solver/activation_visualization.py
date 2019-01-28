from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras import models
import matplotlib.pyplot as plt


model = load_model('model.h5')
model.summary()

img_path = './character_extraction/test/4/4.png'


img = image.load_img(img_path, target_size=(38, 21))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_gray = np.mean(img_tensor, axis=3)
img_tensor = np.expand_dims(img_gray, axis=3)
img_tensor /= 255.

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

layer_names = []
for layer in model.layers[:8]:
	layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
	n_features = layer_activation.shape[-1]

	height, weight = layer_activation.shape[1], layer_activation.shape[2]


	n_cols = n_features // images_per_row
	display_grid = np.zeros((height * n_cols, images_per_row * weight))


	for col in range(n_cols):
		for row in range(images_per_row):
			channel_image = layer_activation[0,:, :,col * images_per_row + row]
			channel_image -= channel_image.mean()
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * height : (col + 1) * height,
			row * weight : (row + 1) * weight] = channel_image

	scale = 1. / height
	plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
	plt.title(layer_name)
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.show()
