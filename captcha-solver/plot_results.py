import json
from matplotlib import pyplot as plt

test_number = 9
input_file = "test_results/character/trainHistoryDict_" + str(test_number)

with open(input_file) as f:
	data = json.load(f)

print(max(data["val_acc"]))
plt.title("Training and testing accuracy")
plt.plot(data["acc"], label="Train accuracy")
plt.plot(data["val_acc"], label="Test accuracy")
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.legend()
#plt.xlim(right=400)
plt.show()