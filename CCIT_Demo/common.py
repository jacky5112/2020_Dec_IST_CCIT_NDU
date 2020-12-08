import matplotlib.pyplot as plt
import numpy as np

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def show_images_with_labels(images, labels, prediction, index, amount):
    f = plt.gcf()
    f.set_size_inches(10, 10)

    for i in range(0, amount):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(images[index], cmap='binary')
        title = 'label = {0}'.format(labels[index])

        if len(prediction) > 0:
            title += ',predict = {0}'.format(prediction[index])

        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        index += 1

    plt.show()
