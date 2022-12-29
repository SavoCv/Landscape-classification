from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
from keras import Sequential, layers
from keras.losses import SparseCategoricalCrossentropy
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


main_path = './Landscape Classification/Landscape Classification/Training Data'
train_path = './Landscape Classification/Landscape Classification/Training Data'
test_path = './Landscape Classification/Landscape Classification/Testing Data'

img_size = [64, 64]
batch_size = 32

ulaz_trening = image_dataset_from_directory(main_path,  subset='training', validation_split=0.2,
                                            seed=52, batch_size=batch_size,
                                            image_size=img_size)

ulaz_test = image_dataset_from_directory(main_path, subset='validation', validation_split=0.2, seed=52,
                                         batch_size=batch_size,
                                         image_size=img_size)

classes = ulaz_trening.class_names

for img, lab in ulaz_trening.take(1):
    plt.figure()
    for k in range(10):
        plt.subplot(2, 5, k + 1)
        plt.imshow(img[k].numpy().astype('uint8'))
        plt.title(classes[lab[k]])
    plt.show()

data_aug = Sequential([
    layers.RandomFlip('horizontal', input_shape=(img_size[0], img_size[1], 3)),
    layers.RandomRotation(0.0625),
    layers.RandomZoom(0.3)
])

# data_aug.build()

for img, lab in ulaz_trening.take(1):
    plt.figure()
    for k in range(10):
        img_aug = data_aug(img)
        plt.subplot(2, 5, k + 1)
        plt.imshow(img_aug[0].numpy().astype('uint8'))
        plt.title(classes[lab[k]])
    plt.show()

model = Sequential([
    data_aug,

    layers.Rescaling(1./255),

    layers.Conv2D(16, 3, activation='relu',  # input_shape=(64, 64, 3),
                  padding='same'),
    layers.MaxPooling2D(2, strides=2),

    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2, strides=2),

    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2, strides=2),

    layers.Dropout(0.4),

    layers.Flatten(), # da ispravi sliku u niz

    layers.Dense(256, activation='relu'),
    layers.Dense(len(classes), 'softmax')
])

model.summary()

model.compile('adam', loss=SparseCategoricalCrossentropy(), metrics='accuracy')
history = model.fit(ulaz_trening, epochs=10, validation_data=ulaz_test) # ne mora batch_size jer je definisan vec gore

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

pred = np.array([])
labels = np.array([])
for img, lab in ulaz_test:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

cm = confusion_matrix(labels, pred)
acc = accuracy_score(labels, pred)
print(cm)
print(acc)

