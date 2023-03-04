import random

from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
from keras import Sequential, layers
from keras.losses import SparseCategoricalCrossentropy
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import os
import keras_tuner as kt
import warnings


warnings.filterwarnings("ignore")

main_path = './Landscape Classification'
# train_path = './Landscape Classification/Landscape Classification/Training Data'
# test_path = './Landscape Classification/Landscape Classification/Testing Data'

classes = []


def show_samples():
    ulaz = image_dataset_from_directory(main_path)
    classes = ulaz.class_names
    num_samples = []
    cnt = 0
    for i in classes:
        num_samples += [i] * len(os.listdir(main_path + "\\" + i))
        cnt += 1
    # print(num_samples)
    plt.figure()
    plt.title("Grafik odbiraka razlicitih klasa")
    plt.hist(num_samples)
    plt.show()


def make_model(k):
    data_aug = Sequential([
        layers.RandomFlip('horizontal', input_shape=(img_size[0], img_size[1], 3)),
        layers.RandomRotation(0.0625),
        layers.RandomZoom(0.3)
    ])

    model = Sequential([
        data_aug,

        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 3, activation='relu',  # input_shape=(64, 64, 3),
                      padding='same'),
        layers.MaxPooling2D(2, strides=2),

        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2, strides=2),

        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2, strides=2),

        layers.Dropout(k),

        layers.Flatten(),  # da ispravi sliku u niz

        layers.Dense(256, activation='relu'),
        layers.Dense(len(classes), 'softmax')
    ])
    model.compile('adam', loss=SparseCategoricalCrossentropy(), metrics='accuracy')
    return model


def make_model_for_search(hp):
    k = hp.Int('dropout_prob', min_value=30, max_value=70, step=1)
    print("Proba ", k)
    return make_model(k/100.)



# show_samples()
# exit(0)

img_size = [64, 64]
batch_size = 32

ulaz_trening = image_dataset_from_directory(main_path,  subset='training', validation_split=0.2,
                                            seed=52, batch_size=batch_size,
                                            image_size=img_size)

ulaz_test = image_dataset_from_directory(main_path, subset='validation', validation_split=0.2,
                                         seed=52, batch_size=batch_size,
                                         image_size=img_size)

classes = ulaz_trening.class_names

## Trazenje optimalne dropout verovatnoce
# tuner = kt.RandomSearch(make_model_for_search, objective='val_accuracy', overwrite=True, max_trials=5)
#
# tuner.search(ulaz_trening, epochs=10, validation_data=ulaz_test)
#
# best_hyperparam = tuner.get_best_hyperparameters(num_trials=1)[0]
#
# print("Optimalana verovatnoca odbacivanja neurona u dropout sloju je:", best_hyperparam['dropout_prob'] / 100)
#
# exit(0)


model = make_model(0.34)

model.summary()


history = model.fit(ulaz_trening, epochs=20, validation_data=ulaz_test)  # ne mora batch_size jer je definisan vec gore

model.save_weights("model.chpt")

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

# model.load_weights("model.chpt")

pred = np.array([])
labels = np.array([])

for img, lab in ulaz_test:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

## kreiranje primera klasifikacije
# map_class = {}
# cnt = 0
# for i in classes:
#     map_class[i] = cnt
#     cnt += 1
#
# plt.figure()
# cnt = 0
# q = -1
# nn = 5
# mm = 6
# for img, lab in ulaz_test:
#     if cnt == 10:
#         break
#     for j in range(32):
#         q = q + 1
#         if cnt < nn * mm:
#             if random.randrange(0, 5) == 0:
#                 plt.subplot(nn, mm, cnt + 1)
#                 plt.imshow(img[j].numpy().astype('uint8'))
#                 tmp = np.argmax(model.predict(img, verbose=0), axis=1)[j]
#                 # tmp = int(pred[q] + 0.5)
#                 t = classes[tmp]
#                 if tmp != lab[j]:
#                     t += "(ac " + classes[lab[j]] + ")"
#                 plt.title(t)
#                 plt.axis("off")
#                 cnt += 1
#         else:
#             break
#
# plt.show()
#
# exit(0)

cm_test = confusion_matrix(labels, pred)
acc = accuracy_score(labels, pred)
print(cm_test)
print(acc)

cm_display_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=classes)
cm_display_test.plot()
plt.show()

pred_train = np.array([])
labels_train = np.array([])
for img, lab in ulaz_trening:
    labels_train = np.append(labels_train, lab)
    pred_train = np.append(pred_train, np.argmax(model.predict(img, verbose=0), axis=1))

cm_train = confusion_matrix(labels_train, pred_train)
cm_display_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=classes)
cm_display_train.plot()
plt.show()



plt.show()
