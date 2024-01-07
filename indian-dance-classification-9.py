# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ![dance.jpg](attachment:dance.jpg)

# %% [markdown]
# # Indian Dance Form Classification.
#
# - <https://www.kaggle.com/datasets/aditya48/indian-dance-form-classification>
#
# The dataset consists of 650 images belonging to 9 categories, namely **manipuri, bharatanatyam, odissi, kathakali, kathak, sattriya, kuchipudi, mohiniyattam and purulia chhau**

# %% _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Activation
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D, Dropout,Activation
from sklearn import metrics
from sklearn.model_selection import train_test_split

# For interupt the training when val loss is stagnant
from tensorflow.keras.callbacks import EarlyStopping

from keras.utils import to_categorical

import itertools
import os
import cv2
import matplotlib.pyplot as plt

# %%
train_path = "./archive/train/"
test_path = "./archive/test/"

kathak = train_path + "kathak/"
odissi = train_path + "odissi/"
sattriya = train_path + "sattriya/"
purulia_chhau = train_path + "purulia_chhau/"

kathak_path = os.listdir(kathak)
sattriya_path = os.listdir(sattriya)
odissi_path = os.listdir(odissi)
purulia_chhau_path = os.listdir(purulia_chhau)


# %% [markdown]
# ## Visualizing the Data.

# %%
def show_img(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

show_img(kathak + kathak_path[2])

# %% [markdown]
# ### Odissi

# %%
show_img(odissi + odissi_path[2])

# %% [markdown]
# ### Sattriya

# %%
show_img(sattriya + sattriya_path[2])

# %% [markdown]
# ### Purulia Chhau

# %%
show_img(purulia_chhau + purulia_chhau_path[2])

# %% [markdown]
# ## Preparing Training Data

# %%
training_data = []
IMG_SIZE = 224
datadir = "./archive/train"
categories = ['bharatanatyam', 'kathak', 'kathakali', 'kuchipudi',  'manipuri', 'mohiniyattam', 'odissi', 'sattriya', 'purulia_chhau']


def create_training_data():
    for (class_num, category) in enumerate(categories):
        path = os.path.join(datadir, category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
                #yield (new_array, class_num)
            except:
                print(img)
                pass


testing_data_with_name = []
def create_testing_data():
    path = os.path.join('./archive/test')
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            testing_data_with_name.append((img, new_array))
        except:
            print(img)
            pass


create_training_data()
create_testing_data()
testing_ids, testing_data = list(zip(*testing_data_with_name))

# %%
#training_data = np.array(training_data)
#print(training_data.shape)
testing_data = np.array(testing_data)
testing_data.shape

# %%
np.random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

# %%
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)

# %%
print(X.shape)
print(y.shape)

# %%
a, b = np.unique(y, return_counts = True)
pd.DataFrame((categories, a, b), index=['class name', 'class label', 'no. of samples'])

# %% [markdown]
# ## Analysing the Training Data:

# %%
plt.figure(figsize=(12, 4))
plt.bar(categories, b)
plt.show()

# %%
# normalizing values to [0, 1] in X
#X = X / 255.0

# %% [markdown]
# ## Train Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Shape of test_x: ",X_train.shape)
print("Shape of train_y: ",y_train.shape)
print("Shape of test_x: ",X_test.shape)
print("Shape of test_y: ",y_test.shape)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.08, random_state=42)

# %%
# one hot encoding class labels
y_train = to_categorical(y_train, num_classes = 9)
y_test = to_categorical(y_test, num_classes = 9)
y_val = to_categorical(y_val, num_classes = 9)

print("Shape of test_x: ",X_train.shape)
print("Shape of train_y: ",y_train.shape)
print("Shape of test_x: ",X_test.shape)
print("Shape of test_y: ",y_test.shape)

# %%
train_x = keras.utils.normalize(X_train, axis=1)
test_x = keras.utils.normalize(X_test, axis=1)

# %% [markdown]
#
# ## VGG-16
#

# %%
model = keras.applications.VGG16(input_shape = (224,224,3), weights = 'imagenet',include_top=False)

for layer in model.layers:
    layer.trainable = False

last_layer = model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)

# add fully-connected & dropout layers
x = Dense(4096, activation='relu',name='fc-1')(x)
x = Dropout(0.2)(x)
x = Dense(4096, activation='relu',name='fc-2')(x)
x = Dropout(0.2)(x)

# x = Dense(4096, activation='relu',name='fc-3')(x)
# x = Dropout(0.2)(x)

# a softmax layer for 8 classes
num_classes = 9
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
model2 = Model(inputs=model.input, outputs=out)

model2.summary()

# %% [markdown]
# ### Training the model

# %% _kg_hide-output=true
model2.compile(optimizer='adam',
              loss ='categorical_crossentropy',
              metrics=['accuracy'])

hist = model2.fit(X_train, y_train, batch_size=30, epochs = 30, validation_data = (X_val, y_val))
hist = hist.history

# %%
# model2.save('dance_vgg16.9.2.keras')
# np.save('dance_vgg16.9.2_history.npy', hist)

# %%
model2 = keras.models.load_model('dance_vgg16.9.2.keras')
hist = np.load('dance_vgg16.9.2_history.npy', allow_pickle='TRUE').item()

# %%
p_train = model2.predict(X_train)


# %%
def performance_metrics(y_true, y_pred, labels):
    print(metrics.classification_report(y_true, y_pred))

    return pd.DataFrame(metrics.confusion_matrix(y_true, y_pred), index=categories, columns=categories)


# %%
print('training')
performance_metrics(y_train.argmax(axis=1), p_train.argmax(axis=1), categories)

# %% [markdown]
# ### Testing

# %%
p_test = model2.predict(X_test)

# %%
print('testing:')
performance_metrics(y_test.argmax(axis=1), p_test.argmax(axis=1), categories)

# %%
predicted = model2.predict(testing_data)
predicted = predicted.argmax(axis=1)
predicted_categories = [categories[p] for p in predicted]

# %%
start = 0
for i in range(start + 1, start + 10):
    plt.subplot(3, 3, i - start)
    plt.imshow(X_test[i][..., ::-1])
    plt.title(predicted_categories[i])
    plt.axis('off')

# %%
pd.DataFrame(zip(testing_ids, predicted_categories), columns=['Image', 'Class Name'])

# %%
epochs = 30

train_loss = hist['loss']
val_loss = hist['val_loss']
train_acc = hist['accuracy']
val_acc = hist['val_accuracy']
xc = range(epochs)

plt.plot(xc,train_loss, label='train')
plt.plot(xc,val_loss, label='val')
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend()
#plt.savefig("train_loss.png")
plt.show()


# %%
plt.plot(xc,train_acc, label='train')
plt.plot(xc,val_acc, label='val')
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend()
#plt.savefig("train_acc.png")
plt.show()
