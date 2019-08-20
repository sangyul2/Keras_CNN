
# coding: utf-8

# In[ ]:

# # When Jupyter notebook Kernel Restart code

# from IPython.display import display_html
# def restartkernel() :
#     display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
# restartkernel()


# In[1]:

import os
from glob import glob
import shutil
import itertools
from datetime import datetime
from tqdm import tqdm_notebook
from datetime import datetime
import pandas as pd
import random

import seaborn as sns
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from keras.models import Model, load_model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from keras.regularizers import l1, l2


# from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D,
# from keras.layers import MaxPooling2D, ZeroPadding2D, AveragePooling2D

get_ipython().magic('matplotlib inline')


# ## 경로 설정

# In[2]:

seed_value= 777

random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)


# In[53]:

test_dir_FR02


# In[ ]:




# In[28]:

data_dir = 'Dataset/'

fold_dir_FR02 = '42_HDI_dim3_RGB_FR02_Gvis_Class_POR/'
data_dir_FR02 = data_dir+fold_dir_FR02
train_dir_FR02 = data_dir_FR02 + 'Train/'
test_dir_FR02 = data_dir_FR02 + 'Test/'

fold_dir_FR08 = '48_HDI_dim3_RGB_FR08_Gvis_Class_POR/'
data_dir_FR08 = data_dir+fold_dir_FR08
train_dir_FR08 = data_dir_FR08 + 'Train/'
test_dir_FR08 = data_dir_FR08 + 'Test/'

fold_dir_FR09 = '49_HDI_dim3_RGB_FR09_Gvis_Class_POR/'
data_dir_FR09 = data_dir+fold_dir_FR09
train_dir_FR09 = data_dir_FR09 + 'Train/'
test_dir_FR09 = data_dir_FR09 + 'Test/'

batch_size = 32
num_classes = len(os.listdir(train_dir_FR08))
# num_epochs = 20
data_augmentation = True

learning_rate = 0.001


# In[29]:

train_path_FR02 = glob(os.path.join(train_dir_FR02, '*/*.jpg'))
test_path_FR02 = glob(os.path.join(test_dir_FR02, '*/*.jpg'))

train_path_FR08 = glob(os.path.join(train_dir_FR08, '*/*.jpg'))
test_path_FR08 = glob(os.path.join(test_dir_FR08, '*/*.jpg'))

train_path_FR09 = glob(os.path.join(train_dir_FR09, '*/*.jpg'))
test_path_FR09 = glob(os.path.join(test_dir_FR09, '*/*.jpg'))




# In[30]:

num_dataset = len(train_path_FR02)
num_testset = len(test_path_FR02)

steps_per_epoch = num_dataset // batch_size
steps_per_validation = num_testset // batch_size

print('steps_per_epoch : ',steps_per_epoch)
print('steps_per_validation : ',steps_per_validation)
path_1 = train_path_FR02[0]
path_2 = train_path_FR08[0]
path_3 = train_path_FR09[0]
print('path_1 : ',path_1)
print('path_2 : ',path_2)
print('path_3 : ',path_3)


# In[9]:

for i in range(5):
    random.shuffle(train_path_1)
    random.shuffle(test_paths_1)
    random.shuffle(train_path_2)
    random.shuffle(test_paths_2)
    i+=1


# In[31]:

# 3 channel image (color)

image = np.array(Image.open(path_1).convert('RGB'))  # color : RGB, gray : L, YCbCr, LAB, HSV, 1
#image = np.expand_dims(image, -1)  # add dimension if gray

print('num_classes  : ',num_classes)
print('Train_dataset: ',num_dataset)
print('Test_dataset : ',num_testset)
print('image.shape  : ',image.shape)

image.shape
input_shape = (100, 100, 3)
h, w, c = input_shape


# In[131]:

# plt.figure(figsize = (20, 20))
# plt.subplot(241)
# plt.imshow(image[:,:,0])

# plt.subplot(242)
# plt.imshow(image[:,:,0], cmap='RdBu')

# plt.subplot(243)
# plt.imshow(image[:,:,0], cmap='jet')

# plt.subplot(244)
# plt.imshow(image[:,:,0]+image[:,:,0])

# plt.subplot(245)
# plt.imshow(image[:,:,0]+image[:,:,0])

# plt.subplot(246)
# plt.imshow(image[:,:,0]+image[:,:,0], cmap='RdBu')

# plt.subplot(247)
# plt.imshow(image[:,:,0]+image[:,:,0], cmap='jet')

# plt.subplot(248)
# plt.imshow(image[:,:,0]+image[:,:,0]+image[:,:,0])

# plt.show()


# In[ ]:

# # 1 channel image (gray)

# image = np.array(Image.open(path).convert("L"))
# image = np.expand_dims(image, -1)
# print('num_classes  : ',num_classes)
# print('Train_dataset: ',num_dataset)
# print('Test_dataset : ',num_testset)
# print('image.shape  : ',image.shape)

# input_shape = image.shape
# h, w, c = input_shape


# In[ ]:

# # VGG16

# inputs = layers.Input(input_shape)
# net = layers.Conv2D(64, (3, 3), padding='same')(inputs)
# net = layers.Conv2D(64, (3, 3), padding='same')(net)
# net = layers.Conv2D(64, (3, 3), padding='same')(net)
# net = layers.BatchNormalization()(net)
# net = layers.Activation('relu')(net)
# net = layers.MaxPooling2D(pool_size=(2, 2))(net)

# net = layers.Conv2D(128, (3, 3), padding='same')(net)
# net = layers.Conv2D(128, (3, 3), padding='same')(net)
# net = layers.Conv2D(128, (3, 3), padding='same')(net)
# net = layers.BatchNormalization()(net)
# net = layers.Activation('relu')(net)
# net = layers.MaxPooling2D(pool_size=(2, 2))(net)
# net = layers.Dropout(0.25)(net)

# net = layers.Conv2D(256, (3, 3), padding='same')(net)
# net = layers.Conv2D(256, (3, 3), padding='same')(net)
# net = layers.Conv2D(256, (3, 3), padding='same')(net)
# net = layers.BatchNormalization()(net)
# net = layers.Activation('relu')(net)
# net = layers.MaxPooling2D(pool_size=(2, 2))(net)
# net = layers.Dropout(0.25)(net)

# net = layers.Conv2D(512, (3, 3), padding='same')(net)
# net = layers.Conv2D(512, (3, 3), padding='same')(net)
# net = layers.Conv2D(512, (3, 3), padding='same')(net)
# net = layers.BatchNormalization()(net)
# net = layers.Activation('relu')(net)
# net = layers.MaxPooling2D(pool_size=(2, 2))(net)
# net = layers.Dropout(0.25)(net)

# net = layers.Conv2D(512, (3, 3), padding='same')(net)
# net = layers.Conv2D(512, (3, 3), padding='same')(net)
# net = layers.Conv2D(512, (3, 3), padding='same')(net)
# net = layers.BatchNormalization()(net)
# net = layers.Activation('relu')(net)
# net = layers.MaxPooling2D(pool_size=(2, 2))(net)
# net = layers.Dropout(0.25)(net)

# net = layers.GlobalAveragePooling2D()(net)
# net = layers.Flatten()(net)
# net = layers.Dense(512)(net)
# net = layers.Activation('relu')(net)
# net = layers.Dropout(0.5)(net)
# net = layers.Dense(num_classes)(net)
# net = layers.Activation('softmax')(net)

# model = tf.keras.Model(inputs=inputs, outputs=net)


# In[7]:

# ResNet 50 + regulize (Total params: 26,215,702)

# # net = layers.MaxoutDense(1024, nb_feature=4, kernel_regularizer=l2(0.001))(net)
# # net = layers.MaxoutDense(512,  nb_feature=4, kernel_regularizer=l2(0.001))(net)
# # net = layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(net)
# # net = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(net)

# base_model_1 = tf.keras.applications.resnet50.ResNet50(weights=None, input_shape=input_shape,
#                                                      include_top=False, classes=num_classes)

# net = base_model_1.output
# net = layers.GlobalAveragePooling2D()(net)
# net = layers.Flatten()(net)
# net = layers.Dense(1024, activation='relu')(net)
# net = layers.Dense(512, activation='relu')(net)
# net = layers.Dropout(0.5)(net)
# net = layers.Dense(num_classes, activation='softmax', name='fc1000')(net)

# model_1 = tf.keras.Model(inputs=base_model_1.input,outputs=net)


# In[32]:

# Xception + regulize (Total params: 23,495,742)

base_model_xc = tf.keras.applications.xception.Xception(weights=None, input_shape=input_shape,
                                                     include_top=False, classes=num_classes)

net = base_model_xc.output
net = layers.GlobalAveragePooling2D()(net)
net = layers.Flatten()(net)
net = layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(net)
net = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(num_classes, activation='softmax', name='fc1000')(net)

model_FR02 = tf.keras.Model(inputs=base_model_xc.input, outputs=net)
model_FR08 = tf.keras.Model(inputs=base_model_xc.input, outputs=net)
model_FR09 = tf.keras.Model(inputs=base_model_xc.input, outputs=net)


# In[ ]:

# # DenseNet 121 + regulize

# base_model = tf.keras.applications.densenet.DenseNet121(weights=None, input_shape=input_shape,include_top=False)

# net = base_model.output
# net = layers.GlobalAveragePooling2D()(net)
# net = layers.Flatten()(net)
# net = layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(net)
# net = layers.Dense(512, activation='relu',  kernel_regularizer=l2(0.001))(net)
# net = layers.Dropout(0.5)(net)
# net = layers.Dense(num_classes, activation='softmax', name='fc1000')(net)

# model = tf.keras.Model(inputs=base_model.input,outputs=net)


# In[8]:

# # NasNet 121 + regulize (Total params: 5,877,334)

# base_model_3 = tf.keras.applications.nasnet.NASNetMobile(weights=None, input_shape=input_shape,
#                                                        include_top=False, classes=num_classes)

# net = base_model_3.output
# net = layers.GlobalAveragePooling2D()(net)
# net = layers.Flatten()(net)
# net = layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(net)
# net = layers.Dense(512, activation='relu',  kernel_regularizer=l2(0.001))(net)
# net = layers.Dropout(0.5)(net)
# net = layers.Dense(num_classes, activation='softmax', name='fc1000')(net)

# model_3 = tf.keras.Model(inputs=base_model_3.input,outputs=net)


# In[ ]:

model_3.summary()


# In[138]:

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        width_shift_range = 0.05,
        height_shift_range = 0.05,
        zoom_range = 0.03,
        brightness_range = [0.7,1.0], # 밝기[min,max]
        fill_mode='nearest',
#        rotation_range = 90, # 돌리기 0 ~ 90
#        shear_range = 0.2,   # 비틀기
        vertical_flip = True,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(h, w),
    color_mode='rgb', # One of : 'rgb', 'grayscale'
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
#     save_to_dir='out_images'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(h, w),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical')


# In[73]:

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator_FR02 = test_datagen.flow_from_directory(
    test_dir_FR02,
    target_size=(h, w),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical')
validation_generator_FR08 = test_datagen.flow_from_directory(
    test_dir_FR08,
    target_size=(h, w),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical')
validation_generator_FR09 = test_datagen.flow_from_directory(
    test_dir_FR09,
    target_size=(h, w),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical')


# In[140]:

MODEL_PATH = 'Model'
FOLD_PATH = fold_dir
MODEL_SAVE_FOLDER_PATH = os.path.join(MODEL_PATH,FOLD_PATH)
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + '{val_loss:.4f}.hdf5'

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
#        monitor='val_acc',
        monitor='val_loss',
        factor=0.05, # 콜백이 호출되면 학습률을 x배 줄임
        patience=10  # 몇 번 확인 후 안 오르면 학습률을 줄임
    ),
    # `val_loss`가 n번의 에포크에 걸쳐 향상되지 않으면 훈련을 멈춥니다.
    tf.keras.callbacks.EarlyStopping(patience=31, monitor='val_loss'),
    # `./logs` 디렉토리에 텐서보드 로그를 기록합니다.
    tf.keras.callbacks.TensorBoard(log_dir='./Logs/'+FOLD_PATH)
]


# In[ ]:

# model.load_weights(loading_path)


# In[142]:

model_list = os.listdir(MODEL_SAVE_FOLDER_PATH)
model_count=len(model_list)

if model_count == 0:
    loading_path=[]
    pass
else:
    model_list.sort()
    loading_path = MODEL_SAVE_FOLDER_PATH + model_list[0]
#    model = tf.keras.models.load_model(loading_path)
    model.load_weights(loading_path)
loading_path


# In[ ]:

if not len(model_list) <= 5:
    n = 1
    for i in model_list:
        if not n <= 5:
            del_model = MODEL_SAVE_FOLDER_PATH + i
            os.remove(del_model)
            print('Delete : ' + MODEL_SAVE_FOLDER_PATH + i)
        n+=1


# In[43]:

MODEL_SAVE_FOLDER_PATH = 'Model\\Ensenble2/'
os.listdir(MODEL_SAVE_FOLDER_PATH)


# In[44]:

model_FR02_path = 'Model/Ensenble2/'+os.listdir('Model/Ensenble2')[0]
model_FR08_path = 'Model/Ensenble2/'+os.listdir('Model/Ensenble2')[2]
model_FR09_path = 'Model/Ensenble2/'+os.listdir('Model/Ensenble2')[3]


# In[45]:

model_FR02.load_weights(model_FR02_path)
model_FR08.load_weights(model_FR08_path)
model_FR09.load_weights(model_FR09_path)


# In[46]:

# learning_rate = 0.01
# learning_rate = 0.001
# learning_rate = 0.0001

model_FR02.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate),
              #optimizer='rmsprop',
              metrics=['accuracy'])

model_FR08.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate),
              #optimizer='rmsprop',
              metrics=['accuracy'])

model_FR09.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate),
              #optimizer='rmsprop',
              metrics=['accuracy'])


# In[ ]:

# model.compile(loss='categorical_crossentropy',
#               optimizer=tf.keras.optimizers.Adam(learning_rate),
#               metrics=['accuracy'])

# # model evaluation
# score = model.evaluate(validation_generator, 
#                          steps=10,
#                          verbose=1)
# # print("%s : %.2f%%" % (score[1], score[0]*100))
# score


# In[47]:

steps_per_epoch, steps_per_validation


# In[ ]:

history = model.fit_generator(
                        train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=200,
                        validation_data=validation_generator,
                        validation_steps=steps_per_validation,
                        callbacks=callbacks,
                        verbose=2)


# In[ ]:

# for epoch in range(num_epochs):

#     history = model.fit_generator(
#                         train_generator,
#                         steps_per_epoch=steps_per_epoch,
#                         epochs=1,
#                         validation_data=validation_generator,
#                         validation_steps=steps_per_validation,
#                         callbacks=callbacks,
#                         verbose=2)
    
#     save_name = '%.4f' % history.history['val_acc'][-1]
#     model.save(MODEL_SAVE_FOLDER_PATH+"Model_%s.h5" % save_name)


# In[ ]:

# from keras.utils import plot_model
# # // 케라스 모델인 model을 보기 쉽게 model.png 파일에 이미지로 떨어뜨려 준다.  
# # // show_shapes = True, layer의 input과 output의 shape을 같이 보여준다.  
# # // show_layers_name = True, layer 이름을 같이 보여 준다.  
# # // rankdir = TB | LR, TB면 수직으로, LR 수평으로 그려준다.  
# plot_model ( model, to_file='model.png', show_shapes=True )


# In[ ]:

# save_name = '%.4f' % history.history['val_loss'][-1]
model.save(MODEL_SAVE_FOLDER_PATH+"Model.h5")


# In[74]:

model_FR02.evaluate_generator(validation_generator_FR02, steps=10, verbose=1)


# In[75]:

model_FR08.evaluate_generator(validation_generator_FR08, steps=10, verbose=1)


# In[76]:

model_FR09.evaluate_generator(validation_generator_FR09, steps=10, verbose=1)


# In[57]:

# Test set check

test_set_FR02 = test_datagen.flow_from_directory(
    test_dir_FR02,
    target_size=(h, w),
    batch_size=2130,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True)

test_set_FR08 = test_datagen.flow_from_directory(
    test_dir_FR08,
    target_size=(h, w),
    batch_size=2130,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True)

test_set_FR09 = test_datagen.flow_from_directory(
    test_dir_FR09,
    target_size=(h, w),
    batch_size=2130,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True)


# In[59]:

from sklearn.metrics import confusion_matrix
image_FR02, label_FR02 = next(test_set_FR02)
image_FR08, label_FR08 = next(test_set_FR08)
image_FR09, label_FR09 = next(test_set_FR09)


# In[61]:

print('image_FR02.shape : ', image_FR02.shape)
print('image_FR08.shape : ', image_FR08.shape)
print('image_FR09.shape : ', image_FR09.shape)
print('label_FR02.shape : ', label_FR02.shape)


# In[62]:

logits_FR02 = model_FR02.predict(image_FR02)
logits_FR08 = model_FR08.predict(image_FR08)
logits_FR09 = model_FR09.predict(image_FR09)


# In[69]:

ensenble = 0.5 * (logits_1 + logits_2)


# In[88]:

logits_1[0]


# In[79]:

logits_1.astype(np.int)


# In[87]:

logits_1[0:20].astype(np.int)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

num_image = image.shape[0]
num_col = 8
a = num_image // num_col
plt.figure(figsize = (20, 8))

for i in range (num_image):
    #print(a, num_col, b+1)
    
    plt.subplot(a, num_col,i+1)
    plt.title('No.'+str(i+1)+'  label : '+ str(np.argmax(label[i],-1))+',  pred :' + str(np.argmax(logits[i],-1)))
    plt.imshow(image[i, :, :,0], cmap='gray')
    #plt.imshow(image[i, :, :,0], cmap='RdBu')
    #plt.imshow(image[i, :, :,0], cmap='jet')
plt.show()


# In[ ]:

num_image = image.shape[0]
num_col = 8
a = num_image // num_col
# plt.figure(figsize = (20, 8))

print('OK')
plt.figure(figsize = (20, 8))
for i in range(num_image):
    if np.argmax(label[i],-1) == np.argmax(logits[i],-1):
        plt.subplot(a, num_col,i+1)
        plt.title('No.'+str(i+1)+'  label : '+ str(np.argmax(label[i],-1))+',  pred :' + str(np.argmax(logits[i],-1)))
        plt.imshow(image[i, :, :,0], cmap='gray')
plt.show()

print('Error 2')
plt.figure(figsize = (20, 8))
for i in range(num_image):
    if np.argmax(label[i],-1) > np.argmax(logits[i],-1):
        plt.subplot(a, num_col,i+1)
        plt.title('No.'+str(i+1)+'  label : '+ str(np.argmax(label[i],-1))+',  pred :' + str(np.argmax(logits[i],-1)))
        plt.imshow(image[i, :, :,0], cmap='gray')
plt.show()

print('Error 1')
plt.figure(figsize = (20, 8))
for i in range(num_image):
    if np.argmax(label[i],-1) < np.argmax(logits[i],-1):
        plt.subplot(a, num_col,i+1)
        plt.title('No.'+str(i+1)+'  label : '+ str(np.argmax(label[i],-1))+',  pred :' + str(np.argmax(logits[i],-1)))
        plt.imshow(image[i, :, :,0], cmap='gray')
plt.show()


# In[63]:

arr_FR02 = confusion_matrix(np.argmax(logits_FR02, -1), np.argmax(label_FR02, -1))  # (tn, fp), (fn, tp)
arr_FR08 = confusion_matrix(np.argmax(logits_FR08, -1), np.argmax(label_FR08, -1))  # (tn, fp), (fn, tp)
arr_FR09 = confusion_matrix(np.argmax(logits_FR09, -1), np.argmax(label_FR09, -1))  # (tn, fp), (fn, tp)


# In[64]:

MODEL_SAVE_FOLDER_PATH


# In[66]:

pd_arr_FR02 = pd.DataFrame(arr_FR02)
pd_arr_FR08 = pd.DataFrame(arr_FR08)
pd_arr_FR09 = pd.DataFrame(arr_FR09)

save_path_FR02 = MODEL_SAVE_FOLDER_PATH+'confusion_'+'FR02'+'.csv'
pd_arr_FR02.to_csv(save_path_FR02, index = True)

save_path_FR08 = MODEL_SAVE_FOLDER_PATH+'confusion_'+'FR08'+'.csv'
pd_arr_FR08.to_csv(save_path_FR08, index = True)

save_path_FR09 = MODEL_SAVE_FOLDER_PATH+'confusion_'+'FR09'+'.csv'
pd_arr_FR09.to_csv(save_path_FR09, index = True)


# In[72]:

df_array1 = pd.DataFrame(arr_FR02)
plt.figure(figsize = (14,10))
sns.heatmap(df_array1, annot=True)
plt.show()


# In[69]:

df_array2 = pd.DataFrame(arr_FR08)
plt.figure(figsize = (14,10))
sns.heatmap(df_array2, annot=True)
plt.show()


# In[68]:

df_array3 = pd.DataFrame(arr_FR09)
plt.figure(figsize = (14,10))
sns.heatmap(df_array3, annot=True)
plt.show()


# In[ ]:



