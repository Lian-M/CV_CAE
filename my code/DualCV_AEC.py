import keras
import os
import h5py
import scipy.io
import math
from keras.utils.generic_utils import get_custom_objects
from keras import layers
from keras.layers import Input, Concatenate, Add, Lambda, Activation
import numpy as np
from tensorflow.keras.optimizers import Adam
from keras.models import Model
import keras.backend as KB
import matplotlib.pyplot as plt
from complexnn import ComplexConv2D

data_dir = 'data/Dual256/nor/'
input_dict1 = h5py.File(os.path.join(data_dir, 'data1_cvnor.mat'))
echo1 = input_dict1['data1_cvnor'][:, :, :, :]
echo1 = echo1.transpose(0, 3, 1, 2)

input_dict2 = h5py.File(os.path.join(data_dir, 'data2_cvnor.mat'))
echo2 = input_dict2['data2_cvnor'][:, :, :, :]
echo2 = echo2.transpose(0, 3, 1, 2)

data_dir1 = 'data/Dual256/test/'
input_dict3 = h5py.File(os.path.join(data_dir1, 'data1_cvnor.mat'))
echo1_test = input_dict3['data1_cvnor'][:, :, :]
echo1_test = echo1_test[np.newaxis, :, :, :]
echo1_test = echo1_test.transpose(0, 3, 1, 2)

input_dict4 = h5py.File(os.path.join(data_dir1, 'data2_cvnor.mat'))
echo2_test = input_dict4['data2_cvnor'][:, :, :]
echo2_test = echo2_test[np.newaxis, :, :, :]
echo2_test = echo2_test.transpose(0, 3, 1, 2)

out_dict = h5py.File(os.path.join(data_dir, 'label_nor.mat'))
imaging = out_dict['label_nor'][:, :, :]
imaging = np.expand_dims(imaging, axis=0)
imaging = imaging.transpose(1, 0, 2, 3)

#print(np.shape(echo))

# 编码器
input_img1 = Input(shape=(2, 256, 256), name='input1')
input_img2 = Input(shape=(2, 256, 256), name='input2')

x1 = ComplexConv2D(16, kernel_size=5, strides=(1,1),
                  activation='relu', padding='same', data_format='channels_first')(input_img1)
x2 = ComplexConv2D(16, kernel_size=5, strides=(1,1),
                  activation='relu', padding='same', data_format='channels_first')(input_img2)

x1 = layers.MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_first')(x1)
x2 = layers.MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_first')(x2)

x1 = ComplexConv2D(8, kernel_size=3, strides=(1,1),
                  activation='relu', padding='same', data_format='channels_first')(x1)
x2 = ComplexConv2D(8, kernel_size=3, strides=(1,1),
                  activation='relu', padding='same', data_format='channels_first')(x2)

x1 = layers.MaxPooling2D((2, 2), padding='same', data_format='channels_first')(x1)
x2 = layers.MaxPooling2D((2, 2), padding='same', data_format='channels_first')(x2)

x1 = ComplexConv2D(8, kernel_size=3, strides=(1,1),
                  activation='relu', padding='same', data_format='channels_first')(x1)
x2 = ComplexConv2D(8, kernel_size=3, strides=(1,1),
                  activation='relu', padding='same', data_format='channels_first')(x2)

encoded1 = layers.MaxPooling2D((2, 2), padding='same', data_format='channels_first')(x1)
encoded2 = layers.MaxPooling2D((2, 2), padding='same', data_format='channels_first')(x2)

# 拼接，特征融合
encoded1_real = [Lambda(lambda x: x[:, :KB.int_shape(x)[1] // 2, :, :], output_shape=(KB.int_shape(encoded1)[1]//2, KB.int_shape(encoded1)[2], KB.int_shape(encoded1)[3]))(encoded1)]
encoded1_imag = [Lambda(lambda x: x[:, KB.int_shape(x)[1] // 2:, :, :], output_shape=(KB.int_shape(encoded1)[1]//2, KB.int_shape(encoded1)[2], KB.int_shape(encoded1)[3]))(encoded1)]
encoded2_real = Lambda(lambda x: x[:, :KB.int_shape(x)[1] // 2, :, :], output_shape=(KB.int_shape(encoded2)[1]//2, KB.int_shape(encoded2)[2], KB.int_shape(encoded2)[3]))(encoded2)
encoded2_imag = Lambda(lambda x: x[:, KB.int_shape(x)[1] // 2:, :, :], output_shape=(KB.int_shape(encoded2)[1]//2, KB.int_shape(encoded2)[2], KB.int_shape(encoded2)[3]))(encoded2)
encoded1_real.append(encoded2_real)
encoded1_imag.append(encoded2_imag)
out = encoded1_real + encoded1_imag
out = Concatenate(axis=1)(out)
fusion = ComplexConv2D(8, kernel_size=1, strides=(1,1),
                  activation='relu', padding='same', data_format='channels_first')(out)

# 解码器

x = ComplexConv2D(8, kernel_size=3, strides=(1,1),
                  activation='relu', padding='same', data_format='channels_first')(fusion)

x = layers.UpSampling2D((2, 2), data_format='channels_first')(x)

x = ComplexConv2D(8, kernel_size=3, strides=(1,1),
                  activation='relu', padding='same', data_format='channels_first')(x)

x = layers.UpSampling2D((2, 2), data_format='channels_first')(x)

x = ComplexConv2D(16, kernel_size=3, strides=(1,1),
                  activation='relu', padding='same', data_format='channels_first')(x)

x = layers.UpSampling2D((2, 2), data_format='channels_first')(x)

decoded = layers.Conv2D(1, kernel_size=3, strides=(1,1), padding='same', name='imaging', data_format='channels_first')(x)

batch_size = 32
print('Training ------------')

model = Model(inputs=[input_img1, input_img2], outputs=decoded)

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

hist = model.fit({'input1': echo1, 'input2': echo2}, {'imaging': imaging}, shuffle=True, epochs=10, batch_size=batch_size, validation_split=0.1, validation_data=None)
#保存模型结构
model_json = model.to_json()
with open('DualCVAEC_architecture.json', 'w') as json_file: json_file.write(model_json)
#保存训练历史损失函数值
with open('DualCVAECloss.txt', 'w') as f: f.write(str(hist.history))
#保存模型权重
model.save_weights('DualCVAEC_weight.h5')
print("Save model to disk")


#score = model.evaluate(echo_test, imaging_test, verbose=1)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
print(hist.history.keys())
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predictions = model.predict({'input1': echo1_test, 'input2': echo2_test})
#print(np.shape(predictions))
#将预测得到的图像保存为.mat格式文件（.mat文件中refocus是给变量起的名称，predictiosn是变量的值）
scipy.io.savemat('CVpredict_image.mat', {'refocus': predictions})



