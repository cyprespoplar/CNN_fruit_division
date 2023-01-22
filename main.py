import os

train_dir = './训练集'
validation_dir = './验证集'

train_apples_dir = './训练集/苹果训练集'
train_bananas_dir = './训练集/香蕉训练集'

validation_apples_dir = './验证集/苹果验证集'
validation_bananas_dir = './验证集/香蕉验证集'

test_apples_dir = './测试集/苹果测试集'
test_bananas_dir = './测试集/香蕉测试集'

# 分组情况
print('total training apple images:', len(os.listdir(train_apples_dir)))
print('total training banana images:', len(os.listdir(train_bananas_dir)))
print('total validation apple images:', len(os.listdir(validation_apples_dir)))
print('total validation banana images:', len(os.listdir(validation_bananas_dir)))
print('total testing apple images:', len(os.listdir(test_apples_dir)))
print('total testing banana images:', len(os.listdir(test_bananas_dir)))

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()


from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
# 数据处理
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 使用flow_from_directory()方法可以实例化一个针对图像batch的生成器
train_generator = train_datagen.flow_from_directory(

    train_dir,  # 目标目录

    target_size=(150, 150),  # 将所有图像大小调整为150*150
    batch_size=20,

    class_mode='binary')  # 因为使用了binary_crossentropy损失，所以需要使用二进制标签

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')


for data_batch, labels_batch in train_generator:
	    print('data batch shape:', data_batch.shape)
	    print('labels batch shape:', labels_batch.shape)
     break


model.save('apples_and_bananas_1.h5')
history = model.fit_generator(
	      train_generator,
	      steps_per_epoch=64,
	      epochs=10,
      validation_data=validation_generator,
	      validation_steps=28)

import matplotlib.pyplot as plt

val_acc = history.history['val_acc']
acc = history.history['acc']
val_loss = history.history['val_loss']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image

# 加载训练好的模型
model = tf.keras.models.load_model('apples_and_bananas_1.h5')

apple_test_dir = './测试集/苹果测试集'
banana_test_dir = './测试集/香蕉测试集'


def predict_func(data_dir, lable):
    # lable==1：香蕉；lable==0：苹果
    t = 0
    for filename in os.listdir(data_dir):
        image_path = os.path.join(data_dir, filename)
        img = image.load_img(image_path, target_size=(150, 150))

        # 图像预处理
        x = image.img_to_array(img)


x = np.expand_dims(x, axis=0)

# 对图像进行分类
preds = model.predict(x)
if preds == lable:
    t = t + 1
else:
    continue
return t / len(os.listdir(data_dir))

predict_result_apple = predict_func(apple_test_dir, 0)
predict_result_banana = predict_func(banana_test_dir, 1)

# 输出模型预测准确率
print('Apple predicting result:', predict_result_apple)
print('Banana predicting result:', predict_result_banana)