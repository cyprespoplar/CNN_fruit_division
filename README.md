# CNN_fruit_division
基于CNN的水果分类
水果分类任务，本实验使用了最基本的图像二分类技术。
对于这样一个图像分类任务，我们采用了keras库来辅助进行实验。这是因为Keras中的keras.preprocessing.image有包含ImageDataGenerator类，可以快速创建Python生成器，能够将硬盘上的图像文件自动转换为预处理好的张量，很方便进行图像数据的处理。下面是我们的实验代码和实验结果及其说明。
首先放上总体的代码和实验结果。


实验结果截图：
 ![image](https://user-images.githubusercontent.com/98015436/213906144-a99b5a13-28a2-419c-9b26-abd6e2130a5e.png)

（红色部分是因为我没有安装CUDA，无法使用GPU协助运算，不必理会。）

 
 
1 实验代码及其说明
1.1准备工作
我们首先在网上找到了一组关于苹果和香蕉的数据集，并将其分成了训练集、验证集、测试集三部分，如图：
 
每个部分中又包含苹果和香蕉的图片若干。为了直观的表示每个数据集中包含图片的多少，我们用了课上老师讲的关于文件路径的知识，统计了每个文件夹中的图片个数。代码和结果如下：
代码：

~~~
import os
train_dir='./训练集'
validation_dir='./验证集'

train_apples_dir='./训练集/苹果训练集'
train_bananas_dir='./训练集/香蕉训练集'
	
validation_apples_dir='./验证集/苹果验证集'
validation_bananas_dir='./验证集/香蕉验证集'

test_apples_dir='./测试集/苹果测试集'
test_bananas_dir='./测试集/香蕉测试集'


#分组情况
print('total training apple images:', len(os.listdir(train_apples_dir)))
print('total training banana images:', len(os.listdir(train_bananas_dir)))
print('total validation apple images:', len(os.listdir(validation_apples_dir)))
print('total validation banana images:', len(os.listdir(validation_bananas_dir)))
print('total testing apple images:', len(os.listdir(test_apples_dir)))
print('total testing banana images:', len(os.listdir(test_bananas_dir)))
~~~
结果：

~~~
C:\Users\K\AppData\Local\Programs\Python\Python39\python.exe E:/实验/实验.py
total training apple images: 680
total training banana images: 680
total validation apple images: 286
total validation banana images: 286
total testing apple images: 234
total testing banana images: 233
~~~ 

1.2构建网络

~~~
#构建网络
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
~~~

这里，我们采用的是conv2D和MaxPooling2D交替组合的形式。特征图的深度在逐渐增加（从32增加到128），而特征图的尺寸在逐渐减小（从150150减小到77），这几乎是所有卷积神经网络的模式。这样做还有一个好处是最后使用Flatten的时候尺寸不会太大。
因为，猫狗识别任务是一个二分类任务，所以网络最后一层使用的是sigmoid激活的单一单元（大小为1的Dense层）对某个类别的概率进行编码。
输出结果如下：
~~~
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 72, 72, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 34, 34, 128)       73856     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 17, 17, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 15, 15, 128)       147584    
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6272)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               3211776   
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 513       
=================================================================
Total params: 3,453,121
Trainable params: 3,453,121
Non-trainable params: 0
Found 1360 images belonging to 2 classes.
Found 572 images belonging to 2 classes.
~~~

下面进行编译：
~~~
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
~~~

1.3数据处理
由于图片不能直接放入神经网络中进行学习，学习之前应该把数据格式化为经过预处理的浮点数张量。所以我们将JPG图像文件解码为RGB像素网格，然后将这些像素网格转换成浮点数张量，最后将像素值（0-255范围内）缩放到[0,1]区间，因为神经网络处理较小的数据输入值较快。下面是我们的代码：
~~~
#数据处理
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)   
	test_datagen = ImageDataGenerator(rescale=1./255)

# 使用flow_from_directory()方法可以实例化一个针对图像batch的生成器
train_generator = train_datagen.flow_from_directory(
        
	        train_dir,     # 目标目录
	        
        target_size=(150, 150),      # 将所有图像大小调整为150*150
	        batch_size=20,
	        
        class_mode='binary')      # 因为使用了binary_crossentropy损失，所以需要使用二进制标签
	
	validation_generator = test_datagen.flow_from_directory(
	        validation_dir,
	        target_size=(150, 150),
	        batch_size=20,
	        class_mode='binary')
	
来看一下其中一个生成器的输出：它生成150*150的RGB图像[形状为（20， 150， 150， 3）]与二进制标签[形状为（20，）]组成的批量。每个批量包含20个样本。由于生成器会不停的生成这种批量，它会不断的循环目标文件夹中的图像，因此需要在某个时刻break迭代循环。
~~~
	for data_batch, labels_batch in train_generator:
	    print('data batch shape:', data_batch.shape)
	    print('labels batch shape:', labels_batch.shape)
     break
    
~~~
输出为：
~~~
data batch shape: (20, 150, 150, 3)
labels batch shape: (20,)
~~~

1.4模型拟合
接下来我们让模型对数据进行拟合。
对于model.fit_generator中的参数：里面含有上面代码的两个生成器train_generator和validation_generator；训练集样本总数是1360，我们设置的batch_size为20，因此steps_per_epoc取值至多为1360/20=68，我们找了一个比较好看的数64作为steps_per_epoc的值；验证集样本总数为572，因此validation_steps取值为572//20=28。
参数设置好后的代码如下：
~~~
	history = model.fit_generator(
	      train_generator,
	      steps_per_epoch=64,
	      epochs=10,
      validation_data=validation_generator,
	      validation_steps=28)
~~~

最后我们把模型存储下来，运行一下：
~~~
model.save('apples_and_bananas_1.h5')

Epoch 1/10
64/64 [==============================] - 27s 284ms/step - loss: 0.3795 - acc: 0.8675 - val_loss: 0.0164 - val_acc: 0.9982
Epoch 2/10
64/64 [==============================] - 17s 268ms/step - loss: 0.0132 - acc: 0.9991 - val_loss: 0.0025 - val_acc: 1.0000
Epoch 3/10
64/64 [==============================] - 16s 258ms/step - loss: 0.0055 - acc: 0.9990 - val_loss: 8.8209e-04 - val_acc: 1.0000
Epoch 4/10
64/64 [==============================] - 16s 257ms/step - loss: 0.0035 - acc: 1.0000 - val_loss: 1.9575e-04 - val_acc: 1.0000
Epoch 5/10
64/64 [==============================] - 17s 259ms/step - loss: 0.0092 - acc: 0.9940 - val_loss: 2.4246e-04 - val_acc: 1.0000
Epoch 6/10
64/64 [==============================] - 17s 269ms/step - loss: 0.0030 - acc: 0.9990 - val_loss: 9.6813e-05 - val_acc: 1.0000
Epoch 7/10
64/64 [==============================] - 18s 287ms/step - loss: 3.3135e-05 - acc: 1.0000 - val_loss: 3.6329e-05 - val_acc: 1.0000
Epoch 8/10
64/64 [==============================] - 18s 277ms/step - loss: 7.7388e-04 - acc: 0.9993 - val_loss: 2.4286e-05 - val_acc: 1.0000
Epoch 9/10
64/64 [==============================] - 17s 261ms/step - loss: 1.9146e-05 - acc: 1.0000 - val_loss: 4.3123e-06 - val_acc: 1.0000
Epoch 10/10
64/64 [==============================] - 17s 266ms/step - loss: 2.1675e-05 - acc: 1.0000 - val_loss: 9.7942e-07 - val_acc: 1.0000

 
2 绘制模型在训练数据和验证数据的损失函数和精度
我们使用了老师在课上讲的matplotlib库进行训练过程中模型在训练数据和验证数据的损失函数和精度的可视化。代码如下：
~~~
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
~~~

我们使用.history()方法提取每次训练中训练数据的损失函数和精度以及验证数据的损失函数和精度，作为坐标数据的来源。其中val_acc和val_loss分别表示验证集的准确度和损失函数；acc和loss分别表示训练集的准确度和损失函数。
可视化结果如下：
 ![image](https://user-images.githubusercontent.com/98015436/213906374-1d52cbad-67e9-4b7a-a32f-c3662aaa46e9.png)
![image](https://user-images.githubusercontent.com/98015436/213906380-2f2413a0-ed78-4f8d-96a6-66ede0b2bf1e.png)

 
3 测试模型
代码截图：
 ![image](https://user-images.githubusercontent.com/98015436/213906385-8a2692ff-b73e-4320-ac64-675fa1e9ead2.png)


将模型保存后，我进行了模型的测试。引用了老师上课讲的os库进行图片路径的获取；引用TensorFlow库加载模型；引用keras库进行图片处理；引用numpy库进行数据预处理。具体的代码如下：
~~~
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image

# 加载训练好的模型
model=tf.keras.models.load_model('apples_and_bananas_1.h5')

apple_test_dir = './测试集/苹果测试集'
banana_test_dir = './测试集/香蕉测试集'
	
def predict_func(data_dir,lable):
    #lable==1：香蕉；lable==0：苹果
	    t=0
	    for filename in os.listdir(data_dir):
	        image_path = os.path.join(data_dir,filename)
	        img = image.load_img(image_path , target_size=(150,150))
	
	        # 图像预处理
	        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # 对图像进行分类
	        preds = model.predict(x)
	        if preds == lable:
	            t = t+1
	        else:
            continue
	    return t/len(os.listdir(data_dir))

~~~
31.	#得到结果
~~~
	predict_result_apple = predict_func(apple_test_dir,0)
	predict_result_banana = predict_func(banana_test_dir,1)
	
	# 输出模型预测准确率
	print('Apple predicting result:', predict_result_apple)
	print('Banana predicting result:',predict_result_banana)
	~~~
结果如下：
1.	Apple predicting result: 0.79
2.	Banana predicting result: 0.82

 
4 实验反思
基本的图像分类任务是比较简单的，并且图像数据比较优良，没有过多杂乱的干扰因素，对于图像数据质量较差或实际应用场景下的识别率可能不高。
