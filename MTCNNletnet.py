import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

train_path = './MT_image_label/MT_data_train/'
train_txt = './MT_image_label/MT_data_train.txt'
x_train_savepath = './MT_image_label/MT_data_x_train.npy'
y_train_savepath = './MT_image_label/MT_data_y_train.npy'

test_path = './MT_image_label/MT_data_test/'
test_txt = './MT_image_label/MT_data_test.txt'
x_test_savepath = './MT_image_label/MT_data_x_test.npy'
y_test_savepath = './MT_image_label/MT_data_y_test.npy'


def generateds(path, txt):
    f = open(txt, 'r')
    contents = f.readlines()  # 按行读取
    f.close()
    x, y_ = [], []
    for content in contents:
        value = content.split()  # 以空格分开，存入数组
        img_path = path + value[0]
        img = Image.open(img_path)
        # plt.imshow(img)  # 绘制图片让我康康img是否读入啦
        # plt.show()
        print("让我康康img的size是多大哇", img)
        img2 = img.resize((224, 224), Image.LANCZOS)
        print("让我康康img2的size是多大哇", img2)
        img3 = np.array(img2.convert('L'))
        img4 = img3 / 255.
        x.append(img4)
        y_.append(value[1])
        print('loading : ' + content)

    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)
    return x, y_



if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
        x_test_savepath) and os.path.exists(y_test_savepath):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), 224, 224))
    x_train = x_train.reshape(x_train.shape[0], 224, 224, 1)
    x_test = np.reshape(x_test_save, (len(x_test_save), 224, 224))
    x_test = x_test.reshape(x_test.shape[0], 224, 224, 1)



else:
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_path, train_txt)
    x_test, y_test = generateds(test_path, test_txt)

    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)
    # print("观察数据集结构")
    # print("训练集大小", x_train_save.shape)
    # print("训练集标签大小", y_train.shape)
    # print("测试集大小", x_test_save.shape)
    # print("测试集标签大小", y_test.shape)
    # print("输入特征", x_train_save[0])
    # print("输入特征图数组大小", x_train_save[0].shape)
    # print("输入特征图元素", x_train_save[0].size)

class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(3, 3),
                         activation='relu')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.c2 = Conv2D(filters=16, kernel_size=(3, 3),
                         activation='relu')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(120, activation='relu')
        # self.f2 = Dense(84, activation='sigmoid')
        self.f3 = Dense(6, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.f1(x)
        # x = self.f2(x)
        y = self.f3(x)
        return y


model = LeNet5()

model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/LeNet5.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=5, epochs=50, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
