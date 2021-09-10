import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint

train_path = './MT_image_labell/MT_data_train/'
train_txt = './MT_image_labell/MT_data_train.txt'
x_data_savepath = './MT_image_labell/MT_data_x_train.npy'
y_data_savepath = './MT_image_labell/MT_data_y_train.npy'

test_path = './MT_image_test/MT_data_test_01/'
test_txt = './MT_image_test/MT_data_test_01.txt'
x_test_savepath = './MT_image_test/MT_data_x_test_01.npy'
y_test_savepath = './MT_image_test/MT_data_y_test_01.npy'


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



if os.path.exists(x_data_savepath) and os.path.exists(y_data_savepath) and os.path.exists( x_test_savepath) and os.path.exists(y_test_savepath):
    print('-------------Load Datasets-----------------')
    x_data_save = np.load(x_data_savepath)
    y_data = np.load(y_data_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_data = np.reshape(x_data_save, (len(x_data_save), 224, 224))
    x_data = x_data.reshape(x_data.shape[0], 224, 224, 1)

    x_test = np.reshape(x_test_save, (len(x_test_save), 224, 224))
    x_test = x_test.reshape(x_test.shape[0], 224, 224, 1)

    np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
    np.random.shuffle(x_data)
    np.random.seed(116)
    np.random.shuffle(y_data)
    tf.random.set_seed(116)

    print("让我瞅瞅xdata进来了米有", x_data[0])
    print(x_data.shape)
    x_train = x_data[:-50]
    y_train = y_data[:-50]
    x_val = x_data[-50:]
    y_val = y_data[-50:]
    print("让我瞅瞅数据进来了米有", x_val)

else:
    print('-------------Generate Datasets-----------------')
    x_data, y_data = generateds(train_path, train_txt)
    x_test, y_test = generateds(test_path, test_txt)

    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_data, (len(x_data), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_data_savepath, x_train_save)
    np.save(y_data_savepath, y_data)
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
    np.random.seed(115)  # 使用相同的seed，保证输入特征和标签一一对应
    np.random.shuffle(x_data)
    np.random.seed(115)
    np.random.shuffle(y_data)
    tf.random.set_seed(115)

    print("让我瞅瞅xdata进来了米有", x_data[0])
    print(x_data.shape)
    x_train = x_data[:-30]
    y_train = y_data[:-5]
    x_val = x_data[-5:]
    y_val = y_data[-5:]
    print("让我瞅瞅数据进来了米有", x_train[0])

class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(7, 7), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        # self.d1 = Dropout(0.2)  # dropout层

        super(Baseline, self).__init__()
        self.c2 = Conv2D(filters=6, kernel_size=(3, 3), padding='same')  # 卷积层
        self.b2 = BatchNormalization()  # BN层
        self.a2 = Activation('relu')  # 激活层
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        # self.d1 = Dropout(0.2)  # dropout层

        super(Baseline, self).__init__()
        self.c3 = Conv2D(filters=6, kernel_size=(3, 3), padding='same')  # 卷积层
        self.b3 = BatchNormalization()  # BN层
        self.a3 = Activation('relu')  # 激活层
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.25)  # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(64, activation='relu')
        self.f3 = Dense(4, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        # x = self.d1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        # x = self.d2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.p3(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        y = self.f3(x)
        return y


model = Baseline()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/Baseline.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=False,
                                                 save_best_only=True)


# filepath = "D:\\Users\\Administrator\\PycharmProjects\\MTAI\\model_{epoch:02d}-accuracy{loss:.2f}.h5"
# checkpoint = ModelCheckpoint(
#         filepath=filepath,
#         # monitor='val_accuracy',
#         save_best_only=True,
#         verbose=1,
#         save_weights_only=False,
#         period=3
#     )
# # checkpoint.save('./save/model.ckpt')

history = model.fit(x_train, y_train, batch_size=4, epochs=1, validation_data=(x_val, y_val), validation_freq=1, callbacks=[cp_callback]
                    )
model.summary()

loss, accuracy = model.evaluate(x_test, y_test)
print('\ntest loss', loss)
print('accuracy', accuracy)
# model.save('model.h5')

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
#  plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
#  plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.figure(2)
plt.subplot(1, 2, 1)
plt.plot(val_acc, label='validation Accuracy')
plt.title('validation Accuracy')
plt.grid()
ax1 = plt.subplot(121)
ax1.set_ylim([0,1])
plt.legend()
plt.show()

plt.subplot(1, 2, 2)
plt.plot(val_loss, label='Validation Accuracy')
plt.title('Validation Loss')
plt.grid()
plt.legend()
plt.show()