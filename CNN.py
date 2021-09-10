#    这是一个用keras实现的CNN神经网络，可以通过更改卷积层数、全连接网络层数等来实现自己的神经网络

import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#    在这里，我们可以输入你自己的训练集数据、标签以及测试集数据与标签的文件储存路径
#    训练集（后续我们会从训练集中分割出验证集）
data_path = './MT_image_labell/MT_data_train/'  # 训练集数据文件地址
data_txt = './MT_image_labell/MT_data_train.txt'  # 训练集标签地址
x_data_savepath = './MT_image_labell/MT_data_x_train.npy'   # 一般在数据集、标签的文件路径下
y_data_savepath = './MT_image_labell/MT_data_y_train.npy'
#  测试集存储路径
test_path = './MT_image_test/MT_data_test_01/'
test_txt = './MT_image_test/MT_data_test_01.txt'
x_test_savepath = './MT_image_test/MT_data_x_test_01.npy'
y_test_savepath = './MT_image_test/MT_data_y_test_01.npy'



#   这里通过自定义函数将训练集与验证集的数据读进程序
def generateds(path, txt):  # 第一个参数是数据集的路径，第二个参数是数据集标签的路径

    f = open(txt, 'r')  # 打开数据集标签文件
    contents = f.readlines()  # 按行读取储存在contents对象中
    f.close()
    x, y_ = [], []  # 建立两个空字典

    for content in contents:   # 类比for i in y
        value = content.split()  # 将每一行以空格分开，存入数组。其中value【0】是图片的文件名，value【1】是标签
        img_path = path + value[0]  # 得到每一张图片的路径= 数据集文件夹+ 标签的第一列
        img = Image.open(img_path)  # 读入图片数据
        # plt.imshow(img)  【调试代码】绘制图片让我康康图片是否读入啦
        # plt.show()
        # print("让我康康img的size是多大哇", img)   【调试代码】
        img2 = img.resize((224, 224), Image.LANCZOS)  # 自制数据集图像数据的大小也许不一样，我们需要将之调节为一样的大小
        # print("让我康康img2的size是多大哇", img2)   【调试代码】
        img3 = np.array(img2.convert('L'))   # 将之转为灰度
        img4 = img3 / 255.  # 将之归一化
        x.append(img4)  # 将读入的图片数据存入x空字典中
        y_.append(value[1])  # 将读入的标签数据存入y_空字典中
        print('loading : ' + content)

    x = np.array(x)  # 数据格式转换
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)
    return x, y_


# 接下来我们调用generateds函数来把数据集给加载进来
if os.path.exists(x_data_savepath) and os.path.exists(y_data_savepath) and os.path.exists(
        x_test_savepath) and os.path.exists(y_test_savepath):   # 看缓存文件是否存在

    #  缓存文件已经存在的情况下，即你已经将数据集读入且生成npy文件的情况下
    print('-------------Load Datasets-----------------')
    #  分别加载训练集与测试集
    x_data_save = np.load(x_data_savepath)
    y_data = np.load(y_data_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)

    #  更改图片数据矩阵的维度
    x_data = np.reshape(x_data_save, (len(x_data_save), 224, 224))
    x_data = x_data.reshape(x_data.shape[0], 224, 224, 1)  # 在这里，如果是彩色图片，请把1通道改为3通道
    x_test = np.reshape(x_test_save, (len(x_test_save), 224, 224))
    x_test = x_test.reshape(x_test.shape[0], 224, 224, 1)  # 同上

    # 打乱数据集。 并且，将原训练集分割为训练集与验证集
    np.random.seed(116)  # 使用相同的seed，保证 输入特征和标签一一对应，seed之后的数值可以随意改变
    np.random.shuffle(x_data)
    np.random.seed(116)
    np.random.shuffle(y_data)
    tf.random.set_seed(116)
    np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
    np.random.shuffle(x_test)
    np.random.seed(116)
    np.random.shuffle(y_test)
    tf.random.set_seed(116)

    #  将原训练集分割为 训练集与验证集
    x_train = x_data[:-50]  # 训练集取后50张之前的  50这个参数依据你的自制数据集来调整即可
    y_train = y_data[:-50]
    x_val = x_data[-50:]  # 训练集取后50张
    y_val = y_data[-50:]
    print("让我瞅瞅数据进来了米有", x_val)

#  这是第一次加载数据集的情况下
else:
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(data_path, data_txt)  # 调用generateds函数
    x_test, y_test = generateds(test_path, test_txt)

    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)


#  在这里，我们定义卷积神经为网络的基本模型
class Baseline(Model):
    def __init__(self):

        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=12, kernel_size=(5, 5), padding='same')  # 卷积层。参数分别为：卷积核个数、卷积核大小、是否进行全零填充
        self.b1 = BatchNormalization()  # BN层 批标准化操作
        self.a1 = Activation('relu')  # 激活层，可以更换激活函数
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层。参数分别为：池化窗口大小，池化步长，是否进行全零填充
        self.d1 = Dropout(0.25)  # dropout层
        #  可以仿照以上结构定义更多的卷积层与池化层

        # super(Baseline, self).__init__()
        # self.c2 = Conv2D(filters=6, kernel_size=(3, 3), padding='same')  # 卷积层
        # self.b2 = BatchNormalization()  # BN层
        # self.a2 = Activation('relu')  # 激活层
        # self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        # # self.d2 = Dropout(0.25)  # dropout层
        #
        # super(Baseline, self).__init__()
        # self.c3 = Conv2D(filters=16, kernel_size=(3, 3), padding='same')  # 卷积层
        # self.b3 = BatchNormalization()  # BN层
        # self.a3 = Activation('relu')  # 激活层
        # self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        # self.d3 = Dropout(0.25)  # dropout层

        # super(Baseline, self).__init__()
        # self.c4 = Conv2D(filters=6, kernel_size=(3, 3), padding='same')  # 卷积层
        # self.b4 = BatchNormalization()  # BN层
        # self.a4 = Activation('relu')  # 激活层
        # self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        # self.d4 = Dropout(0.25)  # dropout层

        # 在这里定义全连接层
        self.flatten = Flatten()  # 拉直操作
        self.f1 = Dense(128, activation='relu')  # 第一层全连接层 参数为神经元个数、激活函数
        self.d5 = Dropout(0.25)   # droupout层
        self.f2 = Dense(64, activation='relu')  # 第二层全连接层
        self.f3 = Dense(4, activation='softmax') # 第三层全连接层

    #  下面是对定义好的神经网络进行调用的函数
    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        # x = self.c2(x)
        # x = self.b2(x)
        # x = self.a2(x)
        # x = self.p2(x)
        # # x = self.d2(x)
        #
        # x = self.c3(x)
        # x = self.b3(x)
        # x = self.a3(x)
        # x = self.p3(x)
        # x = self.d3(x)

        # x = self.c4(x)
        # x = self.b4(x)
        # x = self.a4(x)
        # x = self.p4(x)
        # x = self.d4(x)

        x = self.flatten(x)

        x = self.f1(x)
        x = self.d5(x)
        y = self.f2(x)
        y = self.f3(x)
        return y


model = Baseline()
# 为神经网络配备优化器、损失函数
model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 在这里实现断点续训
checkpoint_save_path = "./checkpoint/Baseline.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
#  这行代码调用我们定义好的网络，开始神经网络训练
history = model.fit(x_train, y_train, batch_size=20, epochs=4, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()
#  在训练结束之后，直接将该模型用于测试集，输出测试集损失值与准确度
loss, accuracy = model.evaluate(x_test, y_test)
print('\ntest loss', loss)
print('accuracy', accuracy)

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.title('Training accuracy')
plt.grid()
ax1 = plt.subplot(121)
ax1.set_ylim([0, 1.2])
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='training loss')
plt.title('training Loss')
plt.grid()
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


plt.subplot(1, 2, 2)
plt.plot(val_loss, label='Validation Accuracy')
plt.title('Validation Loss')
plt.grid()
plt.legend()
plt.show()




#  把得到的训练集损失值、训练集精确度、验证集损失值、验证集精确度数据保存在txt文件中
def Save_list (list1, filename):
    file2 = open(filename + '.txt', 'w')
    for i in list1:
        # c = str(i).replace("'", '').replace(',', '') + '\n'
        c = str(i)
        file2.write(c+'\n')                             # 写完一行立马换行
    file2.close()


Save_list(acc, 'acc')
Save_list(val_acc, 'val_acc')
Save_list(loss, 'loss')
Save_list(val_loss, 'val_loss')


