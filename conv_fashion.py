'''
本文的2D卷积神经网络在fashion-mnist训练集上的正确率可以达到94%,损失0.16, 在测试集上的正确率可以达到92%，损失0.24
使用1D卷积层神经网络训练集正确率91%，损失0.25，测试集上正确率89%，损失0.30
'''
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# 打印tensorflow版本89
print("tensorflow版本: ",tf.__version__)

# 导入数据集
fashion_mnist = keras.datasets.fashion_mnist

# 加载训练集和验证集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 衣物名称，样本标签是0-9的数字不含语义sunshi
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(True)
# plt.show()

# 像素规则化
train_images = train_images / 255.0
test_images = test_images / 255.0

# 由于使用2D卷积，所以把图片的通道信息加上，由于使用的是单通道，所以直接用1
# train_images = train_images.reshape(train_images.shape[0], 28,28,1)
# test_images = test_images.reshape(test_images.shape[0], 28,28,1)

# 使用2D卷积层， 第一层使用32个滤波器（相当于输出深度32）， 
# input_shape第一层时必须指定这个参数，训练集原始shape(60000,28,28)，2D需要输入图像(长，宽，通道)，
# 灰度图像只有一个通道，所以相对于dataformat=channels_last，reshape为(60000,28,28,1)
# 可以考虑1D卷积层，kernel为一维，inputshape二维的情形
# 创建模型，Flatten层第一层输入层扁平化多维数组为一维， Dense第一层全连接层，128个神经元所以输出也是128个使用relu，第三层Dense10个神经元作为输出层使用softmax计算概率
model = keras.Sequential([
    keras.layers.Conv1D(32, 3, padding='same', activation=tf.nn.relu, input_shape=(28, 28)),
    keras.layers.MaxPooling1D(2, strides=2),
    keras.layers.Conv1D(64, 3, padding='same', activation=tf.nn.relu),
    keras.layers.MaxPooling1D(2, strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])
# 编译模型，损失函数，优化器，评估方式数组（当前数组只有accuracy表示只使用成功率）
# metrics 评估函数类似于损失函数，但评估函数不参与模型更新
# 评估函数在一个训练迭代结束时评估当前模型的性能，比如accuracy会计算当前模型在本次训练集合上的正确率。不同的评估函数关注点会有不同。
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型 训练数据集， 训练集的标签， 迭代次数
model.fit(train_images, train_labels, epochs=5)

# 在测试机上评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test loss: ', test_loss)
print('Test accuracy: ', test_acc)

# 进行预测
predictions = model.predict(test_images)
# 打印第一张图片的预测结果(由于输出层采用softmax，所以会输出所有分类的概率)
print("first prediction: ",predictions[0])
# 打印第一张图预测概率最高的下标，也就是打印最高概率预测分类
print("first prediction: ", np.argmax(predictions[0]))
# 打印第一张图的真实标签
print("first real label: ", test_labels[0])

# 打印图片和预测结果（最高概率分类），并把预测结果语义化
def plot_image(i, predictions_array, true_label, img):
    # 
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    # img = img.reshape((28,28))
    # 网格关掉
    plt.grid(False)
    # 坐标轴刻度
    plt.xticks([])
    plt.yticks([])

    # 显示图片
    plt.imshow(img, cmap=plt.cm.binary)

    # 预测最大概率的结果
    predicted_label = np.argmax(predictions_array)
    # 如果预测结果与真实标签相同则显示蓝色，不相同则显示红色
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    # 设置x轴名称， 预测结果， 预测概率最大值，括号里展示真实标签， 如果预测正确文字为蓝色，如果预测错误文字为红色 
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100*np.max(predictions_array),class_names[true_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    # 产生10个柱图，展示预测概率，颜色为灰色
    thisplot = plt.bar(range(10),predictions_array, color = '#777777')
    # y轴上下线
    plt.ylim([0,1])
    # 取预测结果最大值的标签
    predicted_label = np.argmax(predictions_array)

    # 预测结果标签对应的柱图设置为红色，前边的柱图有十个柱子对应了0-9的标签
    thisplot[predicted_label].set_color('red')
    # 真实标签对应的柱图设置为绿色
    thisplot[true_label].set_color('green')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
# 展示第一个图形，和它的预测结果，x轴红色字体为错误，蓝色为正确
plot_image(i,predictions, test_labels, test_images)
# 创建一个subplot，展示第一个图形的预测概率柱图，灰色是普通概率，红色是预测概率，蓝色是真实标签对应的柱状图的预测概率
plt.subplot(1,2,2)
plot_value_array(i,predictions,test_labels)
plt.show()

    