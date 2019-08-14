'''
这段代码用于展示存储模型
'''

import os
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images[:1000].reshape(-1, 28*28)/255.0
train_labels = train_labels[:1000]

test_images = test_images[:1000].reshape(-1, 28*28)/255.0
test_labels = test_labels[:1000]

def create_model():
    model = keras.Sequential([
        keras.layers.Dense(512,activation=tf.nn.relu, input_shape=(784, )), 
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    return model

model = create_model()
model.summary()

# 没有任何特殊设置，只保存最终训练完成的模型
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[cp_callback])

# 创建一个新的模型
model2 = create_model()

loss, acc = model2.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# 加载模型
model2.load_weights(checkpoint_path)
loss, acc = model2.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# 配置检查点回调
# 检查点默认每个训练周期结束生成一个检查点
# 此处为每个检查点设置唯一的名称，并设置创建频率为每5个周期创建一次
# 其中period参数已过期，新的参数是save_freq这个参数按照样本数来确定保存周期，所以样本总数的5倍就是5个间隔。可以使用更小的粒度来得到中间的模型
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True,save_freq=5000)
model = create_model()
model.fit(train_images, train_labels, epochs=50, callbacks=[cp_callback], validation_data=(test_images, test_labels), verbose=0)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# 加载其中最后一个模型(可能训练中途终端，最后一个不一定是最终的那个)
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# 手动保存模型, 相对于使用检查点会掉自动保存模型而言
model.save_weights('./checkpoints/my_checkpoint')

model = create_model()
model.load_weights('./checkpoints/my_checkpoint')
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# 保存整个模型， 之前的保存只保存权重， 而整个模型保存可以保存权重，超参，优化器配置等，保存整个模型，则无需再为新模型配置源码，可以把python的模型在js中加载
# 默认使用HDF5格式保存整个模型
# 注意， keras无法保存tensorflow的优化器
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('my_model.h5')

# 加载整个模型
new_model = keras.models.load_model('my_model.h5')
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
