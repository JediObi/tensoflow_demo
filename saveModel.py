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

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[cp_callback])

# 创建一个新的模型
model2 = create_model()

loss, acc = model2.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

model2.load_weights(checkpoint_path)
loss, acc = model2.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))