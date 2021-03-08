import time
#import myutils
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from keras.layers import Input, Dense, AveragePooling2D, GlobalAveragePooling2D
from keras import backend as K

tf.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()

_id = 0
networks = ['incv3', 'resnet50', 'vgg16', 'vgg19']
selected_network = networks[_id]

# Load Cifar-10 data
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

n_training = X_train.shape[0]
n_testing = X_test.shape[0]

y_train = y_train.flatten()
y_test  = y_test.flatten()

# Create model
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50     import ResNet50
from keras.applications.vgg16        import VGG16
from keras.applications.vgg19        import VGG19

input_shape = {
    'incv3'   : (299,299,3),
    'resnet50': (224,224,3),
    'vgg16'   : (224,224,3),
    'vgg19'   : (224,224,3)
}[selected_network]

def create_model_incv3():
    tf_input = Input(shape=input_shape)
    model = InceptionV3(input_tensor=tf_input, weights='imagenet', include_top=False)
    output_pooled = AveragePooling2D((8, 8), strides=(8, 8))(model.output)
    return Model(model.input, output_pooled)

def create_model_resnet50():
    tf_input = Input(shape=input_shape)
    return ResNet50(input_tensor=tf_input, include_top=False)

def create_model_vgg16():
    tf_input = Input(shape=input_shape)
    model = VGG16(input_tensor=tf_input, include_top=False)
    output_pooled = AveragePooling2D((7, 7))(model.output)
    return Model(model.input, output_pooled )

def create_model_vgg19():
    tf_input = Input(shape=input_shape)
    model = VGG19(input_tensor=tf_input, include_top=False)
    output_pooled = AveragePooling2D((7, 7))(model.output)
    return Model(model.input, output_pooled )

create_model = {
    'incv3'    : create_model_incv3,
    'resnet50' : create_model_resnet50,
    'vgg16'    : create_model_vgg16,
    'vgg19'    : create_model_vgg19
}[selected_network]

# Data generator for tensorflow
batch_of_images_placeholder = tf.placeholder("uint8", (None, 32, 32, 3))
#batch_of_images_placeholder = tf.compat.v1.placeholder("uint8", (None, 32, 32, 3))

batchSize = {
    'incv3'    : 10,
    'resnet50' : 10,
    'vgg16'    : 10,
    'vgg19'    : 10
}[selected_network]

#tf_resize_op = tf.image.resize_images(batch_of_images_placeholder, (input_shape[:2]), method=0)
tf_resize_op = tf.image.resize(batch_of_images_placeholder, (input_shape[:2]))

from keras.applications.inception_v3 import preprocess_input as incv3_preprocess_input
from keras.applications.resnet50     import preprocess_input as resnet50_preprocess_input
from keras.applications.vgg16        import preprocess_input as vgg16_preprocess_input
from keras.applications.vgg19        import preprocess_input as vgg19_preprocess_input

preprocess_input = {
    'incv3'   : incv3_preprocess_input,
    'resnet50': resnet50_preprocess_input,
    'vgg16'   : vgg16_preprocess_input,
    'vgg19'   : vgg19_preprocess_input
}[selected_network]

def data_generator(sess,data,labels):
    def generator():
        start = 0
        end = start + batchSize
        n = data.shape[0]
        while True:
            batch_of_images_resized = sess.run(tf_resize_op, {batch_of_images_placeholder: data[start:end]})
            batch_of_images__preprocessed = preprocess_input(batch_of_images_resized)
            batch_of_labels = labels[start:end]
            start += batchSize
            end   += batchSize
            if start >= n:
                start = 0
                end = batchSize
            yield (batch_of_images__preprocessed, batch_of_labels)
    return generator

# Feature extraction

with tf.Session() as sess:
    # setting tensorflow session to Keras
    #K.set_session(sess)
	tf.compat.v1.keras.backend.set_session(sess)
    # setting phase to training
	#K.set_learning_phase(0)  # 0 - test,  1 - train

	model = create_model()

	data_train_gen = data_generator(sess, X_train, y_train)
	ftrs_training = model.predict(x=data_train_gen(), steps=int(50000/batchSize), verbose=1)

	data_test_gen = data_generator(sess, X_test, y_test)
	ftrs_testing = model.predict(x=data_test_gen(), steps=int(10000/batchSize), verbose=1)

