import tensorflow as tf
from PIL import Image #或直接import Image
import numpy as np
# image_raw_data_jpg = tf.gfile.FastGFile('/media/king/AC88794B887914D4/PycharmProjects/UPSAMPLE/result/res3.jpg', 'rb').read()


def filter(src_img):
    filter = tf.reshape(tf.constant([1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9]), [3, 3, 1, 1])
    mean = tf.nn.depthwise_conv2d(src_img, filter, strides=[1, 1, 1, 1],  rate=[1, 1], padding='SAME')
    mean_square = tf.square(mean)
    src_img_sauqre = tf.square(src_img)
    square_mean = tf.nn.depthwise_conv2d(src_img_sauqre, filter, strides=[1, 1, 1, 1], rate=[1, 1], padding='SAME')
    var = tf.abs(square_mean - mean_square)
    var = tf.reshape(var, [64, 41, 41])
    return var


if __name__ == '__main__':
    print("hello world")