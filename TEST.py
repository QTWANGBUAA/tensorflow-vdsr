import numpy as np
from scipy import misc
from PIL import Image
import tensorflow as tf
import glob, os, re
from PSNR import psnr
import scipy.io
import pickle
import h5py
from MODEL import model
# from MODEL_FACTORIZED import model_factorized
import time

DATA_PATH = "./data/test/"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path


def get_img_list(data_path):
    l = glob.glob(os.path.join(data_path, "*"))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    train_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4] + "_2.mat"): train_list.append([f, f[:-4] + "_2.mat", 2])
            if os.path.exists(f[:-4] + "_3.mat"): train_list.append([f, f[:-4] + "_3.mat", 3])
            if os.path.exists(f[:-4] + "_4.mat"): train_list.append([f, f[:-4] + "_4.mat", 4])
    return train_list


def get_test_image(test_list, offset, batch_size):
    target_list = test_list[offset:offset + batch_size]
    input_list = []
    gt_list = []
    scale_list = []
    for pair in target_list:
        print(pair[1])
        mat_dict = scipy.io.loadmat(pair[1])
        input_img = None
        if "img_2" in mat_dict:
            input_img = mat_dict["img_2"]
        elif "img_3" in mat_dict:
            input_img = mat_dict["img_3"]
        elif "img_4" in mat_dict:
            input_img = mat_dict["img_4"]
        else:
            continue
        print(pair[0])
        gt_img = scipy.io.loadmat(pair[0])['img_raw']
        input_list.append(input_img)
        gt_list.append(gt_img)
        scale_list.append(pair[2])
    return input_list, gt_list, scale_list


def test_VDSR_with_sess(epoch, data_path, sess):
    folder_list = glob.glob(os.path.join(data_path, 'Set*'))
    print('folder_list', folder_list)
    saver.restore(sess, tf.train.latest_checkpoint("./checkpoints/"))

    psnr_dict = {}
    for folder_path in folder_list:
        psnr_list = []
        img_list = get_img_list(folder_path)
        for i in range(len(img_list)):
            input_list, gt_list, scale_list = get_test_image(img_list, i, 1)
            input_y = input_list[0]
            gt_y = gt_list[0]
            start_t = time.time()
            img_vdsr_y = sess.run([output_tensor], feed_dict={
                input_tensor: np.resize(input_y, (1, input_y.shape[0], input_y.shape[1], 1))})
            img_vdsr_y = np.resize(img_vdsr_y, (input_y.shape[0], input_y.shape[1]))
            end_t = time.time()
            print("end_t", end_t, "start_t", start_t)
            print("time consumption", end_t - start_t)
            print("image_size", input_y.shape)

            psnr_bicub = psnr(input_y, gt_y, scale_list[0])
            psnr_vdsr = psnr(img_vdsr_y, gt_y, scale_list[0])
            print("PSNR: bicubic %f\tVDSR %f" % (psnr_bicub, psnr_vdsr))
            psnr_list.append([psnr_bicub, psnr_vdsr, scale_list[0]])
        psnr_dict[os.path.basename(folder_path)] = psnr_list


# with open('psnr/%s' % os.path.basename(ckpt_path), 'wb') as f:
# 	pickle.dump(psnr_dict, f)
def test_VDSR(epoch, ckpt_path, data_path):
    with tf.Session() as sess:
        test_VDSR_with_sess(epoch, data_path, sess)


if __name__ == '__main__':
    # model_list = sorted(glob.glob("./checkpoints/VDSR_const_clip_0.01_epoch_000.ckpt-3364"))
    # print(1)
    # model_list = [fn for fn in model_list if not os.path.basename(fn).endswith("meta")]
    with tf.Session() as sess:
        input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
        shared_model = tf.make_template('shared_model', model)
        output_tensor, weights = shared_model(input_tensor)
        # output_tensor, weights 	= model(input_tensor)
        saver = tf.train.Saver(weights)
        print(1)
        tf.global_variables_initializer()
        print(1)
        # for model_ckpt in model_list:
        #     print("hello")
        #     print(model_ckpt)
        #     epoch = int(model_ckpt.split('epoch_')[-1].split('.ckpt')[0])
        #     # if epoch<60:
        #     #	continue
        #     print("Testing model", model_ckpt)
        test_VDSR_with_sess(80, DATA_PATH, sess)

