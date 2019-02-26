import argparse
import time

import tensorflow as tf
import numpy as np
from tqdm import trange
import os

from utils.config import Config
from utils.image_reader import ImageReader
from model import ICNet, ICNet_BN
from PIL import Image
import cv2
# mapping different model
model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced ICNet")

    parser.add_argument("--model", type=str, default='',
                        help="Model to use.",
                        choices=['train', 'trainval', 'train_bn', 'trainval_bn', 'others'],
                        required=True)
    parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes', 'others'],
                        required=True)
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])
    parser.add_argument("--out", action="store_true", help="Saves prediction images to ./out/")
    parser.add_argument("--all", action="store_true", help="Evaluate over all images")

    return parser.parse_args()

def main():
    args = get_arguments()  
    cfg = Config(dataset=args.dataset, is_training=False, filter_scale=args.filter_scale)
    
    model = model_config[args.model]

    reader = ImageReader(cfg=cfg, mode='eval')
    net = model(image_reader=reader, cfg=cfg, mode='eval')
    
    # mIoU
    pred_flatten = tf.reshape(net.output, [-1,])

    label_flatten = tf.reshape(net.labels, [-1,])

    mask = tf.not_equal(label_flatten, cfg.param['ignore_label'])
    indices = tf.squeeze(tf.where(mask), 1)
    gt = tf.cast(tf.gather(label_flatten, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    if cfg.dataset == 'ade20k':
        pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes']+1)
    elif cfg.dataset == 'cityscapes':
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'])
    elif cfg.dataset == 'others':
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'])
    
    net.create_session()
    net.restore(cfg.model_paths[args.model])
    try:
        if args.out:
            os.makedirs("out", exist_ok=True)
            print("Directing predictions to files")
            i = 0
            for name in reader.image_list:
                name = os.path.basename(name)
                print(name)
                out = net.sess.run(net.output)
                i += 1
                cv2.imwrite(f"out/eval_{name}{'' if name.endswith('.png') else '.png'}", out[0,:,:,0])
        else:
            if args.all:
                for i in trange(len(reader.image_list), desc='evaluation', leave=True):
                    _ = net.sess.run(update_op)
            else:
                for i in trange(cfg.param['eval_steps'], desc='evaluation', leave=True):
                    _ = net.sess.run(update_op)
            print('mIoU: {}'.format(net.sess.run(mIoU)))
    except tf.errors.OutOfRangeError:
        print("Out of images", i)

             

    
if __name__ == '__main__':
    main()
