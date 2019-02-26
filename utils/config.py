import numpy as np
import os

class Config(object):
    # Setting dataset directory
    CITYSCAPES_DATA_DIR = './data/cityscapes_dataset/cityscape/'
    ADE20K_DATA_DIR = './data/ADEChallengeData2016/'
      
    ADE20K_eval_list = os.path.join('./data/list/ade20k_val_list.txt')
    CITYSCAPES_eval_list = os.path.join('./data/list/cityscapes_val_list.txt')
    
    ADE20K_train_list = os.path.join('./data/list/ade20k_train_list.txt')
    CITYSCAPES_train_list = os.path.join('./data/list/cityscapes_train_list.txt')
    
    IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
    
    ADE20k_param = {'name': 'ade20k',
                'num_classes': 150, # predict: [0~149] corresponding to label [1~150], ignore class 0 (background) 
                'ignore_label': 0,
                'eval_size': [480, 480],
                'eval_steps': 2000,
                'eval_list': ADE20K_eval_list,
                'train_list': ADE20K_train_list,
                'data_dir': ADE20K_DATA_DIR}
    
    cityscapes_param = {'name': 'cityscapes',
                    'num_classes': 19,
                    'ignore_label': 255,
                    'eval_size': [1025, 2049],
                    'eval_steps': 500,
                    'eval_list': CITYSCAPES_eval_list,
                    'train_list': CITYSCAPES_train_list,
                    'data_dir': CITYSCAPES_DATA_DIR}
    
    model_paths = {'train': './model/cityscapes/icnet_cityscapes_train_30k.npy', 
              'trainval': './model/cityscapes/icnet_cityscapes_trainval_90k.npy',
              'train_bn': './model/cityscapes/icnet_cityscapes_train_30k_bnnomerge.npy',
              'trainval_bn': './model/cityscapes/icnet_cityscapes_trainval_90k_bnnomerge.npy',
              'others': 'snapshots/model.ckpt-500'}
    
    ## If you want to train on your own dataset, try to set these parameters.
    others_param = {'name': 'YOUR_OWN_DATASET',
                    'num_classes': 34,
                    'ignore_label': 0,
                    'eval_size': [576, 576],
                    'eval_steps': 50,
                    #'eval_list': '/media/hannes/data_disk/ubuntu/milrem-sim-3-v2/test_list.txt',
                    'eval_list': '/media/hannes/data_disk/ubuntu/milrem_cns/val-list.txt',
                    'train_list': '/media/hannes/data_disk/ubuntu/milrem-sim-3-v2/train_list.txt',
                    'data_dir': '/media/hannes/data_disk/ubuntu/milrem-sim-3-v2'}

    ## You can modify following lines to train different training configurations.
    #INFER_SIZE = [1024, 2048, 3]
    INFER_SIZE = [1024, 2048, 3]
    #TRAINING_SIZE = [720, 720]
    TRAINING_SIZE = [576, 576]
    TRAINING_STEPS = 60001
    
    N_WORKERS = 8
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    POWER = 0.9
    RANDOM_SEED = 1234
    WEIGHT_DECAY = 0.0001
    SNAPSHOT_DIR = './snapshots/'
    SAVE_NUM_IMAGES = 4
    SAVE_PRED_EVERY = 500
    
    # Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
    LAMBDA1 = 0.16
    LAMBDA2 = 0.4
    LAMBDA3 = 1.0
    
    def __init__(self, dataset, is_training=False, filter_scale=1, random_scale=False, random_mirror=False):
        print('Setup configurations...')
        
        if dataset == 'ade20k':
            self.param = self.ADE20k_param
        elif dataset == 'cityscapes':
            self.param = self.cityscapes_param
        elif dataset == 'others':
            self.param = self.others_param

        self.dataset = dataset
        self.random_scale = random_scale
        self.random_mirror = random_mirror
        self.is_training = is_training
        self.filter_scale = filter_scale
        
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not isinstance(getattr(self, a), dict):
                print("{:30} {}".format(a, getattr(self, a)))

            if a == ("param"):
                print(a)
                for k, v in getattr(self, a).items():
                    print("   {:27} {}".format(k, v))

        print("\n")