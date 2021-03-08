import sys, os
sys.path.append('..')
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import numpy as np
from model.tbh import TBH
from util.data.dataset import Dataset
from util.eval_tools import eval_cls_map
from util.data.set_processor import SET_SIZE
#np.set_printoptions(threshold=sys.maxsize)

def update_codes(model, data, batch_size, set_name):
    """
	update binary codes for all data
	param model: trained model or the model loaded with pre-trained weights
	param data: loaded data
	"""
    train_iter = iter(data.train_data)
    test_iter = iter(data.test_data)

    for num in range(int(SET_SIZE[set_name][1]/batch_size) + 1):
        test_batch = next(test_iter)
        data.update(test_batch[0].numpy(), model([test_batch], training=False).numpy(), test_batch[2].numpy(), 'test')
        
    for num in range(int(SET_SIZE[set_name][0]/batch_size) + 1):
        train_batch = next(train_iter)
        data.update(train_batch[0].numpy(), model([train_batch], training=False).numpy(), train_batch[2].numpy(), 'train')

def make_PR_plot(path, data):
    """
	plot PR curve
	param path: folder where the plot should be saved
	param data: 2-D array, data[0,:] precision list, data[1,:] recall list, should be sorted w.r.t. recall list
	"""
    plt.figure()
    plt.step(list(data[1,:]), list(data[0,:]), where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.savefig(os.path.join(path, 'P-R.png'))
    plt.close()
    with open(os.path.join(path,'PRLists.data'), 'wb') as f:
        pickle.dump(list(data[0,:]), f)
        pickle.dump(list(data[1,:]), f)

def load_weights(set_name, bbn_dim, cbn_dim, batch_size, middle_dim, path):
    """
    create model and load it with pre-trained weights stored under the path
	"""
    print("loading weights from {}".format(path))
    model = TBH(set_name, bbn_dim, cbn_dim, middle_dim)
    data = Dataset(set_name=set_name, batch_size=batch_size, shuffle=False)

    actor_opt = tf.keras.optimizers.Adam(1e-4)
    critic_opt = tf.keras.optimizers.Adam(1e-4)

    checkpoint = tf.train.Checkpoint( actor_opt=actor_opt, critic_opt=critic_opt, model=model )
    checkpoint.restore(tf.train.latest_checkpoint(path))
    update_codes(model,data,batch_size,set_name)
    
    print("updating codes for the dataset and plot PR code...")
    test_hook,test_precision,pr_curve = eval_cls_map(data.test_code, data.train_code, data.test_label, data.train_label, 1000, True)
    make_PR_plot(path, pr_curve)

if __name__ == '__main__':
    load_weights('cifar10', 32, 512, 500, 1024, "/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/TBH/result/cifar10/model/Mon08Mar2021-102000/")

