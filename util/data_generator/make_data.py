import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import functools
import sys
from sklearn.model_selection import KFold

sess = tf.compat.v1.Session()
np.set_printoptions(threshold=sys.maxsize)

folder = r'E:\Users\yuan\MasterThesis\TBH\data\test\MS-COCO'
test_names = ['test_batch.mat']
train_names = ['data_batch_1.mat', 'data_batch_2.mat', 'data_batch_3.mat', 'data_batch_4.mat', 'data_batch_5.mat']

#feat_dim = 500
#feat_dim = 512
feat_dim = 2048
#feat_dim = 3072
#feat_dim = 4096
#class_num = 10
#class_num = 21
class_num = 79

num_folds = 6

def _int64_feature(value):
    """Create a feature that is serialized as an int64."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def read_data(files):
    num = 0	
    for f in files:
        print("processing: " + f)
        name = os.path.join(folder, f)
        data_mat = sio.loadmat(name)
        if num==0: 
            feat = data_mat['data']/255.0
            label = data_mat['labels']
        else:
            feat = np.concatenate((feat, data_mat['data']/255.0))
            label = np.concatenate((label, data_mat['labels']))
        num += 1
    label = np.eye(class_num)[np.squeeze(label)]
    fid = np.arange(0, feat.shape[0])
    return {'feat': feat, 'label':label, 'fid':fid}

def convert_data(files, name):
    out_path = os.path.join(folder, 'out')
    if not os.path.exists(out_path): os.makedirs(out_path)
    file_name = os.path.join(out_path, name+'.tfrecords')
    writer = tf.io.TFRecordWriter(file_name)

    data = read_data(files)
    size = data['feat'].shape[0]
    for i in range(size):
        this_id = _int64_feature(data['fid'][i])
        this_feat = _float_feature(data['feat'][i, :])
        this_label = _float_feature(data['label'][i, :])
        feat_dict = {'id': this_id,
                     'feat': this_feat,
                     'label': this_label}
        feature = tf.train.Features(feature=feat_dict)
        example = tf.train.Example(features=feature)
        writer.write(example.SerializeToString())
    writer.close()

def convert_data_from_npz(name):
    print("Processing data from {} trained".format(name))
    def write_data(f_name, data, label):
        writer = tf.io.TFRecordWriter(f_name)
        size = data.shape[0]
        for i in range(size):
            this_id = _int64_feature(i)
            this_feat = _float_feature(data[i, :])
            #this_label = _float_feature(np.eye(class_num)[label[i]]) # enable this when processing CIFAR-10 dataset
            this_label = _float_feature(label[i,:])
            feat_dict = {'id': this_id,
                        'feat': this_feat,
                        'label': this_label}
            feature = tf.train.Features(feature=feat_dict)
            example = tf.train.Example(features=feature)
            writer.write(example.SerializeToString())
        writer.close()

    out_path = os.path.join(folder, 'out_'+name)
    if not os.path.exists(out_path): os.makedirs(out_path)
    #data = np.load(os.path.join(folder, "CIFAR10_{}-keras.npz".format(name)))
    data = np.load(os.path.join(folder, "{}.npz".format(name)))
        
    # define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    inputs = data['features']
    targets = data['labels']
    for train, test in kfold.split(inputs, targets):
        print(" running the fold {}".format(fold_no))
        #print(train)
        #print(test)
        file_name = os.path.join(out_path, 'train_{}.tfrecords'.format(fold_no))
        train_data = inputs[train]
        train_label = targets[train]
        write_data(file_name, train_data, train_label)
        
        file_name = os.path.join(out_path, 'test_{}.tfrecords'.format(fold_no))
        test_data = inputs[test]
        test_label = targets[test]
        write_data(file_name, test_data, test_label)
        fold_no += 1

#def parse_data(out_f, tf_example: tf.train.Example):
def parse_data(tf_example: tf.train.Example):
    feat_dict = {'id': tf.io.FixedLenFeature([], tf.int64),
                 'feat': tf.io.FixedLenFeature(feat_dim, tf.float32),
                 'label': tf.io.FixedLenFeature(class_num, tf.float32)}
    features = tf.io.parse_single_example(tf_example, features=feat_dict)
    return features

def disp_data(p,f):
    print("processing {}".format( os.path.join(p,f) ))
    #out_path = os.path.join(folder, 'out', p)
    out_path = os.path.join(folder, p)
    in_f = os.path.join(out_path, f)
    raw_dataset = tf.data.TFRecordDataset(in_f)
    data = raw_dataset.map(parse_data)
    with open(os.path.join(out_path, f+".txt"),"w") as f:
        for element in data.take(10):
            print(element["id"].numpy())
            print(type(element["id"].numpy()))
            _id = element["id"].numpy()
            _label = element["label"].numpy()
            #_label = np.where(_label==1)[0]
            _feat = element["feat"].numpy()
            f.write("{}:\n".format(_id))
            f.write(' '.join( [str(e) for e in _label] ))
            f.write('\n')
            f.write(' '.join( [str(e) for e in _feat] ))
            f.write('\n')
    #f.write(np.array2string(_feat, precision=1, separator=','))

convert_data_from_npz("MS-COCO_incv3-all")
#disp_data("out_NUS-WIDE_incv3-all", "test_1.tfrecords")
