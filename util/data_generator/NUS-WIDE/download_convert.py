import requests
import numpy as np
import os,sys
import re
from PIL import Image
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.inf)

label_file = "./Labels21.npz"
image_list = "./Imagelist.txt"
mapping_file = "./mapping.npz"
links_file = "./NUS-WIDE-urls.txt"

folder = r"E:\Users\yuan\MasterThesis\TBH\data\test\out_NUS-WIDE_incv3-keras"

total_images = 269648

num_test = 100 #  *28
num_train = 2000 # *28

def download_pic(url, name):
    try:
        response = requests.get(url)

        f = open("./{}.jpg".format(name),"wb")
        f.write(response.content)
        f.close()
    except Exception as e:
        print("Error: {}".format(e.message))

def save_map(f_name):
    mapping = np.zeros(total_images)
    with open(f_name, "r") as f:
        num = 0
        for line in f:
            x = re.search(r'_(\d+)\.', line.strip('\n'))
            mapping[num] = int(x.group(1))
            num += 1
    np.savez_compressed("mapping.npz", mapping = mapping)

def construct_id_link(link_file):
    d = {}
    with open(link_file, "r") as f:
        num = 0
        for line in f:
            if(num > 0):
                l = line.strip('\n').strip(' ').split()
                print(l)
                d[int(l[1])] = [l[2], l[3], l[4], l[5]]
            num += 1
    return d
def get_link(l):
    for i in [2,1,0,3]:
        if('http' in l[i]):
            return l[i]
    return None

def download_relevent_pics(label_file, mapping_file, link_file):
    mapping = np.load(mapping_file)["mapping"]
    links = construct_id_link(link_file)
    print(links)
    label = np.load(label_file)["label"]
    label = np.sum(label,axis=1)

    for i in range(label.shape[0]):
        if(label[i]>0):
            pid = int(mapping[i])
            print("{}. downloading photo id {}".format(i, pid))
            link = get_link(links[pid])
            if link: download_pic(link, "{}_{}".format(i, pid))

def convert_to_array(label_file, mapping_file, folder):
    mapping = np.load(mapping_file)["mapping"]
    labels = np.load(label_file)["label"]
    label_sum = np.sum(labels,axis=1)

    cnt  = 0
    num = 0
    f_cnt = 0
    print("Start processing ...")
    for i in range(label_sum.shape[0]):
        if (num+1)%1000 == 0: print("{} finished.".format(num+1))
        if (cnt+1)%5000 == 0: # save and restart
            np.savez_compressed("data_{}.npz".format(f_cnt),\
                        data = data,\
                        label = _labels)
            f_cnt += 1
            cnt = 0
            
        num += 1
        if(label_sum[i]>0):
            pid = int(mapping[i])
            p_name = "{}_{}.jpg".format(i,pid)
            p_path = os.path.join(folder,"pics", p_name)
            if os.path.isfile(p_path):
                img = Image.open(p_path).resize((256,256))
                label = labels[i]
                feat = np.asarray(img)
                if len(feat.shape) < 3: continue
                if not feat.shape[2] == 3: continue
                if cnt == 0:
                    _labels = [label]
                    data = [feat]
                else:
                    _labels = np.append(_labels,[label], axis=0)
                    data = np.append(data,[feat], axis=0)
                cnt += 1
    print(_labels.shape)
    print(data.shape)
    print("Total {} images have been saved".format(5000*f_cnt + cnt))

def split_test_train(folder):
    for i in  range(29):
        f_name = os.path.join(folder, "data_{}.npz".format(i))
        print("processing  the file  {}...".format(f_name))
        d = np.load(f_name)
        data = d["data"]
        label = d["label"]
        _X_train, _X_test, _Y_train, _Y_test = train_test_split(data, label, train_size=num_train, test_size=num_test, random_state=10, shuffle=True)
        X_train = np.zeros((num_train*29,256,256,3))
        X_test = np.zeros((num_test*29,256,256,3))
        Y_train = np.zeros((num_train*29,21))
        Y_test = np.zeros((num_test*29,21))
        start_train = 0
        start_test = 0
       
        X_train[start_train:start_train+num_train, :,:,:] = _X_train
        X_test[start_test:start_test+num_test, :,:,:] = _X_test
        Y_train[start_train:start_train+num_train, :,:,:] = _Y_train
        Y_test[start_test:start_test+num_test, :,:,:] = _Y_test
        start_train += num_train
        start_test += num_test


    np.savez_compressed("test_train.npz",\
                        features_training = X_train,\
                        label_training = Y_train,\
                        features_testing = X_test,\
                        label_testing = Y_test)

if __name__ == '__main__':

#save_map(image_list)
#download_relevent_pics(label_file, mapping_file, links_file)
#convert_to_array(label_file, mapping_file, folder)

split_test_train(os.path.join(folder, "./"))
