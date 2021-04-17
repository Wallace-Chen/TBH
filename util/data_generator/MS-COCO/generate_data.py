from pycocotools.coco import COCO
import numpy as np
import os,sys
import re
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

valFile = 'annotations/instances_val2014.json'
trainFile = 'annotations/instances_train2014.json'

# num per file
num_test = 200
num_train = 2000

coco = COCO(trainFile)

index = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 13:11, 14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19, 22:20, 23:21, 24:22, 25:23, 27:24, 28:25, 31:26, 32:27, 33:28, 34:29, 35:30, 36:31, 37:32, 38:33, 39:34, 40:35, 41:36, 42:37, 43:38, 44:39, 46:40, 47:41, 48:42, 49:43, 50:44, 51:45, 52:46, 53:47, 54:48, 55:49, 56:50, 57:51, 58:52, 59:53, 60:54, 61:55, 62:56, 63:57, 64:58, 65:59, 67:60, 70:61, 72:62, 73:63, 74:64, 75:65, 76:66, 77:67, 78:68, 79:69, 80:70, 81:71, 82:72, 84:73, 85:74, 86:75, 87:76, 88:77, 89:78, 90:79}

folder = "/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/cocoapi/"

def get_catIds(img_id):
	ids = []
	names = []
	annIds = coco.getAnnIds(img_id)
	anns = coco.loadAnns(annIds)
	for ann in anns:
		if ann['category_id'] not in ids:
			ids.append(ann['category_id'])
			#names.append( coco.loadCats(ann['category_id'])[0]['name'] )
	return ids

def get_label(ids):
	labels = np.zeros(80)
	for i in ids: labels[index[i]] = 1
	return labels

def convert_to_array(folder):
	val_list = [os.path.join(folder, "val2014/val2014",f) for f in os.listdir(os.path.join(folder, "val2014/val2014")) if f.endswith('jpg')]
	train_list = [os.path.join(folder, "train2014/train2014",f) for f in os.listdir(os.path.join(folder, "train2014/train2014")) if f.endswith('jpg')]
	#f_list = val_list + train_list
	f_list = train_list
	num = 0
	cnt  =  0
	f_cnt = 0
	print("There are total {} images to be processed".format(len(f_list)))
	for f in f_list:
		cnt += 1
		if (cnt+1)%1000 == 0: print("{} finished.".format(cnt+1))
		if (cnt+1)%5000 == 0: # save and restart
			np.savez_compressed("data_{}.npz".format(f_cnt),\
								data = data,\
								label = labels)
			f_cnt += 1
			num = 0
		x = int(re.search(r"_(\d+).jpg", f).group(1))
		ids = get_catIds(x)
		if len(ids)>0:
			label = get_label(ids)
			img = Image.open(f).resize((256,256))
			feat = np.asarray(img)
			if len(feat.shape)<3: continue
			if not feat.shape[2] == 3: continue
			if num == 0:
				labels = [label]
				data = [feat]
			else:
				labels = np.append(labels,[label], axis=0)
				data = np.append(data,[feat], axis=0)
			num += 1
#		else:
#			print("Error, ids are not found: {}, pid:{}, for the file: {}".format(ids,x,f))
	print(labels.shape)
	print(data.shape)
	print("Total {} images have been saved".format(5000*f_cnt + num))

def split_test_train(folder):
	X_train = np.zeros((48000,256,256,3))
	X_test = np.zeros((4800,256,256,3))
	Y_train = np.zeros((48000,80))
	Y_test = np.zeros((4800,80))
	start_train = 0
	start_test = 0
	for i in range(24):
		print("processing {} file...".format(i))
		f_name = os.path.join(folder, "data_{}.npz".format(i))
		d = np.load(f_name)
		data = d["data"]
		label = d["label"]
		_X_train, _X_test, _Y_train, _Y_test = train_test_split(data, label, test_size=num_test,train_size=num_train,  random_state=10, shuffle=True)
		
		X_train[start_train:start_train+num_train,:,:,:] = _X_train
		X_test[start_test:start_test+num_test,:,:,:] = _X_test
		Y_train[start_train:start_train+num_train,:] = _Y_train
		Y_test[start_test:start_test+num_test,:] = _Y_test

		start_train = start_train+num_train
		start_test = start_test+num_test


	print("writing to the file...")
	np.savez_compressed("test_train.npz",\
						features_training = X_train,\
						label_training = Y_train,\
						features_testing = X_test,\
						label_testing = Y_test)
	
if __name__ == '__main__':
	#convert_to_array(folder)
	split_test_train(folder)
