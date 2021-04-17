import numpy as np
import os, sys
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=sys.maxsize)

base = "/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/test/NUS-WIDE/"
tag_path = os.path.join(base, "Groundtruth/AllLabels") 
concept_file = os.path.join(base, "Concepts81.txt")
concept_file21 = os.path.join(base, "Concepts21.txt")
data_BoW = os.path.join(base, "Low_Level_Features", "BoW_int.dat")
label_file = os.path.join(base,"Labels21.npz")

total_images = 269648
num_labels = 21
_test_size = 100
_train_size = 500


def read_file(f_name, t=str, delim=' '):
	l = []
	with open(f_name,'r') as f:
		for line in f:
			if t==list:
				l.append(list(map(int, (line.strip('\n').strip(' ').split(delim)))) )
			else:
				l.append(t(line.strip('\n')))
	return l

def write_file(f_name, data):
	with open(f_name, 'w') as f:
		for d in data:
			f.write("{}\n".format(str(d)))

def save_top_concept(con_name, tag_path):
	out_f = os.path.join(base, "Concepts21.txt")
	out_labels = os.path.join(base, "Labels21.npz")
	concepts = read_file(con_name,str)
	
	concepts21 = []
	labels = np.zeros((total_images, num_labels))
	
	num = 1
	cnt = 0
	for c in concepts:
		print("{}. processing {}...".format(num,c))
		tag_file = os.path.join(tag_path, "Labels_{}.txt".format(c))
		arr = read_file(tag_file, int)
		print(arr.count(1))
		if arr.count(1) > 5000:
			concepts21.append(c)
			labels[:,cnt] = np.squeeze(np.array([arr]).T)
			cnt += 1
		num += 1
	write_file(out_f, concepts21)
	np.savez_compressed(out_labels, label = labels)

def generate_BoW_data(con_name, label_f, bow_file):
	out_f = os.path.join(base, "bow_data_full.npz")
	concepts = read_file(con_name,str)
	data = np.array(read_file(bow_file, list, ' '))
	label = np.load(label_f)["label"]
	_id = np.arange(total_images)
	used_ind = np.array([])

#	feat_train = np.zeros((_train_size*num_labels, 500))
#	label_train = np.zeros((_train_size*num_labels, num_labels))
	feat_test = np.zeros((_test_size*num_labels, 500))
	label_test = np.zeros((_test_size*num_labels, num_labels))

	num = 1
	start_train = 0
	start_test = 0
	for c in concepts:
		print("{}. processing {}...".format(num,c))
		tag_file = os.path.join(tag_path, "Labels_{}.txt".format(c))
		arr = np.array(read_file(tag_file, int))
		ind = np.where(arr==1)
		ind = np.setdiff1d(ind, used_ind)
# This is to split training and test data		
#		X_train, X_test, Z_train, Z_test = train_test_split(data[ind,:],_id[ind], test_size=_test_size,train_size=_train_size, random_state=10,shuffle=False)
		
#		feat_train[start_train:start_train+_train_size,:] = X_train
#		label_train[start_train:start_train+_train_size,:] = label[Z_train,:]
#		feat_test[start_test:start_test+_test_size,:] = X_test
#		label_test[start_test:start_test+_test_size,:] = label[Z_test,:]
#		start_train = start_train+_train_size
#		start_test = start_test+_test_size
		
# First get the test data: 100 images from each label, then remaining as the train data
		Z_train, Z_test = train_test_split(_id[ind], test_size=_test_size,train_size=_train_size, random_state=10,shuffle=False)
		feat_test[start_test:start_test+_test_size,:] = data[Z_test,:]
		label_test[start_test:start_test+_test_size,:] = label[Z_test,:]
		start_test = start_test+_test_size

		used_ind = np.hstack((used_ind, Z_test))
		num += 1
	
	relevent_id = np.sum(label, axis=1)
	remaining_id = np.setdiff1d(np.where(relevent_id>0), used_ind)
	feat_train = data[remaining_id,:]
	label_train = label[remaining_id,:]
	
	print("debug, the total size of used_ind is: {}".format(len(set(list(used_ind)))))
	print("Saving to the file {}".format(out_f))
	np.savez_compressed(out_f,\
						features_training = feat_train,\
						label_training = label_train,\
						features_testing = feat_test,\
						label_testing = label_test)
def disp_data(f_name):
	dataset = np.load(f_name)
	test_label = dataset["label_testing"]
	train_label = dataset["label_training"]
	sim_mat = np.matmul(test_label, train_label.T)
	sim_mat[sim_mat>0] = 1
	test_sim = np.sum(sim_mat, axis = 1)
	print(len(test_sim))
	print(np.sum(test_sim))
#	print(dataset["feat_training"])
#	print(dataset["label_training"])
#	print(dataset["feat_testing"])
#	print(dataset["label_testing"])


if __name__ == '__main__':
	save_top_concept(concept_file, tag_path)
#	generate_BoW_data(concept_file21, label_file, data_BoW)
#	disp_data(os.path.join(base, "bow_data.npz"))
		
