from __future__ import absolute_import, division, print_function, unicode_literals
from train import tbh_train
import time
from time import gmtime, strftime
time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())

#delay = 150 * 60
#print("waiting {} sec...".format(delay))
#time.sleep(delay)

#tbh_train.train_kfold('cifar10', 32, 512, 600)
#tbh_train.train('nus-wide', 32, 512, 500, True)
#tbh_train.train('ms-coco', 32, 512, 500, True)

# running for different lamda
#print("run unsupervised version")
#tbh_train.train_kfold('cifar10', 32, 512, 1500, False)

#etas = [120]
#tbh_train.train('cifar10', 32, 512, 1500, time_string, 6, True, 20)
#etas = [40]
#for eta in etas:
#    print("start training eta={}...".format(eta))
#    tbh_train.train_kfold('cifar10', 32, 512, 1500, True, 20, eta)

# running for NUS-WIDE
#print("training for unsupervised version...")
#tbh_train.train_kfold('nus-wide', 32, 512, 1500, False, 1, 1, 1, 1024, 40000)
#etas = [1, 5, 10, 15, 20]
tbh_train.train('nus-wide', 32, 512, 1500, time_string, 5, True, 10,20,1,1024,40000)
tbh_train.train('nus-wide', 32, 512, 1500, time_string, 6, True, 10,20,1,1024,40000)
etas = [30, 40]
for eta in etas:
    print("start training eta={}...".format(eta))
    tbh_train.train_kfold('nus-wide', 32, 512, 1500, True, 10, eta, 1, 1024, 40000)

# running for different lamda
#print("run unsupervised version")
#tbh_train.train_kfold('ms-coco', 32, 512, 1500, False)

#etas = [30]
#for eta in etas:
#    print("start training eta={}...".format(eta))
#    tbh_train.train_kfold('ms-coco', 32, 512, 1500, True, 30, eta, 1, 1024, 60000)

