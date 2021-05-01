from __future__ import absolute_import, division, print_function, unicode_literals
from train import tbh_train
import time

#delay = 150 * 60
#print("waiting {} sec...".format(delay))
#time.sleep(delay)

#tbh_train.train_kfold('cifar10', 32, 512, 600)
#tbh_train.train('nus-wide', 32, 512, 500, True)
#tbh_train.train('ms-coco', 32, 512, 500, True)

# running for different lamda
#etas = [1, 5, 10, 15, 20]
#etas = [30, 40, 50]
#for eta in etas:
#    print("start training eta={}...".format(eta))
#    tbh_train.train_kfold('cifar10', 32, 512, 600, True, eta)

# running for NUS-WIDE
print("training for unsupervised version...")
tbh_train.train_kfold('nus-wide', 32, 512, 1500, False)
etas = [1, 5, 10, 15, 20]
for eta in etas:
    print("start training eta={}...".format(eta))
    tbh_train.train_kfold('nus-wide', 32, 512, 1500, True, eta)

