from __future__ import absolute_import, division, print_function, unicode_literals
from train import tbh_train

#tbh_train.train('cifar10', 32, 512, 500, True)
#tbh_train.train('nus-wide', 32, 512, 500, True)
tbh_train.train('ms-coco', 32, 512, 500, True)
