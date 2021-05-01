import sys, os
sys.path.append('..')
import matplotlib
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import numpy as np
import math
import seaborn as sns
from model.tbh import TBH
from util.data.dataset import Dataset
from util.eval_tools import eval_cls_map
from util.data.set_processor import SET_SIZE
from scipy.special import entr
np.set_printoptions(threshold=sys.maxsize)

cnames = ['aquamarine', 'bisque', 'black', 'blue', 'blueviolet', 'brown', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'gainsboro', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey',  'hotpink', 'indianred', 'indigo', 'khaki', 'lavender', 'lawngreen', 'lightblue', 'lightcoral', 'lightgreen', 'lightpink', 'lightsalmon', 'lightskyblue', 'lightsteelblue', 'lime', 'limegreen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']

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

def load_weights(set_name, bbn_dim, cbn_dim, batch_size, middle_dim, path, fold_num=0):
    """
    create model and load it with pre-trained weights stored under the path
	"""
    print("loading weights from {}".format(path))
    model = TBH(set_name, bbn_dim, cbn_dim, middle_dim)
    data = Dataset(set_name=set_name, batch_size=batch_size, shuffle=False, kfold=fold_num, code_length=bbn_dim)

    actor_opt = tf.keras.optimizers.Adam(1e-4)
    critic_opt = tf.keras.optimizers.Adam(1e-4)

    checkpoint = tf.train.Checkpoint( actor_opt=actor_opt, critic_opt=critic_opt, model=model )
    checkpoint.restore(tf.train.latest_checkpoint(path))
    update_codes(model,data,batch_size,set_name)

    # saving codes to pickle file
    with open(os.path.join(path,'output_codes'), 'wb') as f:
        pickle.dump(data.test_code, f)
        pickle.dump(data.test_label, f)
        pickle.dump(data.train_code, f)
        pickle.dump(data.train_label, f)
#    return 
    print("updating codes for the dataset and plot PR code...")
    test_hook,test_precision,pr_curve = eval_cls_map(data.test_code, data.train_code, data.test_label, data.train_label, 1000, True)
    make_PR_plot(path, pr_curve)
    print("The mPA is: {}".format(test_hook))
    print("The mean precision@1000 is: {}".format(test_precision))
    return test_hook, test_precision

def load_kfold(set_name, bbn_dim, cbn_dim, batch_size, middle_dim, path):
    kfold = 6
    hooks = []
    precs = []
    for i in range(1, kfold+1):
        model_path = os.path.join(path, "model_{}".format(i))
        hook, prec = load_weights(set_name, bbn_dim, cbn_dim, batch_size, middle_dim, model_path, i)
        hooks.append(hook)
        precs.append(prec)
    result_f = "results.txt"
    if os.path.isfile(os.path.join( path, result_f )): result_f = "results_recompute.txt"
    with open( os.path.join(path,result_f) ) as f:
        f.write("MAPS:")
        f.write(" ".join(hooks))
        f.write("precision@1000:")
        f.write(" ".join(precs))
    print("MAP: ")
    print(hooks)
    print("precision@1000: ")
    print(precs)

def get_dists(labels, codes, label_dim):
    print("total {} samples\n".format(codes.shape[0]))
    dists = np.zeros([label_dim, codes.shape[1]])
    _labels = labels.astype(int)
    for i in range(label_dim):
        ind = np.where(np.squeeze(_labels[:, i]) == 1)[0]
        dists[i,:] = np.sum(codes[ind, :], axis=0)
    return dists

def compute_entropy(y):
    total = np.sum(y)
    prob = y / total
    entropy = entr(prob).sum()
    return entropy

def make_plot_merge(codes, path, name):
    x = range(np.shape(codes)[1])
    y = np.sum(codes, axis=0)
    entropy = compute_entropy(y)
    print("entropy is: {}\n".format(entropy))

    plt.bar(x, y)
    plt.xlabel("Index of Codes")
    plt.ylabel("Number of 1's")
    plt.title("Bits Distribution for All Classes")
    plt.savefig(os.path.join(path, name+"_distributionMerged.png"))
    plt.show()
    plt.close()

def plot_hist(data, path, name, t):
    print("Mean is: {}, standard deviation is: {}\n".format(np.mean(data), np.std(data) ))
    f, ax = plt.subplots(figsize=(11, 9))
    ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.6)
    plt.hist(data, bins=40, range=[0,1], alpha=0.9, histtype='bar', ec='black')
    plt.xlabel("Abs(Correlation)")
    plt.ylabel("Counts")
    plt.savefig(os.path.join(path, name+"_corrHist"+t+".png"))
    plt.show()
    plt.close()

def plot_hist_binary(data, rng, path, name):
    f, ax = plt.subplots(figsize=(11, 9))
    ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.6)
    data = np.sum(data, axis=0)
    plt.bar(range(rng), data)
    plt.ylabel("Counts")
    plt.title(name)
    plt.savefig(os.path.join(path, name+".png"))
    plt.show()
    plt.close()

def plot_correlation(corr, path, name, t, showAnnot=True):
    mask = np.triu(np.ones_like(corr, dtype=bool))
    mask = np.logical_not(mask)
    # output the number of correlation > 0.5
    thred = 0.5
    num = 0
    rows = np.shape(corr)[0]
    cols = np.shape(corr)[1]
    abs_corr = []
    for i in range(rows):
        for j in range(cols):
            if mask[i][j]:
                abs_corr.append(abs(corr[i][j]))
                if abs(corr[i][j]) > thred: num += 1
    print("The number of large correlation: {}\n".format(num))
    plot_hist(abs_corr, path, name, t)

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(0,230,90,60,as_cmap=True)
    if showAnnot:
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=.5, cmap=cmap, vmin=-1,vmax=1,square=True)
    else:
        sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, vmin=-1,vmax=1,square=True)
    plt.title("Correlation Matrix "+t)
    plt.savefig(os.path.join(path, name+"_corr"+t+".png"))
    plt.show()
    plt.close()

def get_abs_corrs(corr):
    mask = np.triu(np.ones_like(corr, dtype=bool))
    mask = np.logical_not(mask)
    rows = np.shape(corr)[0]
    cols = np.shape(corr)[1]
    abs_corr = []
    for i in range(rows):
        for j in range(cols):
            if mask[i][j]:
                abs_corr.append(abs(corr[i][j]))
    return abs_corr

def make_plot_labels(dists, path, name):
    x = range(np.shape(dists)[1])
    label_dim = np.shape(dists)[0]
    for i in range(label_dim):
        y = dists[i]
        plt.plot(x,y, label="Class {}".format(i))
    plt.xlabel("Index of Codes")
    plt.ylabel("Number of 1's")
    plt.title("Bits Distribution for each Classes")
    plt.legend()
    plt.savefig(os.path.join(path, name+"_separatedAll.png"))
    plt.show()
    plt.close()

def make_subplot_labels(dists, path, name):
    x = range(np.shape(dists)[1])
    label_dim = np.shape(dists)[0]
    rows = int(math.sqrt(label_dim))
    cols = int(label_dim / rows)
    if cols * rows < label_dim: cols +=  1
    corr_labels = np.corrcoef(dists)
    if "ms-coco" in name:
        plot_correlation(corr_labels, path, name, "Labels", False)
    else:
        plot_correlation(corr_labels, path, name, "Labels")

    fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=(11, 9))
    i = 0
    for row in ax:
        for col in row:
            if i >= label_dim:
                col.clear()
                continue
            col.bar(x, dists[i], color=cnames[i])
            col.set_title("Class {}".format(i))
            i += 1
    plt.setp(ax[-1, :], xlabel='Index of Bits')
    plt.setp(ax[:, 0], ylabel='Number of 1s')
    plt.savefig(os.path.join(path, name+"_separatedLabels.png"))
    plt.show()
    plt.close()

def make_subplot_bits(dists, path, name):
    label_dim = np.shape(dists)[0]
    bbn_dim = np.shape(dists)[1]

    rows = int(math.sqrt(bbn_dim))
    cols = int(bbn_dim / rows)
    if cols * rows < bbn_dim: cols +=  1
    _dists = np.transpose(dists)
    corr_bits = np.corrcoef(_dists)
    plot_correlation(corr_bits, path, name, "Bits", False)

    x = range(label_dim)
    i = 0
    fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=(11, 9))
    for row in ax:
        for col in row:
            if i>= bbn_dim:
                col.clear()
                continue
            col.bar(x, _dists[i], color=cnames[i])
            col.set_title("{}: {:.2f}".format(i, compute_entropy(_dists[i])))
            i += 1
    plt.setp(ax[-1, :], xlabel='Class Number')
    plt.setp(ax[:, 0], ylabel='Number of 1s')
    plt.savefig(os.path.join(path, name+"_separatedBits.png"))
    plt.show()
    plt.close()

def load_codes(path):
    with open(os.path.join(path, 'output_codes'), "rb") as f:
        test_code = pickle.load(f)
        test_label = pickle.load(f)
        train_code = pickle.load(f)
        train_label = pickle.load(f)
    return test_code,test_label,train_code,train_label

def normalize_dists(dists):
    _dists = dists / np.sum(dists, axis=1, keepdims=True)

    return _dists * 10000

def plot_distribution(name, label_dim, path, normalize=False):
    """
    :param name: the dataset name
    :label_dim: number of labels
    :param path: the file path from which the codes and labels can be loaded: test_code, test_label, train_code, train_label
    
    can make plots of bits distribution, correlation matrix
    """
    test_code,test_label,train_code,train_label = load_codes(path)
    plot_hist_binary(train_label, label_dim, path, "Labels_Distribution")
    plot_hist_binary(train_code, 32, path, "Bits_Distribution")
    dists_test = get_dists(test_label, test_code, label_dim)
    dists_train = get_dists(train_label, train_code, label_dim)

    if normalize:
        dists_train = normalize_dists(dists_train)
        dists_test = normalize_dists(dists_test)
        name += "_Normalized"
    make_plot_merge(dists_train, path, name+"_train")
    make_subplot_labels(dists_train, path, name+"_train")
    make_subplot_bits(dists_train, path, name+"_train")

def compare_hist(data1, leg1, data2, leg2, path, name, t):
    f, ax = plt.subplots(figsize=(11, 9))
    ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.6)
    kws = dict(alpha= 0.7, linewidth = 2)
    bins = np.linspace(0, 1, 40)

    plt.hist([data1, data2], bins, label=[leg1, leg2])

#    plt.hist(data1, bins=bins, label=leg1, color="gold", edgecolor="crimson", **kws)
#    plt.hist(data2, bins=bins, label=leg2, color="lightseagreen", edgecolor="k", **kws)
    plt.legend()
    plt.xlabel("Abs(Correlation)")
    plt.ylabel("Counts")
    plt.savefig(os.path.join(path, name+"_Compare_corrHist"+t+".png"))
    plt.show()
    plt.close()

def compare_bits_distribution_merge(codes1, leg1, codes2, leg2, path, name):
    print(np.shape(codes1))
    x = np.arange((np.shape(codes1)[1]))
    y1 = np.sum(codes1, axis=0)
    entropy1 = compute_entropy(y1)
    y2 = np.sum(codes2, axis=0)
    entropy2 = compute_entropy(y2)
    print("entropy of {}_{} is: {}\n".format(name, leg1, entropy1))
    print("entropy of {}_{} is: {}\n".format(name, leg2, entropy2))

    kws = dict(alpha= 0.7, linewidth = 2)
    width  = 0.35
    fig, ax = plt.subplots()
    ax.bar(x, y1, color="gold", edgecolor="crimson", label=leg1, **kws)
    ax.bar(x, y2, color="lightseagreen", edgecolor="k", label=leg2, **kws)
    ax.set_ylabel("Number of 1's")
    plt.title("Bits Distribution for All Classes")
    plt.legend()
    plt.savefig(os.path.join(path, name+"_Compare_distributionMerged.png"))
    plt.show()
    plt.close()

def compare_subplot_labels(dists1, leg1, dists2, leg2, path, name):
    x = range(np.shape(dists1)[1])
    label_dim = np.shape(dists1)[0]
    rows = int(math.sqrt(label_dim))
    cols = int(label_dim / rows)
    if cols * rows < label_dim: cols +=  1

    corr_labels1 = np.corrcoef(dists1)
    corr_labels2 = np.corrcoef(dists2)
    compare_hist(get_abs_corrs(corr_labels1), leg1, get_abs_corrs(corr_labels2), leg2, path, name, "Labels")

    fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=(11, 9))
    kws = dict(alpha= 0.7, linewidth = 2)
    i = 0
    for row in ax:
        for col in row:
            if i >= label_dim:
                col.clear()
                continue
            l1 = col.bar(x, dists1[i], color="gold", edgecolor="crimson", label=leg1, **kws)
            l2 = col.bar(x, dists2[i], color="lightseagreen", edgecolor="k", label=leg2, **kws)
            col.set_title("Class {}".format(i))
            i += 1
    plt.setp(ax[-1, :], xlabel='Index of Bits')
    plt.setp(ax[:, 0], ylabel='Number of 1s')
    fig.legend((l1,l2), (leg1, leg2), 'upper center')
    plt.savefig(os.path.join(path, name+"_Compare_separatedLabels.png"))
    plt.show()
    plt.close()

def compare_subplot_bits(dists1, leg1, dists2, leg2, path, name):
    label_dim = np.shape(dists1)[0]
    bbn_dim = np.shape(dists1)[1]
    rows = int(math.sqrt(bbn_dim))
    cols = int(bbn_dim / rows)
    if cols * rows < bbn_dim: cols +=  1
    _dists1 = np.transpose(dists1)
    _dists2 = np.transpose(dists2)

    corr_labels1 = np.corrcoef(_dists1)
    corr_labels2 = np.corrcoef(_dists2)
    compare_hist(get_abs_corrs(corr_labels1), leg1, get_abs_corrs(corr_labels2), leg2, path, name, "Bits")

    x = range(label_dim)
    i = 0
    fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=(11, 9))
    kws = dict(alpha= 0.7, linewidth = 2)
    for row in ax:
        for col in row:
            if i>= bbn_dim:
                col.clear()
                continue
            l1 = col.bar(x, _dists1[i], color="gold", edgecolor="crimson", label=leg1, **kws)
            l2 = col.bar(x, _dists2[i], color="lightseagreen", edgecolor="k", label=leg2, **kws)
            col.set_title("Bit Index {}".format(i))
            i += 1
    plt.setp(ax[-1, :], xlabel='Class Number')
    plt.setp(ax[:, 0], ylabel='Number of 1s')
    fig.legend((l1,l2), (leg1, leg2), 'upper center')
    plt.savefig(os.path.join(path, name+"_Compare_separatedBits.png"))
    plt.show()
    plt.close()

def compare_distribution(name, label_dim, path1, leg1, path2, leg2, normalize=False):
    """
    make superposition plots to compare bits distribution for two different settings
    """
    test_code1,test_label1,train_code1,train_label1 = load_codes(path1)
    test_code2,test_label2,train_code2,train_label2 = load_codes(path2)
    dists_train1 = get_dists(train_label1, train_code1, label_dim)
    dists_train2 = get_dists(train_label2, train_code2, label_dim)

    if normalize:
        dists_train1 = normalize_dists(dists_train1)
        dists_train2 = normalize_dists(dists_train2)
        name += "_Normalized"
    compare_bits_distribution_merge(dists_train1, leg1, dists_train2, leg2, path2, name)
    compare_subplot_labels(dists_train1, leg1, dists_train2, leg2, path2, name)
    compare_subplot_bits(dists_train1, leg1, dists_train2, leg2, path2, name)

if __name__ == '__main__':
#    load_weights('ms-coco', 32, 512, 100, 1024, r"E:\Users\yuan\MasterThesis\TBH\result\ms-coco\model\Sat13Mar2021-112144") 
#    load_weights('cifar10', 32, 512, 100, 1024, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/cifar10/model/compare/Sun18Apr2021-125100_incv3_-criticloss_actor-logloss")
#    load_weights('cifar10', 32, 512, 100, 1024, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/cifar10/model/compare/Sun18Apr2021-181709_incv3_criticloss_actor-logloss_supervised_1")
#    load_weights('cifar10', 32, 512, 100, 1024, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/cifar10/model/compare/Sun18Apr2021-215535_incv3_-criticloss_actor-logloss_supervised_10")
#    load_weights('cifar10', 32, 512, 100, 1024, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/cifar10/model/compare/Mon19Apr2021-081941_incv3_-criticloss_actor-logloss_supervised_1_regression_10")
#  NUS-WIDE
#    load_weights('nus-wide', 32, 512, 100, 1024, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/nus-wide/model/compare/Sun18Apr2021-200514_unsupervised")
#    load_weights('nus-wide', 32, 512, 100, 1024, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/nus-wide/model/compare/Mon19Apr2021-101434_supervised_1")
#    load_weights('nus-wide', 32, 512, 100, 1024, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/nus-wide/model/compare/Mon19Apr2021-204850_supervised_1_regression_5") 
#MS-COCO
#    load_weights('ms-coco', 32, 512, 100, 1024, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/ms-coco/model/compare/Mon19Apr2021-124910_unsupervised") 
#    load_weights('ms-coco', 32, 512, 100, 1024, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/ms-coco/model/compare/Mon19Apr2021-142248_supervised_1_regression_1") 
#    load_weights('ms-coco', 32, 512, 100, 1024, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/ms-coco/model/compare/Tue20Apr2021-095547_supervised_1_regression_5") 

    load_kfold('cifar10', 32, 512, 500, 1024, \
            r"E:\Users\yuan\MasterThesis\TBH\result\cifar10\Tue27Apr2021-225947_kfold6"
            )
    load_kfold('cifar10', 32, 512, 500, 1024, \
            r"E:\Users\yuan\MasterThesis\TBH\result\cifar10\Wed28Apr2021-233318lr0.0001sup_eta1_lamda1"
            )
    load_kfold('cifar10', 32, 512, 500, 1024, \
            r"E:\Users\yuan\MasterThesis\TBH\result\cifar10\Thu29Apr2021-035933lr0.0001sup_eta5_lamda1"
            )
#    load_kfold('cifar10', 32, 512, 500, 1024, \
#            r"E:\Users\yuan\MasterThesis\TBH\result\cifar10\Thu29Apr2021-082613lr0.0001sup_eta10_lamda1"
#            )
#    load_kfold('cifar10', 32, 512, 500, 1024, \
#            r"E:\Users\yuan\MasterThesis\TBH\result\cifar10\Thu29Apr2021-125315lr0.0001sup_eta15_lamda1"
#            )
#    load_kfold('cifar10', 32, 512, 500, 1024, \
#            r"E:\Users\yuan\MasterThesis\TBH\result\cifar10\Thu29Apr2021-172030lr0.0001sup_eta20_lamda1"
#            )

#    plot_distribution('cifar10', 10, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/cifar10/model/compare/Sun18Apr2021-125100_incv3_-criticloss_actor-logloss")
#    plot_distribution('cifar10', 10, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/cifar10/model/compare/Sun18Apr2021-181709_incv3_criticloss_actor-logloss_supervised_1") 
#    plot_distribution('cifar10', 10, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/cifar10/model/compare/Sun18Apr2021-215535_incv3_-criticloss_actor-logloss_supervised_10")
#    plot_distribution('cifar10', 10, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/cifar10/model/compare/Mon19Apr2021-081941_incv3_-criticloss_actor-logloss_supervised_1_regression_10")
#  NUS-WIDE
#    plot_distribution('nus-wide', 21, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/nus-wide/model/compare/Sun18Apr2021-200514_unsupervised", True)
#    plot_distribution('nus-wide', 21, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/nus-wide/model/compare/Mon19Apr2021-101434_supervised_1_regression_1", True)
#    plot_distribution('nus-wide', 21, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/nus-wide/model/compare/Mon19Apr2021-204850_supervised_1_regression_5", True) 
# MS-COCO
#    plot_distribution('ms-coco', 80, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/ms-coco/model/compare/Mon19Apr2021-124910_unsupervised", True)
#    plot_distribution('ms-coco', 80, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/ms-coco/model/compare/Mon19Apr2021-142248_supervised_1_regression_1", True) 
#    plot_distribution('ms-coco', 80, r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/ms-coco/model/compare/Tue20Apr2021-095547_supervised_1_regression_5", True) 

# CIFAR-10
#    compare_distribution('cifar10', 10,\
#        r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/cifar10/model/compare/Sun18Apr2021-125100_incv3_-criticloss_actor-logloss",\
#        "Unsupervised",\
#        r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/cifar10/model/compare/Mon19Apr2021-081941_incv3_-criticloss_actor-logloss_supervised_1_regression_10",\
#        "Supervised $\eta=1$ $\gamma=10$")
# NUS-WIDE
#    compare_distribution('nus-wide', 21,\
#        "/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/nus-wide/model/compare/Sun18Apr2021-200514_unsupervised",\
#        "Unsupervised",\
#        "/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/nus-wide/model/compare/Mon19Apr2021-101434_supervised_1",\
#        "Supervised $\eta=1$ $\gamma=1$", True)
#    compare_distribution('nus-wide', 21,\
#        "/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/nus-wide/model/compare/Sun18Apr2021-200514_unsupervised",\
#        "Unsupervised",\
#        r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/nus-wide/model/compare/Mon19Apr2021-204850_supervised_1_regression_5",\
#        "Supervised $\eta=1$ $\gamma=5$", True)
# MS-COCO
#    compare_distribution('nus-wide', 80,\
#        r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/ms-coco/model/compare/Mon19Apr2021-124910_unsupervised",\
#        "Unsupervised",\
#        r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/ms-coco/model/compare/Mon19Apr2021-142248_supervised_1_regression_1",\
#        "Supervised $\eta=1$ $\gamma=1$", True)
#    compare_distribution('nus-wide', 80,\
#        r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/ms-coco/model/compare/Mon19Apr2021-124910_unsupervised",\
#        "Unsupervised",\
#        r"/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/results/ms-coco/model/compare/Tue20Apr2021-095547_supervised_1_regression_5",\
#        "Supervised $\eta=1$ $\gamma=5$", True)

