import sys, os
sys.path.append('..')
import matplotlib
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import numpy as np
import math
import seaborn as sns
import scipy.stats as st
from model.tbh import TBH
from util.data.dataset import Dataset
from util.eval_tools import eval_cls_map
from util.data.set_processor import SET_SIZE
from scipy.special import entr
from tensorflow.python.training import py_checkpoint_reader
from matplotlib.colors import LogNorm
from matplotlib.colors import DivergingNorm
np.set_printoptions(threshold=sys.maxsize)

cnames = ['aquamarine', 'bisque', 'black', 'blue', 'blueviolet', 'brown', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'gainsboro', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey',  'hotpink', 'indianred', 'indigo', 'khaki', 'lavender', 'lawngreen', 'lightblue', 'lightcoral', 'lightgreen', 'lightpink', 'lightsalmon', 'lightskyblue', 'lightsteelblue', 'lime', 'limegreen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']


maps_cifar10_BS1500 = {
    'unsupervised':  [ 0.6277661067355135, 0.6271457011178395, 0.6472887822789963, 0.6418900057852549, 0.6458793156052561, 0.6283929713746502 ],
    'supervised_eta1': [ 0.6394864384308668, 0.6451760555346215, 0.6607460592583302, 0.6426766962445823, 0.6566410490787133, 0.6529254902261147 ],
    'supervised_eta10': [ 0.7477564027827889, 0.7377129320320592, 0.7485809037575453, 0.7563957338468829, 0.7526493015183902, 0.7552535012862501 ],
    'supervised_eta20': [ 0.8226960791997466, 0.82360803823062, 0.8261520598590774, 0.8253284169965182, 0.8262724142687446, 0.8191813502842782 ],
    'supervised_eta30': [ 0.8505551395100475, 0.8419827656983644, 0.8544466993014631, 0.8513076115402212, 0.8558323728330582, 0.8515907154440957 ],
    'supervised_eta40': [0.8662518938939066, 0.8643738709188845, 0.861925253095899, 0.8663076701624739, 0.86853985549839, 0.8653899638643929 ], 
    'supervised_eta50': [ 0.8737202685678068, 0.8693116314175888, 0.8698518285499547, 0.8750182211346161, 0.8762095827636818, 0.8729805790139403 ]
}

precs_cifar10_BS1500 = {
    'unsupervised': [ 0.5822624262426237, 0.5783580506354458, 0.6027931345076041, 0.5957169150745222, 0.5990659131826381, 0.5838067420226073 ],
    'supervised_eta1': [ 0.5891980396079205, 0.5946132292604869, 0.6143072843706258, 0.5918891667500241, 0.6084965475833118, 0.6041701510453154 ],
    'supervised_eta10': [ 0.7090937187437502, 0.6958044022010998, 0.7093878163449037, 0.718840336134452, 0.7115358607582238, 0.7140678067806802 ],
    'supervised_eta20': [ 0.7956059241469047, 0.8013949739687596, 0.7888717434869749, 0.7991197838486932, 0.7990296118447314, 0.7944439663798277 ],
    'supervised_eta30': [ 0.8314781171757563, 0.8229422961330357, 0.8359517601042927, 0.8311467200801149, 0.8377023130069068, 0.8338843264897295 ],
    'supervised_eta40': [ 0.8518844341986432, 0.8509180541624763, 0.8481952247190925, 0.8520034092048453, 0.853471785105737, 0.8545144230769147 ],
    'supervised_eta50': [ 0.8613893627696833, 0.8577796218064618, 0.8583703368526783, 0.8646941484013569, 0.8637492977527991, 0.8617083375137836 ]
}

maps_cifar10_BS1500_gamma20 = {
    'unsupervised': [ 0.6277661067355135, 0.6271457011178395, 0.6472887822789963, 0.6418900057852549, 0.6458793156052561, 0.6283929713746502 ],
    'eta20':  [ 0.8226960791997466, 0.82360803823062, 0.8261520598590774, 0.8253284169965182, 0.8262724142687446, 0.8191813502842782 ],
    'eta90':  [ 0.8013827877437719, 0.8048757338455343, 0.8068136842053956, 0.8094337356720024, 0.8095372210779197, 0.8026969021336929 ],
    'eta100': [ 0.8029631231811433, 0.8006389295746249, 0.7991474612866267, 0.7997298211552073, 0.8058587789651633, 0.8013346686759037 ],
    'eta110': [ 0.7933831000694159, 0.7975814821392545, 0.7996953465894179, 0.7997599556053478, 0.8027548765713867, 0.7886168973734655 ],
    'eta120': [ 0.7922113992408223, 0.7945934167258767, 0.7935748673032129, 0.7990133809202954, 0.7979686261822427, 0.7868147265096254 ]
}

precs_cifar10_BS1500_gamma20 = {
    'unsupervised': [ 0.5822624262426237, 0.5783580506354458, 0.6027931345076041, 0.5957169150745222, 0.5990659131826381, 0.5838067420226073 ],
    'eta20':  [ 0.7956059241469047, 0.8013949739687596, 0.7888717434869749, 0.7991197838486932, 0.7990296118447314, 0.7944439663798277 ],
    'eta90':  [ 0.7714695347673807, 0.7798020624749664, 0.7801047733413387, 0.7821164698819263, 0.7829071535767895, 0.7774490143100186 ],
    'eta100': [ 0.7746176470588195, 0.7746765736015195, 0.769406925540431, 0.7709523714228482, 0.7770543380366243, 0.7713445411788243],
    'eta110': [ 0.7647966983491757, 0.7704377940146058, 0.7710572457966364, 0.7712921753051813, 0.7733537122273336, 0.7552306461292252 ],
    'eta120': [ 0.7611492447734297, 0.765730838503102, 0.7629328395556033, 0.7704147659063627, 0.7677636581949157, 0.755768461076649 ]
}

maps_nuswide_BS1500 = {
    'unsupervised': [  0.7063323179200155, 0.7000608494344213, 0.7070123766035628, 0.7055682316347244, 0.6980871885070713, 0.7014837689031542],
    'supervised_eta1': [ 0.737908966732637, 0.7381900901956747, 0.7320664892996025, 0.7345372615324542, 0.7277427228270046, 0.7279423462855152 ],
    'supervised_eta10': [ 0.811109963341777, 0.8116952491782364, 0.8105253846608722, 0.807102871131445, 0.8138969478548225, 0.8112020607078764 ],
    'supervised_eta20': [ 0.8293083898943491, 0.8247551636371944, 0.8227531607783234, 0.8248780987626139, 0.8294454996003942, 0.8316333353573591 ],
    'supervised_eta30': [ 0.8432612374868703, 0.8430637107336743, 0.8436403144765288, 0.8395788150460566, 0.8382090099836371, 0.8468114067483821 ],
    'supervised_eta40': [ 0.8589322698749956, 0.8541767968880377, 0.8503853168248917, 0.8534113434291188, 0.8474189491264448, 0.8524762778081302 ],
    'supervised_eta50': [ 0.8558448169564858, 0.861058728373101, 0.8616867947387894, 0.858145743888444, 0.8588294252618709, 0.8578766962639454]
}

precs_nuswide_BS1500 = {
    'unsupervised': [ 0.6832539269634799, 0.6780783156631357, 0.6846385277055448, 0.6808617861786169, 0.6749740896358543, 0.6771921384276851 ],
    'supervised_eta1': [ 0.7105815907953986, 0.7116429928978679, 0.7044319431943246, 0.7083012903871131, 0.7009736842105224, 0.7006730999999956 ],
    'supervised_eta10': [ 0.7895844337735047, 0.7905519551955178, 0.7898213464039182, 0.7853730492196904, 0.7908660000000014, 0.7899955999999979 ],
    'supervised_eta20': [ 0.8101532306461248, 0.8056603999999975, 0.8044641999999936, 0.8059105642256852, 0.8104757378689292, 0.8134976999999979 ],
    'supervised_eta30': [ 0.825939275710273, 0.8263739999999976, 0.8290443088617591, 0.8231737868934365, 0.8221081432572938, 0.8314701999999885 ],
    'supervised_eta40': [ 0.8436927463731727, 0.8387973986993432, 0.8361185237047261, 0.8386622297838111, 0.8320149119295337, 0.8379427999999892 ],
    'supervised_eta50': [ 0.8399070442265218, 0.8471065106510556, 0.8481873936968389, 0.8439625738016403, 0.844838503101851, 0.8442072414482753 ]
}

maps_nuswide_BS1500_gamma10 = {
    'unsupervised': [ 0.7063323179200155, 0.7000608494344213, 0.7070123766035628, 0.7055682316347244, 0.6980871885070713, 0.7014837689031542 ],
    'eta10': [ 0.8094462336330654, 0.8073204187251494, 0.8074272225752549, 0.8089518711132276, 0.8094116751176826, 0.806759721202822 ],
    'eta20': [ 0.8044904406743858, 0.8045155175868909, 0.8029444269248127, 0.8021596882086786, 0.8025760652694733, 0.8014667478235789 ],
    'eta30': [ 0.8013490523379548, 0.7931639096052343, 0.7967859369849933, 0.8016095746326203, 0.798133509124444, 0.8007228495108448 ],
    'eta40': [ 0.800944003432728, 0.7930216753211566, 0.7944692591533865, 0.8013980499204935, 0.7958898571113782, 0.7974828159045596 ]
}

precs_nuswide_BS1500_gamma10 = {
    'unsupervised': [ 0.6832539269634799, 0.6780783156631357, 0.6846385277055448, 0.6808617861786169, 0.6749740896358543, 0.6771921384276851 ],
    'eta10': [ 0.7859268780634172, 0.7839935987197444, 0.7843663098929681, 0.7866595978793636, 0.7874338301490472, 0.7832372474494895 ],
    'eta20': [ 0.7804047619047566, 0.7810437043704369, 0.7799295788736543, 0.7791670835417716, 0.777662932586517, 0.7770524052405278 ],
    'eta30': [ 0.7770261026102588, 0.7679144914491424, 0.7731452999999946, 0.7786292887866356, 0.7720891267380225, 0.7775999199839915 ],
    'eta40': [ 0.7773757503001195, 0.7689423884776861, 0.7689377875575067, 0.7765223567070129, 0.7706214621462115, 0.7741389999999984 ]
}

maps_mscoco_BS1500_c80 = {
    'unsupervised': [ 0.7536605111234497, 0.7562965906103257, 0.7556608210182059, 0.7552864281048478, 0.7405255692995414, 0.7517568659182756 ], 
    'supervised_eta1': [ 0.7649147121700528, 0.765186519238365, 0.7703041707849986, 0.7632040489400514, 0.7637913030723109, 0.7645416696278616 ], 
    'supervised_eta10': [ 0.5252597207597902, 0.5367841350282158, 0.4934148436469891, 0.6561299533275093, 0.5136148214866398, 0.5133307405999246 ], 
    'supervised_eta20': [ 0.5408639622925211, 0.5449171633364834, 0.4555115725125462, 0.5669273360396674, 0.5511078103701064, 0.5403213468280541 ], 
    'supervised_eta30': [ 0.5588504529419663, 0.5871092056933572, 0.5555603405642848, 0.583234450760479, 0.5760755073116901, 0.5431553308773301 ], 
    'supervised_eta40': [ 0.5463380476310008, 0.5698354204308681, 0.5561962708747958, 0.5896644919421016, 0.5705679128065687, 0.5637194026625971 ], 
    'supervised_eta50': [ 0.5582578786037852, 0.567859966515328, 0.5501289171262778, 0.6119703080914466, 0.5910287424888734, 0.5569749133071226 ] 
}

precs_mscoco_BS1500_c80 = {
    'unsupervised': [ 0.6967917375212519, 0.702352440976384, 0.6999851940776294, 0.7020766153230573, 0.6858932359415593, 0.7020496546200821 ], 
    'supervised_eta1': [ 0.7086183855156545, 0.7134284285285608, 0.7126273136568281, 0.7149138827765533, 0.7076528569998978, 0.7078003401020267 ], 
    'supervised_eta10': [ 0.516832933173274, 0.5302351470294012, 0.49059325932592557, 0.6297788894447178, 0.5092624262426273, 0.5099143999999972 ], 
    'supervised_eta20': [ 0.5360344103230731, 0.5405246524652272, 0.45458353341336044, 0.5622084833933579, 0.5466593978193313, 0.5360514102820536 ], 
    'supervised_eta30': [ 0.5548180272108671, 0.5762406925540335, 0.5505776000000128, 0.5796738021406532, 0.5709679871948807, 0.5394401080756404 ], 
    'supervised_eta40': [ 0.5415459459459222, 0.5668725235141238, 0.5537385477095051, 0.587008612056846, 0.5659279855971096, 0.5597678678678623 ], 
    'supervised_eta50': [ 0.5551327929550405, 0.563450745074496, 0.5483136076899613, 0.6089815926370731, 0.5873689475790022, 0.55404712356176 ] 
}

maps_mscoco_BS1500_c79 = {
    'unsupervised': [ 0.6764159646337556, 0.6593732032287909, 0.678911618522232, 0.659456242971663, 0.6733878978722142, 0.6721012106011272 ], 
    'supervised_eta1': [ 0.6875972709795226, 0.68466918237394, 0.6699411023574655, 0.6722474213450685, 0.6868141642368993, 0.6796729416439399 ], 
    'supervised_eta10': [ 0.6977231936985421, 0.7247948763061574, 0.7076077884889421, 0.6946361702320052, 0.7036145205087975, 0.6954010076999269 ], 
    'supervised_eta20': [ 0.6989854567358187, 0.7096775468800227, 0.755193490865632, 0.7266385156716129, 0.7056128735254125, 0.7675598100879819 ], 
    'supervised_eta30': [ 0.7783170280559197, 0.7563810528412067, 0.7782229172073143, 0.7609084662247791, 0.7709543021372075, 0.7711201262921605 ], 
    'supervised_eta40': [ 0.7411009665534924, 0.7630241795028396, 0.7651561911639547, 0.767649091920817, 0.7761050110611702, 0.7767873435184285 ], 
    'supervised_eta50': [ 0.7586224122164477, 0.7514797361114121, 0.7653794321672976, 0.7659555391385325, 0.7618168655650954, 0.7798701063769932 ] 
}

precs_mscoco_BS1500_c79 = {
    'unsupervised': [ 0.5779619904976222, 0.5568501563477155, 0.5791907442151355, 0.5572862681340643, 0.5761592842842825, 0.5744919939954964 ], 
    'supervised_eta1': [ 0.5832803551331754, 0.5828505632040065, 0.5667627563781924, 0.5724792396198112, 0.5897824890556596, 0.5852839089089064 ], 
    'supervised_eta10': [ 0.6202533466783433, 0.651598473473473, 0.6324873045653538, 0.6206200425372211, 0.6315257099962415, 0.6223000125109492 ], 
    'supervised_eta20': [ 0.6264080790395203, 0.6364290181363342, 0.6735486493246579, 0.6531534709193189, 0.6355925972239537, 0.6921335667833916 ], 
    'supervised_eta30': [ 0.7019114668000492, 0.687250093879081, 0.7014740527697807, 0.6901590539356736, 0.6986098049024514, 0.6988629879879824 ], 
    'supervised_eta40': [ 0.6697275797373305, 0.6971418910457041, 0.6953451338503824, 0.6932702702702662, 0.71031569731082, 0.7047865281081684 ], 
    'supervised_eta50': [ 0.68964360770578, 0.6815562930494615, 0.695788670751528, 0.697143857893415, 0.696420387742335, 0.7092538153615192 ] 
}

maps_mscoco_BS1500_c79_gamma30 = {
    'unsupervised': [ 0.6764159646337556, 0.6593732032287909, 0.678911618522232, 0.659456242971663, 0.6733878978722142, 0.6721012106011272 ],
    'eta30': [ 0.7856361433427086, 0.7680418066730739, 0.7781136890143693, 0.7659881962215612, 0.7701362279512184, 0.7847869854397967 ],
    'eta50': [ 0.7330492456318213, 0.7215707189364247, 0.7401437543155499, 0.7319396367778717, 0.7499468874420582, 0.7388286912072579 ],
    'eta60': [ 0.7296797714130809, 0.7128143302879942, 0.7391370346129791, 0.7312848812818059, 0.7224185123244142, 0.7400033073681869 ]
}

precs_mscoco_BS1500_c79_gamma30 = {
    'unsupervised': [ 0.5779619904976222, 0.5568501563477155, 0.5791907442151355, 0.5572862681340643, 0.5761592842842825, 0.5744919939954964 ],
    'eta30': [ 0.6890469160515466, 0.6852626705043107, 0.6891063563563545, 0.6838879299562196, 0.6828753597797478, 0.7059768576432269 ],
    'eta50': [ 0.6473293308317687, 0.6375206456456461, 0.6543339589589591, 0.6513319574734177, 0.6631213865598823, 0.6468754693366677 ],
    'eta60': [ 0.6330292536567029, 0.6226712980347952, 0.6483298311444647, 0.6394045295295291, 0.6265285285285289, 0.6540122668669432 ]
}

maps =  maps_nuswide_BS1500_gamma10 
precs = precs_nuswide_BS1500_gamma10 

def print_weights(set_name, bbn_dim, cbn_dim, batch_size, middle_dim, path, fold_num=0):
    print("loading weights from {}".format(path))
    model = TBH(set_name, bbn_dim, cbn_dim, middle_dim)
    data = Dataset(set_name=set_name, batch_size=batch_size, shuffle=False, kfold=fold_num, code_length=bbn_dim)

    actor_opt = tf.keras.optimizers.Adam(1e-4)
    critic_opt = tf.keras.optimizers.Adam(1e-4)

    checkpoint = tf.train.Checkpoint( actor_opt=actor_opt, critic_opt=critic_opt, model=model )
    checkpoint.restore(tf.train.latest_checkpoint(path))

    train_iter = iter(data.train_data)
    test_iter = iter(data.test_data)
    random_binary = (tf.sign(tf.random.uniform([batch_size, 32]) - 0.5) + 1) / 2
    random_cont = tf.random.uniform([batch_size, 512])

    test_batch = next(test_iter)
    model_input = [test_batch, random_binary, random_cont]
    model(model_input, training=True)
    vals=model.variables
    
    print(len(vals))
    for v in vals:
        print(v.name)

    kernel = vals[16].numpy()
    bias = np.atleast_2d(vals[17].numpy())
#    print(kernel)
#    print(bias)
    weights = np.r_[kernel, bias]
#    weights = kernel

#    print("The regularization term of classifer layer is: {}".format( np.sum(weights*weights) ))
    print("The regularization term of classifer layer is: {}".format( np.sum(np.abs(kernel)) ))

#    f, ax = plt.subplots(figsize=(5, 11))
    f, ax = plt.subplots(figsize=(11,5))
    cmap = sns.diverging_palette(0,230,90,60,as_cmap=True)
    divnorm = DivergingNorm(vmin=np.min(weights), vcenter=0, vmax=np.max(weights))
    #sns.heatmap(weights, linewidths=.5, annot=True, cmap=cmap)
    sns.heatmap(weights, linewidths=.5, cmap=cmap, norm=divnorm)
    plt.title("Weights of Classifer Layer")
    plt.savefig(os.path.join(path, "Weights_Classifier"))
    plt.show()
    plt.close()

    return

def compute_p_value(l1, l2, sig=0.01):
    n1 = len(l1)
    n2 = len(l2)
    mean_1 = st.tmean(l1)
    mean_2 = st.tmean(l2)
    var_1  = st.variation(l1)
    var_2  = st.variation(l2)
    std_1 = st.tstd(l1)
    std_2 = st.tstd(l2)

    mean_diff = mean_2  - mean_1
    df_num = pow( var_1/n1 + var_2/n2,2)
    df_den = pow(var_1/n1,2)/n1 + pow(var_2/n2,2)/n2
    df  = df_num/df_den
    # compute confidenece interval at the level of significance
    interval = st.t.ppf(1-sig/2, df)
    interval *= np.sqrt(var_1/n1+var_2/n2)

    # compute p value, H0: x2<=x1, H1:x2>x1
    test_den = np.sqrt(var_1/n1+var_2/n2)
    test = mean_diff/test_den
    p_value = 1 - st.t.cdf(test,df)

    print("l1 mean: {}, std: {}".format(mean_1, std_1))
    print("l2 mean: {}, std: {}".format(mean_2, std_2))
    return [mean_diff-interval, mean_diff+interval], p_value

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

def compare_PR_plot(path, data1, data2, name1, name2):
    """
    plot and compare two PR curves
    """
    plt.step(list(data1[1,:]), list(data1[0,:]), where='post', label=name1)
    plt.step(list(data2[1,:]), list(data2[0,:]), where='post', label=name2)

    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.legend()
    plt.title("P-R Curve")
    plt.savefig(os.path.join(path, 'P-R_{}_{}.png'.format(name1, name2)))

def load_PR_plot(path1, path2, name1, name2):
    data1 = []
    data2 = []
    with open(os.path.join(path1, 'PRLists.data'), "rb") as f:
        data1.append(pickle.load(f))
        data1.append(pickle.load(f))
    with open(os.path.join(path2, 'PRLists.data'), "rb") as f:
        data2.append(pickle.load(f))
        data2.append(pickle.load(f))
    data1 = np.array(data1)
    data2 = np.array(data2)
    compare_PR_plot(path2, data1, data2, name1, name2)

def compare_PR_plots(path, data, names, dataset):
    for i in range(len(data)):
        plt.step(list(data[i][1,:]), list(data[i][0,:]), where='post', label=names[i])

    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.legend()
    plt.title("P-R Curve on {}".format(dataset))
    plt.savefig(os.path.join(path, 'P-R_Comparisons.png'))

def load_PR_plots(paths, names, dataset):
    data = []
    for p in paths: 
        d = []
        with open(os.path.join(p, 'PRLists.data'), "rb") as f:
            d.append(pickle.load(f))
            d.append(pickle.load(f))
        d = np.array(d)
        data.append(d)
    compare_PR_plots(paths[0], data, names, dataset)


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
    with open( os.path.join(path,result_f), "w" ) as f:
        f.write("MAPS:")
        hooks = [str(i) for i in hooks]
        f.write(" ".join(hooks))
        f.write("precision@1000:")
        precs = [str(i) for i in precs]
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
    print("Mean is: {}, median is : {},  standard deviation is: {}\n".format(np.mean(data), np.median(data), np.std(data) ))
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

def plot_hist_binary(data, rng, path, name, compute_KL=False):
    f, ax = plt.subplots(figsize=(11, 9))
    ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.6)
    data = np.sum(data, axis=0)

    if compute_KL:
        sample = [1/rng for i in range(rng)]
        kl_d = st.entropy(list(data), sample)
        print("Kullback-Leibler divergence: {}".format(kl_d))
        print("ENtropy is: {}".format(compute_entropy(data)))
        
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
    print(type(train_label))
    print(np.shape(train_label))
    plot_hist_binary(train_label, label_dim, path, "Labels_Distribution")
    plot_hist_binary(train_code, 32, path, "Bits_Distribution", True)
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
    unsup = 'unsupervised'
    sups = ['supervised_eta1', 'supervised_eta10', 'supervised_eta20', 'supervised_eta30', 'supervised_eta40', 'supervised_eta50']

    unsup = 'unsupervised'
    sups = ['eta10', 'eta20', 'eta30', 'eta40']
#    for sup in sups:
#        print("\n-----l1: {}, l2:{}".format(unsup, sup))
#        print("  MAP")
#        r1, r2 = compute_p_value(maps[unsup], maps[sup])
#        print(r1)
#        print("p value: {}".format(r2))
#        print("  precision")
#        r1, r2 = compute_p_value(precs[unsup], precs[sup])
#        print(r1)
#        print("p value: {}".format(r2))


#    load_PR_plot(\
#                '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/NUS-WIDE_varyEta/Sat01May2021-221007lr0.0001_unsupervised/model_1',\
#                '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/NUS-WIDE_varyEta/gamma10_varyEta/Sat05Jun2021-042336lr0.0001sup_gamma10_eta20_lamda1/model_1',\
#                "Unsupervised","Supervised $\gamma=10,\eta=20$"\
#                )

    load_PR_plots([\
                    '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/MS-COCO_varyEta/6fold_class79_good_complete/Tue25May2021-203245lr0.0001/model_1',\
                    '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/MS-COCO_varyEta/6fold_class79_good_complete/Wed26May2021-041840lr0.0001sup_eta1_lamda1/model_1',\
                    '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/MS-COCO_varyEta/6fold_class79_good_complete/Wed26May2021-121322lr0.0001sup_eta10_lamda1/model_1',\
                    '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/MS-COCO_varyEta/6fold_class79_good_complete/Thu27May2021-035417lr0.0001sup_eta20_lamda1/model_1',\
                    '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/MS-COCO_varyEta/6fold_class79_good_complete/Thu27May2021-114748lr0.0001sup_eta30_lamda1/model_1',\
                    '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/MS-COCO_varyEta/6fold_class79_good_complete/Thu27May2021-194232lr0.0001sup_eta40_lamda1/model_1',\
                    '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/MS-COCO_varyEta/6fold_class79_good_complete/Fri28May2021-033839lr0.0001sup_eta50_lamda1/model_1'\
                    ],\
                    [
                    'TBH',\
                    'STBH, $\gamma=\eta=1$',\
                    'STBH, $\gamma=\eta=10$',\
                    'STBH, $\gamma=\eta=20$',\
                    'STBH, $\gamma=\eta=30$',\
                    'STBH, $\gamma=\eta=40$',\
                    'STBH, $\gamma=\eta=50$'\
                    ],\
                    'MS-COCO'\
                 )

#    print_weights('cifar10', 32, 512, 600, 1024,\
#                  '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/CIFAR10_varyEta/BatchSize1500/gamma20_varyEta/Sun30May2021-180929lr0.0001sup_gamma20_eta100_lamda1/model_1',\
#                1\
#        )
#    print_weights('nus-wide', 32, 512, 500, 1024, \
#                    '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/NUS-WIDE_varyEta/gamma10_varyEta/Sat05Jun2021-042336lr0.0001sup_gamma10_eta20_lamda1/model_1',\
#                    1 \
#                )
#    print_weights('ms-coco', 32, 512, 600, 1024, \
#                    '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/MS-COCO_varyEta/6fold_class79_good_complete/gamma30_varyEta/Wed02Jun2021-203800lr0.0001sup_gamma30_eta30_lamda1/model_1',\
#                    1 \
#                )
#    load_weights('cifar10', 32, 512, 600, 1024,\
#                   '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/CIFAR10_varyEta/BatchSize1500/Wed05May2021-145440lr0.0001sup_eta20_lamda1/model_1',\
#                1\
#        )
#    load_weights('nus-wide', 32, 512, 500, 1024, \
#                    '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/NUS-WIDE_varyEta/Tue04May2021-083225lr0.0001sup_eta50_lamda1/model_1',\
#                    1 \
#                )

#            )
#    load_kfold('ms-coco', 32, 512, 600, 1024, \
#                '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/MS-COCO_varyEta/6fold_class79_good_complete/gamma30_varyEta/Wed02Jun2021-203800lr0.0001sup_gamma30_eta30_lamda1'
#            )
#    load_kfold('ms-coco', 32, 512, 600, 1024, \
#                '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/MS-COCO_varyEta/6fold_class79_good_complete/gamma30_varyEta/Tue01Jun2021-192050lr0.0001sup_gamma30_eta50_lamda1'
#            )
#    load_kfold('ms-coco', 32, 512, 600, 1024, \
#                '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/MS-COCO_varyEta/6fold_class79_good_complete/gamma30_varyEta/Wed02Jun2021-075620lr0.0001sup_gamma30_eta60_lamda1'
#            )

#  CIFAR-10
#    plot_distribution('cifar10', 10,\
#                        '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/CIFAR10_varyEta/BatchSize1500/gamma20_varyEta/Sun30May2021-180929lr0.0001sup_gamma20_eta100_lamda1/model_1'\
#        )
#  NUS-WIDE
#    plot_distribution('nus-wide', 21,\
#                        '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/NUS-WIDE_varyEta/gamma10_varyEta/Sat05Jun2021-042336lr0.0001sup_gamma10_eta20_lamda1/model_1',\
#                        True\
#                      )
#  MS-COCO
#    plot_distribution('ms-coco', 80,\
#                        '/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/MS-COCO_varyEta/6fold_class80/Sun16May2021-141946lr0.0001/model_1',\
#                        True\
#                        )

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

