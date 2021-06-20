from plot_PR import *

from sklearn.manifold import TSNE
import pandas as pd

def plot_tSNE(path):
    test_code,test_label,train_code,train_label = load_codes(path)

    x = test_code
    y = test_label
    y = y.astype(int)

    df = pd.DataFrame()
    df["y"] = np.where( y==1 )[1]

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(x)
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df).set(title="")
    plt.savefig(os.path.join(path, "_tSNE"))
    plt.show()
    plt.close()

if __name__ == '__main__':
    plot_tSNE(
             "/Users/yuanchen/Documents/University of Geneva/Thesis/L2H/Reproduce/report_v2/CIFAR10_varyEta/BatchSize1500/Wed05May2021-083923lr0.0001sup_eta10_lamda1/model_1",\
             )
