import json
import random
from sklearn import decomposition
import matplotlib.pyplot as plt
from tflib.gmm import GMM, shownormal
from tflib.normal import Normal



file_path = "./data/"

with open(file_path + 'cate_item.json') as f:
    # temp = zip(*zip(*json.load(f)))
    # print temp[0]
    train_bycate = dict(json.load(f))

catelist = random.sample(train_bycate.keys(), 3)


def generate_all_vector(itemlist):
    vectorpath = '/home/cuizeyu/image_vector/'
    vect_batch = []
    for name in itemlist:
        with open(vectorpath + name + '.json') as f:
            vect_batch.append(json.load(f))
    return vect_batch


def pca_2(vector_list):
    pca = decomposition.PCA(n_components=2, copy=True)
    vector_2 = pca.fit_transform(vector_list)

    return vector_2


def plot_pca(vector_2):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_title('Scatter Plot')

    plt.xlabel('X')
    plt.ylabel('Y')

    ax1.scatter(vector_2[:,0], vector_2[:,1], c='r', marker='o')
    # plt.legend('x1')
    plt.show()

    return


for cate in catelist:
    itemlist = train_bycate[cate]
    vector_list = generate_all_vector(itemlist)
    vector_2 = pca_2(vector_list)
    # print(vector_2.shape)
    # plot_pca(vector_2)

    gmm = GMM(dim = 2, ncomps = 4,data = vector_2, method = "random")
    print gmm
    shownormal(vector_2,gmm)

    gmm.em(vector_2,nsteps=1000)
    shownormal(vector_2,gmm)
    print gmm
    ngmm = gmm.condition([0],[-3])
    print ngmm.mean()
    print ngmm.covariance()







