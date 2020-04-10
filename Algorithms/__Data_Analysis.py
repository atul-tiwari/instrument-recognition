import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn
import pandas as pd

def plot_conf_matrix(matrix):
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')

    top = matrix.reshape(9,)
    bottom = np.zeros_like(top)
    width = depth = 2

    x = [5,10,15]*3
    y = [5]*3+[10]*3+[15]*3
    color = ['green']+['red']*3+['green']+['red']*3+['green']

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True,color=color,label='correct')
    ax1.set_title('Confussion matrix')
    ax1.set_xlabel('True')
    ax1.set_ylabel('Predection')
    ax1.set_zlabel('Samples')
    plt.show()

        # data to plot
    n_groups = 3
    a = matrix.diagonal()
    b = np.subtract(matrix.sum(axis = 0), np.array(a) )

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, a, bar_width,
    alpha=opacity,
    color='b',
    label='correct')

    rects2 = plt.bar(index + bar_width, b, bar_width,
    alpha=opacity,
    color='g',
    label='wrong')

    plt.xlabel('Categories')
    plt.ylabel('test samples')
    plt.xticks(index + bar_width, ('ele', 'flu', 'pia'))
    plt.legend()

    plt.tight_layout()
    plt.show()
    df_cm = pd.DataFrame(matrix, range(3),
                    range(3))
    #plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt = '.1f')# font size
    plt.show()
