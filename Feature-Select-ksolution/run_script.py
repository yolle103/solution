# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import logging
from scipy.stats import ttest_ind
import load
from sklearn.cross_validation import train_test_split
from elm import ELMClassifier, ELMRegressor, GenELMClassifier, GenELMRegressor
from sklearn import cross_validation,svm
from random_layer import RandomLayer, MLPRandomLayer, RBFRandomLayer, GRBFRandomLayer
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from functools import partial
import multiprocessing
from itertools import combinations
from elm import  GenELMClassifier
from sklearn.naive_bayes import MultinomialNB 
from sklearn.tree import DecisionTreeClassifier
import util
from sklearn import neighbors  
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
import time

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

def get_parser():
    parser = argparse.ArgumentParser(description='multiple ELM test')
    parser.add_argument('-d', '--data_set', help='data set')
    parser.add_argument('-s', '--save_dir', help='save dir')
    parser.add_argument('-t', '--triple_num', help='number of feature used to assemble triple')
    return parser.parse_args()



def load_data(data_path):
    logging.info('loading data from: {}'.format(data_path))
    class1 = ''
    class2 = ''
    load_method = 'file2matrixOneLine'
    if 'Adenoma' in data_path:
        class1 = 'Tumor'
        class2 = 'Normal'
    if 'CNS' in data_path:
        class1 = '1'
        class2 = '0'

    if 'DLBCL' in data_path:
        class1 = 'DLBCL'
        class2 = 'follicular_lymphoma'
        load_method = 'file2matrix'
    if 'ALL' in data_path:
        class1 = 'TRUE'
        class2 = 'FALSE'
        load_method = 'file2matrix'
    option = {'file2matrix':load.file2matrix, 
            'file2matrixOneLine':load.file2matrixOneLine}

    x, y = option[load_method](data_path, class1, class2)
    
    return x , y


def p_value_sort(dataset,labelset):
    xtoy, ytoy= dataset, labelset
    xtoy = xtoy.T
    l,c = xtoy.shape  #l代表特征数,c代表样本个数

    # 计算每个特征的T-Test值，二元组（T值，P值）
    # calculate each feature's T-test score (t, p value)
    ttestsocre=[]
    
    for j in range(0, l): 
        a=xtoy[j]
        set1=[] #正样本
        set2=[] #负样本
        for i in range(0,len(a)): #每个样本分类
            if(ytoy[i] == 1):
                set1.append(a[i])
            else:
                set2.append(a[i])        
        ttestsocre.append(ttest_ind(set1,set2))
    logging.info('t test finish ')
    pvalue=[]    

    for i in range(0,len(ttestsocre)):
        pvalue.append((i,ttestsocre[i][1]))#读出P值           

    pvalue_sort=sorted(pvalue,key=lambda d:d[1],reverse=False)#根据P值排名，并记录下特征的索引
    return pvalue_sort


def run_ELM(x, y, threshold, n_hidden, random_state = 2018, kernel_type='MLP'):
    #  split the data set into train/test
    x_train,x_test,y_train,y_test=cross_validation.train_test_split(
            x, y, test_size=0.3, random_state=random_state)
    # currently only support test_num <=100k
    assert test_num <= 100000

    def powtanh_xfer(activations, power=1.0):
        return pow(np.tanh(activations), power)

    model_count = 0
    result = []
    hidden_options = {
            'MLP': MLPRandomLayer, 
            'RBF': RBFRandomLayer, 
            'GRBF': GRBFRandomLayer}

    for i in range(0, test_num):
        tanh_rhl = hidden_options[kernel_type](
                n_hidden=n_hidden, 
                random_state=i, 
                activation_func=powtanh_xfer, 
                activation_args={'power':3.0})
        elmc_tanh = GenELMClassifier(hidden_layer=tanh_rhl)
        # start Training
        elmc_tanh.fit(x_train, y_train)
        # calculate score
        train_acc = elmc_tanh.score(x_train, y_train)
        test_acc = elmc_tanh.score(x_test, y_test)
        if train_acc  > threshold and test_acc > threshold:
            logging.info(
                    'find model satisfiy threshold, train_acc: {}, test_acc: {}'
                    .format(train_acc, test_acc))
            result.append((train_acc, 
                test_acc,
                tanh_rhl.components_['weights']))

            model_count += 1
    logging.info('fininsh training, get {} valid models'
            .format(model_count))

    result.sort(key=lambda x: x[1],reverse=True)
    return result

def baseline_train(x, y, baseline_method = 'SVC', random_state = 2018):
    #  split the data set into train/test
    x_train,x_test,y_train,y_test=cross_validation.train_test_split(
            x, y, test_size=0.3, random_state=random_state)

    method_option = {'SVC' : SVC, 'GaussianNB': GaussianNB}
    model = method_option[baseline_method]()
    model.fit(x_train, y_train)
    test_acc = accuracy_score(model.predict(x_test), y_test)
    train_acc = accuracy_score(model.predict(x_train), y_train)
    logging.info('baseline method: {}, train_acc: {}, test_acc: {}'
            .format(baseline_method, train_acc, test_acc))

    return train_acc, test_acc

def save_models(models, p_value_list, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    model_path = os.path.join(save_dir, 'models.npy')
    p_value_list_dir = os.path.join(save_dir, 'p_value_list.npy')
    np.save(model_path, models)
    np.save(p_value_list_dir, p_value_list)

def classify_func(combination, clf, xtoy, ytoy, i):    
    a=np.zeros((3,int(len(xtoy[1]))))#存储特征穷举三元组的原始数据 
    a[0]=xtoy[combination[i][0]] #F1
    a[1]=xtoy[combination[i][1]] #F2
    a[2]=xtoy[combination[i][2]] #F3
    aa=a.T #将a转置为样本*特征的矩阵
    Sn, Sp, Acc, Avc=util.statistics(ytoy,util.kFoldTest(clf,aa,ytoy))
    logging.info('case: {}, sn: {}, sp: {}, Acc:{}'
            .format(i, Sn, Sp, Acc))
    return Sn,Sp,Acc,Avc,i

def TripleTest(x, y, pvalue_sort, top_k, threshold, classifer):
    index=[] 
    count=0
    for i in range(0, top_k): #取p_value值top x进行穷举
         index.append(pvalue_sort[i][0])
    
    if classifer == 'ELM' :      
        rbf_rhl = RBFRandomLayer(n_hidden=20, rbf_width=0.01,random_state=2018)
        clf = GenELMClassifier(hidden_layer=rbf_rhl)    
    elif classifer == 'SVM':
        clf = SVC(kernel='linear', C=1)
    elif classifer == 'KNN':
        clf=neighbors.KNeighborsClassifier(n_neighbors = 3 )
    elif classifer == 'Normal_Bayes':
        clf=MultinomialNB(alpha=0.01)
    else:
        clf = DecisionTreeClassifier(random_state=0)

    combination = list(combinations(index,3))#前50个特征穷举，公19600组
                
    result=[] 
    #存储测试集正确率和训练集正确率都大于0.9的特征组合
    #（（特征组合），训练集正确率，测试集正确率）   
    value_set = []
    i_list = list(range(len(combination)))
    worker = partial(classify_func, combination, clf, x.T, y)    
    # running in multithread
    pool = multiprocessing.Pool(4)

    pool_result = pool.map(worker, i_list)
    pool.close()
    pool.join()

    for res in pool_result:
        if res[2] >= threshold:
            result.append(
                    [combination[res[4]],res[2],res[3],res[0],res[1]])
            count+=1
        value_set.append(res[2])

    return result, count, max(value_set)

def get_rank(feature_index, p_value_list):
    for i in range(len(p_value_list)):
        if p_value_list[i][0] == feature_index:
            return i

def draw_3D_pic(result, p_value_list, save_dir, best_ten=False):
    if best_ten:
        result.sort(key=lambda x: x[1],reverse=True)
        result = result[:10]

    x = []
    y = []
    z = []
    c = []
    for each in result:
        tri = each[0]
        x.append(get_rank(tri[0], p_value_list))
        y.append(get_rank(tri[1], p_value_list))
        z.append(get_rank(tri[2], p_value_list))
        c.append(each[1])
    fig1 = plt.figure(figsize=(4.7,2.5))
    title = "all triplets"
    fig1.suptitle(title)            
    cm1 = plt.cm.get_cmap('RdYlBu_r')  
    axes = fig1.add_subplot(111, projection='3d')
    vmin = min(c)
    vmax = max(c)

    p=axes.scatter(x,y,z,c=c,vmin=vmin,vmax=vmax,s=50,cmap=cm1)              #enable this line and disable the previous line to draw all models with accuracy over cut-off value
    axes.set_xlabel('F1')
    axes.set_ylabel('F2')
    axes.set_zlabel('F3')
    axes.set_xlim(0, 50)
    axes.set_ylim(0, 50)
    axes.set_zlim(0, 50)
    my_x_ticks = np.arange(0, 50, 5)
    
    axes.set_xticks(my_x_ticks)
    axes.set_yticks(my_x_ticks)
    axes.set_zticks(my_x_ticks)
    axins = inset_axes(axes,
        width="2%",  # width = 10% of parent_bbox width
        height="40%",  # height : 50%
        loc=3,
        bbox_to_anchor=(1.00, 0., 1, 1),
        bbox_transform=axes.transAxes,
        borderpad=0,
        )
    
    
    plt.colorbar(p,axins)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, 'triple_result.svg'), format='svg') 
    plt.show()


def main():
    args = get_parser()
    data_path = args.data_set
    save_dir = args.save_dir
    triple_num = int(args.triple_num)
    logging.info('runing test with data: {}'
            .format(data_path))
    x, y = load_data(data_path)
    logging.info('load data shape: x shape:{}, y_shape:{} '
            .format(np.shape(x), np.shape(y)))
    p_value_list = p_value_sort(x, y)
    logging.info('sorted by p_value, showing top 5 p_value& feature index:{}'
            .format(p_value_list[:5]))

    logging.info('use top {} features to form triple'
            .format(triple_num))
    logging.info('triple test on ELM')
    start = time.time()
    ELM_result, count, max_acc  = TripleTest(
            x, y, p_value_list, triple_num, 0.8, 'ELM')
    end = time.time()
    logging.info('run time: {} seconds'.format(end - start))
    logging.info('{} case satisfy threshold, max acc: {}'
            .format(count, max_acc))
    draw_3D_pic(ELM_result, p_value_list, save_dir, True)
    save_models(ELM_result, p_value_list, save_dir)
#    logging.info('triple test on SVM')
#    TripleTest(x, y, p_value_list, triple_num, 0.8, 'SVM')
#    logging.info('triple test on KNN')
#    TripleTest(x, y, p_value_list, triple_num, 0.8, 'KNN')
     


if __name__ == '__main__':
    main()
