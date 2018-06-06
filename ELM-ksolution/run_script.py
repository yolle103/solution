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


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

def get_parser():
    parser = argparse.ArgumentParser(description='multiple ELM test')
    parser.add_argument('-d', '--data_set', help='data set')
    parser.add_argument('-s', '--save_dir', help='save dir')
    parser.add_argument('-f', '--feature_num', help='number of feature used to train ELM/other algorithm')
    parser.add_argument('-m', '--model_num', help='run model num')
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

def make_new_data(dataset, pvalue_sort, top_k):
    index = []
    x = []
    dataset = dataset.T
    new_data = np.zeros((top_k, int(len(dataset[1]))))
    for i in range(0,20):
        index.append(pvalue_sort[i][0])
        x.append(dataset[pvalue_sort[i][0]])
        new_data[i]=dataset[pvalue_sort[i][0]]
    new_data = new_data.T
    logging.info('new data shape: {}'.format(np.shape(new_data)))
    return new_data
 
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


def run_ELM(x, y, threshold, test_num, n_hidden, random_state = 2018, kernel_type='MLP', ):
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

    method_option = {'SVC' : SVC, 
            'GaussianNB': GaussianNB}
    model = method_option[baseline_method]()
    model.fit(x_train, y_train)
    test_acc = accuracy_score(model.predict(x_test), y_test)
    train_acc = accuracy_score(model.predict(x_train), y_train)
    logging.info('baseline method: {}, train_acc: {}, test_acc: {}'
            .format(baseline_method, train_acc, test_acc))

    return train_acc, test_acc

def save_models(models, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    score_list = []
    model_weight = []
    for item in models:
        score_list.append([item[0], item[1]])
        model_weight.append(item[2])

    model_path = os.path.join(save_dir, 'model.npy')
    score_txt_path = os.path.join(save_dir, 'score.txt')
    np.save(model_path, model_weight)
    np.savetxt(score_txt_path, score_list)

def main():
    args = get_parser()
    data_path = args.data_set
    save_dir = args.save_dir
    feature_num = int(args.feature_num)
    model_num = int(args.model_num)
    logging.info('runing test with data: {}, feature num: {}'
            .format(data_path, feature_num))
    x, y = load_data(data_path)
    logging.info('load data shape: x shape:{}, y_shape:{} '
            .format(np.shape(x), np.shape(y)))
    p_value_list = p_value_sort(x, y)
    logging.info('sorted by p_value, showing top 5 p_value& feature index:{}'
            .format(p_value_list[:5]))

    if feature_num == -1:
        # use all feature!
        feature_num = np.shape(x)[1]
    logging.info('select top {} features'.format(feature_num))
    x = make_new_data(x, p_value_list, feature_num) 

    ELM_result = run_ELM(x, y, 0.8, model_num, 200, 2018)
    # calculate baseline result
    baseline_train(x, y, 'GaussianNB')
    baseline_train(x, y, 'SVC')
    save_models(ELM_result, save_dir)


if __name__ == '__main__':
    main()
