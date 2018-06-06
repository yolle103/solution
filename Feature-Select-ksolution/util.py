# -*- coding: utf-8 -*-
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
#函数功能：传入分类方法，进行10折交叉检验，返回测试集
def kFoldTest(clf, raw_data, raw_target):
    predict = []
    kf = KFold(n_splits=10)    
    for train_index, test_index in kf.split(raw_data):
        X_train, X_test = raw_data[[train_index]], raw_data[[test_index]]
        #Y_test在这里没作用，为了数据变量对齐0.0
        Y_train = raw_target[:test_index[0]] + raw_target[test_index[-1]+1:]        
        clf.fit(X_train, Y_train)
        test_target_temp=clf.predict(X_test)
        predict.append(test_target_temp)    
    test_target = [i for temp in predict for i in temp]#将10次测试集展平
    return test_target



#函数功能：统计正确率，以及每个类的正确数
def statistics(y_real, y_pred):
    conf_matrix = confusion_matrix(y_real, y_pred)
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    Sn = float(TP)/(TP+FN)
    Sp = float(TN)/(TN+FP)
    Acc = float(TP+TN)/(TP+FP+TN+FN)
    Avc = float(Sn+Sp)/2
    return Sn,Sp,Acc,Avc

if __name__ == '__main__':
    from sklearn import neighbors
    clf=neighbors.KNeighborsClassifier(n_neighbors = 1 )
    print(statistics([0,1,0,0],kFoldTest(clf,[1,2,3,4],[0,1,0,1])))
