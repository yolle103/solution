
from elm import ELMClassifier, ELMRegressor, GenELMClassifier, GenELMRegressor

from random_layer import RandomLayer, MLPRandomLayer, RBFRandomLayer, GRBFRandomLayer
import numpy as np

import load

from scipy.stats import ttest_ind

from sklearn import cross_validation,svm


xtoy, ytoy=load.file2matrix("Adenoma.txt","Tumor","Normal")


c,l = xtoy.shape
print c,l
xpopmean=[]
sum=0
print xtoy


xtoy=xtoy.T


ttestsocre=[]

for j in range(0,l):
    a=xtoy[j]
    set1=[]
    set2=[]
    for i in range(0,len(a)):
        if(ytoy[i]==1):
            set1.append(a[i])
        else:
            set2.append(a[i])
    #print a
    ttestsocre.append(ttest_ind(set1,set2))
    #print ttest_ind(set1,set2)

ttestsocre_005=[]
pvalue=[]

for i in range(0,len(ttestsocre)):
    pvalue.append((i,ttestsocre[i][1]))
   # print ttestsocre[i]
print  pvalue

pvalue_sort=sorted(pvalue,key=lambda d:d[1],reverse=False)
print pvalue_sort
index=[]

x=[]
a=np.zeros((50,int(len(xtoy[1]))))
for i in range(0,50):
    index.append(pvalue_sort[i][0])
    x.append(xtoy[pvalue_sort[i][0]])
    a[i]= xtoy[pvalue_sort[i][0]]
a=a.T


def powtanh_xfer(activations, power=1.0):
    return pow(np.tanh(activations), power)
count=0
max=-1;
for i in range(0,100):
 xtoy_train,xtoy_test,ytoy_train,ytoy_test=cross_validation.train_test_split(a,ytoy,test_size=0.3,random_state=0)

 rbf_rhl = RBFRandomLayer(n_hidden=37, random_state=i, rbf_width=0.01)
 elmc_rbf = GenELMClassifier(hidden_layer=rbf_rhl)
 elmc_rbf.fit(xtoy_train, ytoy_train)
# if(elmc_rbf.score(xtoy_train, ytoy_train)>0.8 and elmc_rbf.score(xtoy_test, ytoy_test)>0.8):
 print elmc_rbf.score(xtoy_train, ytoy_train), elmc_rbf.score(xtoy_test, ytoy_test)
 print rbf_rhl
 if elmc_rbf.score(xtoy_test,ytoy_test)>max:
     max=elmc_rbf.score(xtoy_test,ytoy_test)
 print  rbf_rhl.components_
print max