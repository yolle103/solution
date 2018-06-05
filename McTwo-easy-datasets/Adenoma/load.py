#This file aims to resolve the problem of load of input matrix and label
#It contains 3 functions
from numpy import *
import operator
from os import listdir 
from time import sleep
import numpy as np
#This function aims to load the input matrix
#parameter filename is supposed to be the name of raw file
#mat refers to the input matrix
#name refers to the list of gene names
def file2matrixOneLine(filename,PosName,NeName):
    fr=open(filename)
    arr=fr.readline()
    arr=arr.split('\r')
    List=arr[0].split('\t')
    Label=List[1:int(shape(List)[0])]
    tempMat=arr[1:int(shape(arr)[0])]
    index=0
    returnMat = zeros((int(shape(tempMat)[0]),int(shape(Label)[0])))
    for line in tempMat:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[1:int(shape(listFromLine)[0])]
        index += 1
    label=[]
    for na in Label:
        if(na==PosName):
             label.append(1)
        if(na==NeName):
             label.append(-1)
    np.save("Fea.npy",tempMat)
    mat = transpose(returnMat)
    return mat,label
def file2matrix(filename,PosName,NeName):
    fr = open(filename)
    arr=fr.readline()
    arr = arr.strip()
    name=arr.split('\t')
    #print(shape(name))
    #print(name[0])
    #print(name[1])
    name=name[1:shape(name)[0]]
    print(shape(name))
    label=[]
    for na in name:
        if(na==PosName):
             label.append(1)
        if(na==NeName):
             label.append(-1)
    print(shape(label),shape(name))
    arrayOLines = fr.readlines()
    listFromLine = arrayOLines[0].split('\t')
    numberOfLines = len(arrayOLines)       #get the number of lines in the file
    returnMat = zeros((numberOfLines,int(shape(listFromLine)[0])-1))#prepare matrix to return
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[1:int(shape(listFromLine)[0])]
        index += 1
	mat = transpose(returnMat);
    return mat,label
#This function aims to load the first set of label	
#parameter filename is supposed to be the name of raw file containing first set of labels
#labelMat refers to the label
def loadlabel3(filename):
    fr=open(filename)
    fr.readline()
    array=fr.readlines()
    list=array[0].split('\t')
    number=len(array)
    labelMat=[]
    for line in array:
        line = line.strip()
        list = line.split('\t')
        if(list[-1]=='FS'):
             labelMat.append(1.0)
        if(list[-1]=='NS'):
             labelMat.append(-1.0)
    return labelMat	
#This function aims to load the second set of label	
#parameter filename is supposed to be the name of raw file containing second set of labels
#labelMat refers to the label
def loadlabel2(filename):
    fr=open(filename)
    fr.readline()
    array=fr.readlines()
    list=array[0].split('\t')
    number=len(array)
    labelMat=[]
    for line in array:
        line = line.strip()
        list = line.split('\t')
        if(list[-1]=='S'):
             labelMat.append(1.0)
        else:
             labelMat.append(-1.0)
    return labelMat	
