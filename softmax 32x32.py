import random
import os
import sys
import math
import numpy as np



trainPath = "D://python2.7.6//MachineLearning//softmax 0-9//trainingDigits"
testPath = "D://python2.7.6//MachineLearning//softmax 0-9//testDigits"
outfile1 = "D://python2.7.6//MachineLearning//softmax 0-9//1.txt"
outfile2 = "D://python2.7.6//MachineLearning//softmax 0-9//2.txt"
outfile3 = "D://python2.7.6//MachineLearning//softmax 0-9//3.txt"
outfile4 = "D://python2.7.6//MachineLearning//softmax 0-9//4.txt"
     

global classDic;classDic={}
 
global epoch;epoch=6
global alpha;alpha=0.2
global nBatch;nBatch=20
######################

def loadData():
    dataList=[]
    labelList=[]
    ###################build obs list 
    for filename in os.listdir(trainPath):
        pos=filename.find('_')
        clas=int(filename[:pos])
        if clas not in classDic:classDic[clas]=1.0
        else:classDic[clas]+=1.0
        labelList.append(clas)
        ##########
        obs=[]
        content=open(trainPath+'/'+filename,'r')
        line=content.readline().strip('\n')
        while len(line)!=0:
            for num in line:
                obs.append(float(num))
            line=content.readline().strip('\n')
        #print '1',len(obs) # 1x1024 dim for each obs
        obs.append(1)#####bias for wx+b 
        dataList.append(obs);#print 'datalist',len(dataList)
    ##########
    print '%d obs loaded'%len(dataList),len(labelList),'labels',len(dataList[0]),'dim'
    #print labelList,classDic
    ####
    outPutfile=open(outfile1,'w')
    for obs in dataList:
        outPutfile.write(str(obs))
        outPutfile.write('\n')
    outPutfile.close()
    #####
    dataMat=np.mat(dataList)
    validMat=np.mat(dataList[125:500]) #each time calc LL need based on same obs
    validLabel=labelList[125:500]
    return dataMat,labelList,validMat,validLabel

def shuffleBatch(num):
    batchList=[]
    numList=range(num)
    random.shuffle(numList)
    eachb=int(num/nBatch)#how many members in each batch
    for i in range(nBatch):
        batchList.append(numList[i*eachb:i*eachb+eachb])
    #print batchList #[[10, 6], [4, 0], [8, 1], [5, 2], [3, 9]]
    return batchList

def initialW(dim,value):
    wMat=np.zeros((10,dim))+value
    wMat=np.mat(wMat)
    return wMat
    
    
def calcLL(validMat,wMat,validLabel):
    num,dim=np.shape(validMat)
    LL=0.0
    for n in range(num):
        fenmu=0.0
        ######
        for k in range(10):
            each=wMat[k,:]*validMat[n,:].T
            each=math.exp(each[0,0])
            fenmu+=each
            if validLabel[n]==k:
                fenzi=wMat[k,:]*validMat[n,:].T
                fenzi=math.exp(fenzi[0,0])
        ########
        LL-=math.log(fenzi/fenmu)
    print 'LL',LL


def calcGrad(wMat,dataMat,labelList): #dataMat could be batch : num of eachbatch x dim #label list change as well
    num,dim=np.shape(dataMat)
    g1Mat=initialW(dim,0.0)#10 x dim

    for kk in range(10): #calc gradk 1xdim

        for n in range(num):
            fenzi=wMat[kk,:]*dataMat[n,:].T
            fenzi=math.exp(fenzi[0,0])
            if labelList[n]==kk:indic=1.0
            else:indic=0.0
            ####
            fenmu=0.0
            for k in range(10):
                each=wMat[k,:]*dataMat[n,:].T
                each=math.exp(each[0,0])
                fenmu+=each
            #####
            prob=math.log(fenzi/fenmu)
            g1Mat[kk,:]=g1Mat[kk,:]-dataMat[n,:]*(indic-prob)
        #######
    #######divide num of obs and normalize
    g1Mat=g1Mat/float(num)
    for kk in range(10):
        ss=g1Mat[kk,:]*g1Mat[kk,:].T
        ss=math.sqrt(ss[0,0])
        g1Mat[kk,:]=g1Mat[kk,:]/ss
    ########
    outPutfile=open(outfile2,'w')
    for k in range(10):
        for d in range(dim):
            outPutfile.write(str(g1Mat[k,d]))
            outPutfile.write(' ')
        outPutfile.write('\n')
    outPutfile.close()
    ####
    return g1Mat
    
    
        
#####################
def loadData1():
    testList=[]
    testLabelList=[]
    ###################build obs list 
    for filename in os.listdir(testPath):
        pos=filename.find('_')
        clas=int(filename[:pos])
        if clas not in classDic:classDic[clas]=1.0
        else:classDic[clas]+=1.0
        testLabelList.append(clas)
        ##########
        obs=[]
        content=open(testPath+'/'+filename,'r')
        line=content.readline().strip('\n')
        while len(line)!=0:
            for num in line:
                obs.append(float(num))
            line=content.readline().strip('\n')
         
        obs.append(1.0)#####bias for wx+b 
        testList.append(obs); 
    ##########
    print '%d test loaded'%len(testList),len(testLabelList),'labels',len(testList[0]),'dim'
     
    ####
    outPutfile=open(outfile3,'w')
    for obs in testList:
        outPutfile.write(str(obs))
        outPutfile.write('\n')
    outPutfile.close()
    #####
    testMat=np.mat(testList)
    return testMat,testLabelList
        

                

        
        
            




###################main
dataMat,labelList,validMat,validLabel=loadData()
num,dim=np.shape(dataMat)
batchList=shuffleBatch(num)#[[10, 6], [4, 0], [8, 1], [5, 2], [3, 9]]
####matric 10xdim 
wMat=initialW(dim,0.2)
wIncMat=initialW(dim,0.0)
gMat=initialW(dim,0.0)

#####
calcLL(validMat,wMat,validLabel)

#gMat=calcGrad(wMat,dataMat,labelList)

 
    
#######train
for ep in range(epoch):
    if ep>2:alpha/=2.0
    batchList=shuffleBatch(num)
    for batch in batchList:
        numobs=len(batch)
        labelList1=[labelList[i] for i in batch]
        dataMat1=np.zeros((numobs,dim))
        dataMat1=np.mat(dataMat1)
        for ii in range(numobs):
            dataMat1[ii,:]=dataMat[batch[ii],:]
        ########## dataMat1 labelList1 ready
        gMat=calcGrad(wMat,dataMat1,labelList1)
        wIncMat=wIncMat/4.0+alpha*(-1)*gMat
        wMat=wMat+wIncMat
        calcLL(validMat,wMat,validLabel)
        
########test
testMat,testLabelList=loadData1()
n1,d1=np.shape(testMat)
err=0.0
for n in range(n1):
    maxP=None;trueL=testLabelList[n];print 'true', trueL
    maxL=0.0
     
    for k in range(10):
        prob=wMat[k,:]*testMat[n,:].T
        prob=prob[0,0]
        if maxP==None or maxP<prob:
            maxP=prob
            maxL=k
    #print 'predict',maxL
    if maxL!=trueL:err+=1.0

print 'err',err/n1
    
        
        
        
        
        
    
    
    







    
    
