import pandas as pd
import numpy as np
from collections import Counter
import sys

def gini_index(groups):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(group.shape[0])
        if size == 0:
            continue
        score = 0.0
        for i in range(2):
            p = [row[-1] for row in group].count(i) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini
def getSplit(train):
    index, value, score, groups = 0, 0, 1, [train]
    indices = [x*train.shape[0]//10 for x in range(10)]
    for j in range(train.shape[1]-1):
        data = train[train[:,j].argsort()].copy()
        for rowIndex in indices:
            groups1 = np.split(data,np.where(np.diff(data[:,j] < data[rowIndex,j]))[0]+1)
            gini = gini_index(groups1)
            if ( gini < score):
                index, value, score, groups = j, data[rowIndex,j], gini, groups1
    print(score)
    return {'index':index, 'value':value, 'groups':groups, 'score':score}
def split(node, maxDepth, minSize, depth):
    
    if(len(node['groups'])==1 or node['score']==0.0 or depth==maxDepth):
        node['left'] = None
        node['right']= None
        return

    left, right = node['groups']
    if(len(left)+len(right) <= minSize):
        node['left'] = None
        node['right']= None
        return
    
    node['left'] = getSplit(left)
    split(node['left'], maxDepth, minSize, depth+1)
    node['right']= getSplit(right)
    split(node['right'], maxDepth, minSize, depth+1)
    return
def printTree(node,depth,maxDepth):
    if (node==None):
        return
    print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
    printTree(node['left'], depth+1,maxDepth)
    printTree(node['right'], depth+1,maxDepth)
    return
def predict(node,row):
    if(node['left']==None and node['right']==None):
        return (Counter(node['groups'][0][:,-1]).most_common(1)[0][0])
    if( row[node['index']] < node['value'] ):
        return predict(node['left'],row)
    return predict(node['right'],row)
def getAcc(root,data):
    correct = 0;
    for row in data:
        correct += ( row[-1] == predict(root,row))
    return correct
def numerical(x):
    for i in range(x.shape[0]):
        for j in [1,3,5,6,7,8,9,13]:
            x[i,j]= dictionary[x[i,j]]
    return x
def prune(root,node,data,depth):
    if( node==None ):
        return
    if(node['left']==None and node['right']==None):
        return
    accNow = getAcc(root,data)
    left, right = node['left'], node['right']
    prune(root,left,data,depth+1)
    prune(root,right,data,depth+1)
    if(depth>=8):
        node['left'], node['right'] = None, None
        accPruned = getAcc(root, data)
        if( accNow < accPruned ):
            print(accNow, accPruned)
            return
        node['left'], node['right'] = left, right
    return
dictionary={' Private':1, ' Self-emp-not-inc':2, ' Self-emp-inc':3, ' Federal-gov':4, ' Local-gov':5, ' State-gov':6, ' Without-pay':7, ' Never-worked':8,
           ' Bachelors':1, ' Some-college':2, ' 11th':3, ' HS-grad':4, ' Prof-school':5, ' Assoc-acdm':6, ' Assoc-voc':7, ' 9th':8, ' 7th-8th':9, ' 12th':10, ' Masters':11, ' 1st-4th':12, ' 10th':13, ' Doctorate':14, ' 5th-6th':15, ' Preschool':16,
           ' Married-civ-spouse':1, ' Divorced':2, ' Never-married':3, ' Separated':4, ' Widowed':5, ' Married-spouse-absent':6, ' Married-AF-spouse':7,
           ' Tech-support':1, ' Craft-repair':2, ' Other-service':3, ' Sales':4, ' Exec-managerial':5, ' Prof-specialty':6, ' Handlers-cleaners':7, ' Machine-op-inspct':8, ' Adm-clerical':9, ' Farming-fishing':10, ' Transport-moving':11, ' Priv-house-serv':12, ' Protective-serv':13, ' Armed-Forces':14,
           ' Wife':1, ' Own-child':2, ' Husband':3, ' Not-in-family':4, ' Other-relative':5, ' Unmarried':6,
           ' White':1, ' Asian-Pac-Islander':2, ' Amer-Indian-Eskimo':3, ' Other':4, ' Black':5,
           ' Female':1, ' Male':2,
            ' United-States':1, ' Cambodia':2, ' England':3, ' Puerto-Rico':4, ' Canada':5, ' Germany':6, ' Outlying-US(Guam-USVI-etc)':7, ' India':8, ' Japan':9, ' Greece':10, ' South':11, ' China':12, ' Cuba':13, ' Iran':14, ' Honduras':15, ' Philippines':16, ' Italy':17, ' Poland':18, ' Jamaica':19, ' Vietnam':20, ' Mexico':21, ' Portugal':22, ' Ireland':23, ' France':24, ' Dominican-Republic':25, ' Laos':26, ' Ecuador':27, ' Taiwan':28, ' Haiti':29, ' Columbia':30, ' Hungary':31, ' Guatemala':32, ' Nicaragua':32, ' Scotland':33, ' Thailand':34, ' Yugoslavia':35, ' El-Salvador':36, ' Trinadad&Tobago':37, ' Peru':38, ' Hong':39, ' Holand-Netherlands':40}
def main(tr,va,te,vo,to):
    train = pd.read_csv(tr)
    valid = pd.read_csv(va)
    test = pd.read_csv(te)
    # print(train.shape,valid.shape,test.shape)
    dRows, vRows, tRows = train.shape[0], valid.shape[0], test.shape[0]
    # print(dRows,vRows,tRows)
    catColumns = { ' Work Class',' Education',' Marital Status',' Occupation',' Relationship',' Race',' Sex',' Native Country'}
    result = pd.concat(objs=[train,valid,test],axis=0).values
    # result = pd.get_dummies(result,columns=catColumns)
    # print(list(result.columns.values))
    # print(result[0,:])
    result = numerical(result)
    # print(result[0,:])
    train = result[:dRows,:]
    valid = result[dRows:dRows+vRows,:]
    test = result[-tRows:,:]
    # print(train.shape,valid.shape,test.shape)
    
    
    root = getSplit(train)
    split(root, 13, 5, 1)
    prune(root,root,valid,1)
    
    correct = count = 0;
    testPred, validPred = [], []
    
    for row in test:
        testPred.append(predict(root,row))
    
    for row in valid:
        validPred.append(predict(root,row))
    
    np.savetxt(vo,[p for p in validPred],delimiter=',',fmt='%s')
    np.savetxt(to,[p for p in testPred],delimiter=',',fmt='%s')

main(*sys.argv[1:])