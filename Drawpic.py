import os.path
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import re

sns.set_theme(style="darkgrid")
sns.set()
"""
few-shot 作图
"""
#1 read floder
prompt="logs/prompt"
textcnn="logs/textcnn"
RNN="logs/rnn"
Rcnn="logs/rcnn"
Dpcnn="logs/dpcnn"
floderlist=[prompt,textcnn,RNN,Rcnn,Dpcnn]
def readtxt(file):
    with open(file=file,encoding="utf-8") as file:
        lines=file.readlines()
        acc=lines[0].split("测试")[0]
        # acc=lines[0].split("测试")[0]
        # acc=lines[0].split("测试")[0]
        f1=lines[3].split("测试")[0]
    return float(acc),float(f1)

def getmodelvaluelist(floder):
    files=os.listdir(floder)
    files=[ floder+"/" +x for x in files]
    acclist=[]
    f1list=[]
    for file in files:
        acc,f1=readtxt(file)
        acclist.append(acc)
        f1list.append(f1)
    return acclist,f1list

def getfewdf(floder):
    valuedic={}
    few = [10, 20, 30, 40, 50]
    acc,f1=getmodelvaluelist(floder)
    valuedic["few-shot"]=few
    valuedic["acc"]=acc
    valuedic["f1"]=f1
    df=pd.DataFrame.from_dict(valuedic)
    return df

def getallfew(floderlist):
    hislist=[]
    name=[x.split("/")[1] for x in floderlist ]
    for floder in  floderlist:
        hislist.append(getfewdf(floder))
    return hislist,name


def plotfewacc(historylist, namelist):
    for i in range(len(historylist)):
        history = historylist[i]
        sns.lineplot(data=history,x="few-shot",y='acc',label="Acc-" + namelist[i])
def plotfewf1(historylist, namelist):
    for i in range(len(historylist)):
        history = historylist[i]
        sns.lineplot(data=history,x="few-shot",y='f1',linestyle="--", label="F1-" + namelist[i])

all,name=getallfew(floderlist)
# plotfewacc(all,name)
# plt.show()
plotfewf1(all,name)
plt.show()




"""
fulldata 作图
"""
def gethisfiles(floder):
    files=os.listdir(floder)
    name=[x.split(".txt")[0] for x in files]
    files=[floder+x for x in files]
    return files,name

fullhisfld="logs/fulldata/"
fullhis=gethisfiles(fullhisfld)

#1:getvalue
def gethisvalues(txt):
    with open(txt,encoding="utf-8") as f:
        lines=f.readlines()
        print(lines)

def readlogs(file):
    valuelist=[]
    with open(file=file,encoding="utf-8") as f:
        lines=f.readlines()
        for i in range(len(lines)):
            data = lines[i].strip(" \n")
            if i==0:
                datalist=getepochlist(data)
            else:
                datalist = getvalue(data)
            valuelist.append(datalist)
        return todf(valuelist)

def todf(datalist):
    datadic={}
    datadic["epoch"]=datalist[0]
    datadic["loss"]=datalist[1]
    datadic["Val_loss"]=datalist[2]
    datadic["accuracy"]=datalist[3]
    datadic["Val_accuracy"]=datalist[4]
    df=pd.DataFrame.from_dict(datadic)
    return df

def getepochlist(epoch):
    epochlist=list(epoch)
    if len(epochlist)==10:
        return list(range(0, 10, 1))
    if len(epochlist)>10:
        tmp=epochlist[-2:][0]+epochlist[-2:][1]
        return list(range(0,int(tmp)+1,1))
    else:
        return list(range(0,int(epochlist[-1])+1,1))
def getvalue(value):
    # print(len(value))
    list_index = [i.start() for i in re.finditer("\.", value)]
    # print(list_index)
    valuelist=[]
    for i in range(len(list_index)-1):
        start=list_index[i]
        end=list_index[i+1]
        relvalue=value[start:end]
        valuelist.append(relvalue)
    # print(list_index[-1])
    s=list_index[-1]-1
    valuelist.append(value[s:])
    valuelist=[float(x) for x in valuelist]
    return valuelist

def gethisdf(floder):
    alldflist=[]
    fullhis ,name= gethisfiles(floder)
    # for file in fullhis:
    # df=readlogs(fullhis)
    for i in range(len(fullhis)):
        df=readlogs(fullhis[i])
        alldflist.append(df)
    return alldflist,name

def plotfullacc(historylist, namelist):
    for i in range(len(historylist)):
        history = historylist[i]
        sns.lineplot(data=history,x="epoch",y='accuracy',label="Training-" + namelist[i])
        sns.lineplot(data=history,x="epoch",y='Val_accuracy',linestyle="--", label="Val-" + namelist[i])
def plotfullloss(historylist, namelist):
    for i in range(len(historylist)):
        history = historylist[i]
        sns.lineplot(data=history,x="epoch",y='loss',label="Training-" + namelist[i])
        sns.lineplot(data=history,x="epoch",y='Val_loss',linestyle="--", label="Val-" + namelist[i])


"""
tncc全量 loss
"""
# alldf,name=gethisdf(fullhisfld)
# hislist1=alldf[0:3]
# namelist1=name[0:3]
# hislist2=alldf[3:7]
# namelist2=name[3:7]
# hislist3=alldf[7:10]
# namelist3=name[7:10]
# plt.subplot(1,3,1)
# plotfullloss(hislist1,namelist=namelist1)
# plt.legend(fontsize=8)
# plt.subplot(1,3,2)
# plotfullloss(hislist2,namelist=namelist2)
# plt.legend(fontsize=8)
# plt.subplot(1,3,3)
# plotfullloss(hislist3,namelist=namelist3)
# plt.legend(fontsize=8)
# plt.show()

"""
tncc全量 acc
"""
# plt.subplot(1,3,1)
# plotfullacc(hislist1,namelist=namelist1)
# plt.legend(fontsize=8)
# plt.subplot(1,3,2)
# plotfullacc(hislist2,namelist=namelist2)
# plt.legend(fontsize=8)
# plt.subplot(1,3,3)
# plotfullacc(hislist3,namelist=namelist3)
# plt.legend(fontsize=8)
# plt.show()

"""
wcm training
"""
# logsdir='logs/WCM/'
# alldf,name=gethisdf(fullhisfld)
#
# # vauledic3=readlogs(logsfiles[2])
# hislist=alldf[0:3]
# namelist = ["Using-Prompt1-small","Using-Prompt2-small","Using-Prompt3-small"]
# #
# plt.subplot(1,2,1)
# plotfullloss(hislist,namelist=namelist)
# plt.legend(fontsize=8)
# plt.subplot(1,2,2)
# plotfullacc(hislist,namelist=namelist)
# plt.legend(fontsize=8)
# plt.show()