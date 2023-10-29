import random

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import sklearn
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


#plot结果
def plotloss(history,label,n):
    #使用大的范围来显示 数据的 wide range
    plt.plot(history.epoch,history.history["loss"],colors[n],label="Train"+label)
    plt.plot(history.epoch,history.history["val_loss"],colors[n],label="Val"+label,linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

def plot_metrics(history):
    metrics=["loss","auc","precision","recall"]
    for n,metric in enumerate(metrics):
        name=metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,history.history[metric],color=colors[0],label="Train")
        plt.plot(history.epoch,history.history["val_"+metric],color=colors[0],linestyle="--",label="val")
        plt.xlabel("Epoch")
        plt.ylabel(name)
        if metric=="loss":
            plt.ylim([0,plt.ylim()[1]])
        elif metric=="auc":
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend()

def plot_metrics2(history,nforc,modelname):
    metrics=["loss","auc","precision","recall"]
    for n,metric in enumerate(metrics):
        name=metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,history.history[metric],color=colors[nforc],label=modelname+"Train")
        plt.plot(history.epoch,history.history["val_"+metric],color=colors[nforc],linestyle="--",label=modelname+"val")
        plt.xlabel("Epoch")
        plt.ylabel(name)
        if metric=="loss":
            plt.ylim([0,plt.ylim()[1]])
        elif metric=="auc":
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend()

def plotconfusionMatrix(labels,predictions,p=0.6):
    #confusion matrix
    cm=confusion_matrix(labels,predictions>p)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm,annot=True,fmt='d',cmap=plt.cm.Greens)
    plt.title('confusion matrix @{:.2f}'.format(p))
    plt.ylabel("Actual label")
    plt.xlabel("Prection label")
    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))

def plotROC(name,labels,predictions,**kwargs):
    fp, tp,_=sklearn.metrics.roc_curve(labels,predictions)
    plt.plot(100*fp,100*tp,label=name,linewidth=2,**kwargs)
    plt.xlabel("False positives[%]")
    plt.ylabel("True positives[%]")
    plt.xlim([-0.5, 100])
    plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.legend(loc="lower right")

def plotLoss(historylist, namelist):
    for i in range(len(historylist)):
        history = historylist[i]
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.plot(history.epoch, history.history["loss"], color=colors[i], label="Training-" + namelist[i])
        plt.plot(history.epoch, history.history["val_loss"], color=colors[i],linestyle="--", label="Val-" + namelist[i])
        plt.legend()

def plotLoss2(historylist, namelist):
    for i in range(len(historylist)):
        history = historylist[i]
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.plot(history["epoch"], history["loss"], color=colors[i], label="Training-" + namelist[i])
        plt.plot(history["epoch"], history["Test_loss"], color=colors[i],linestyle="--", label="Test-" + namelist[i])
        plt.legend()


def plotACC(historylist, namelist):
    for i in range(len(historylist)):
        history = historylist[i]
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.plot(history.epoch, history.history["accuracy"], color=colors[i], label="Training-" + namelist[i])
        plt.plot(history.epoch, history.history["Val_accuracy"], color=colors[i],linestyle="--", label="Val-" + namelist[i])
        plt.legend()

def plotACC2(historylist, namelist):
    for i in range(len(historylist)):
        history = historylist[i]
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.plot(history['epoch'], history["accuracy"], color=colors[i], label="Training-" + namelist[i])
        plt.plot(history['epoch'], history["Test_accuracy"], color=colors[i],linestyle="--", label="Test-" + namelist[i])
        plt.legend()


def plotPrecsion(historylist, namelist):
    for i in range(len(historylist)):
        history = historylist[i]
        plt.xlabel("epoch")
        plt.ylabel("Precision")
        plt.plot(history.epoch, history.history["precision"], color=colors[i], label="Training-" + namelist[i])
        plt.plot(history.epoch, history.history["val_precision"], color=colors[i], linestyle="--",label="Val-"+namelist[i])
        plt.legend()


def plotRecall(historylist, namelist):
    for i in range(len(historylist)):
        history = historylist[i]
        plt.xlabel("epoch")
        plt.ylabel("Recall")
        plt.plot(history.epoch, history.history["recall"], color=colors[i], label="Training-" + namelist[i])
        plt.plot(history.epoch, history.history["val_recall"], color=colors[i], linestyle="--",label="Val-" + namelist[i])
        plt.legend()


def plotfewshotacc(acclist,name,i):
    plt.xlabel("few-shot")
    plt.ylabel("Model ACC")
    few=[10,20,30,40,50]
    plt.plot(few, acclist, color=colors[i], label="Acc-" + name)
    plt.legend()


def plotfewshotf1(f1list, name,i):
    plt.xlabel("few-shot")
    plt.ylabel("Model F1")
    few = [10, 20, 30, 40, 50]
    plt.plot(few, f1list, color=colors[i], linestyle="--", label="F1-" + name)
    plt.legend()

def plotfewshot(acclist,f1list,name,i):
    # plotfewshotacc(acclist,name,i)
    plotfewshotf1(f1list,name,i)
