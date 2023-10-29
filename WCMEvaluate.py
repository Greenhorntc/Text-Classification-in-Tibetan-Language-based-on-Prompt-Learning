import tensorflow as tf
from transformers import XLMRobertaTokenizer
import numpy as np
from collections import Counter
from datasets import Dataset
from sklearn.metrics import recall_score,precision_score,f1_score
from GetModel import Prompt
from transformers import XLMRobertaTokenizer, TFXLMRobertaModel

#0 读取测试集合
def getTFDataset(config,tokenzier):
    testds = Dataset.load_from_disk(config["testpath"])
    print(testds)
    def tokenize_fun(example):
        return tokenzier(example["textfeatures"], truncation=True, padding=True, max_length=125)

    test_tokenized_ds = testds.map(tokenize_fun, batched=True)
    tf_test_dataset = test_tokenized_ds.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["labels"],
        shuffle=False,
        # collate_fn=data_collator,
        batch_size=7,
    )
    return testds,tf_test_dataset

def get_orlabel(labellist):
    labels = list(reversed(labellist))
    # 因为使用的后缀根据标签添加的，所以每条数据将会产生7条数据
    # 每次取bathcsize条数据，就能得到原始标签
    reallabels = []
    temp = []
    while len(labels) > 0:
        label = labels.pop()
        temp.append(label)
        if len(temp) == 7:
            index = temp.index(1)
            temp.clear()
            reallabels.append(index)
    return reallabels

def compare_lable(labellist):
    temp_class_pre= np.argmax(labellist, axis=1)
    temp_pos=[]
    temp_index=[]
    #标记在这一组预测中，为1的下标和概率
    for i in range(len(labellist)):
        index=np.argmax(labellist[i])
        if index==1:
            temp_index.append(i)
            temp_pos.append(labellist[i][1])

    #判断出现复数个预测为1的概率哪一个更小，然后改变tempclass小的下标为0
    # print(temp_index)
    # print(temp_pos)
    if len(temp_pos) > 1:
        maxvalue=max(temp_pos)
        idx = temp_pos.index(maxvalue)
        index=temp_index[idx]
        #获得了最大的下标，那么其他位置的1都要被设置为0
        for i in range(len(temp_class_pre)):
            if temp_class_pre[i]==1:
                if i !=index:
                    temp_class_pre[i]=0

    elif len(temp_pos)==0:
        #另外一种情况，在temp中一个1 都没有
        temp_max_1=[]
        for i in  range(len(labellist)):
            temp_max_1.append(labellist[i][1])
        maxtemp=max(temp_max_1)
        maxidx=temp_max_1.index(maxtemp)
        #获取了最大下标后就将原来的预测labellist指定的下标置为1
        temp_class_pre[maxidx]=1
    else:
        pass
    # print(temp_class_pre)
    return get_orlabel(temp_class_pre)

def getpredic_label(predicts):
    predicts=tf.nn.softmax(predicts)
    predicts=predicts.numpy()
    labellist=[]
    templist=[]
    for i in range(len(predicts)):
        templist.append(predicts[i])
        if len(templist)==7:
            prelabel=compare_lable(templist)
            labellist.append(prelabel[0])
            templist.clear()
        else:
            pass
    return labellist

def getPredictions(config):
    tokenzier = XLMRobertaTokenizer.from_pretrained(config["checkpoint"])
    test_dataset_reloaded, tf_test_dataset = getTFDataset(config=config, tokenzier=tokenzier)
    basemodelfilepath = config["savemodelname"]
    basemodel = tf.keras.models.load_model(basemodelfilepath)
    print("predict")
    # predics = basemodel.predict(tf_test_dataset.take(10))
    predics=basemodel.predict(tf_test_dataset)

    # 实际标签
    # rellabels = get_orlabel(test_dataset_reloaded["labels"][:10])
    rellabels=get_orlabel(test_dataset_reloaded["labels"])
    return predics,rellabels

def getTarget(predics,rellabels):
    # predics,rellabels=getPredictions(config)
    # [0,1,2,4,5,6,9]
    result = Counter(rellabels)
    print("统计标签个数")
    print(result)
    print("转换后测试集合的标签")
    print(rellabels)
    pre_label = getpredic_label(predics)
    print("模型预测的结果")
    print(pre_label)
    matrix = tf.keras.metrics.Accuracy()
    matrix.update_state(rellabels, pre_label)
    acc = matrix.result().numpy()
    print("测试ACC")
    print(acc)
    Precision = precision_score(rellabels, pre_label, average='weighted')
    print("测试Precision")
    print(Precision)
    recall = recall_score(rellabels, pre_label, average='weighted')
    print("测试Recall")
    print(recall)
    f1 = f1_score(rellabels, pre_label, average='weighted')
    print("测试F1")
    print(f1)
    #返回模型的输出，用于集成学习计算概率。

    """
    tf matirx 函数只能用来计算二分类问题，不能使用计算多分类
    """
    # matrix2=tf.keras.metrics.Precision()
    # matrix2.update_state(rellabels, pre_label)
    # Precision = matrix2.result().numpy()

    # matrix3=tf.keras.metrics.Recall()
    # matrix3.update_state(y_true=rellabels, y_pred=pre_label)
    # recall =matrix3.result().numpy()

    #
    # print("测试F1")
    # F1 = 2 * (Precision * recall) / (Precision + recall)
    # print(F1)



def mergeResultArray(resultlist):
    start=np.concatenate((resultlist[0],resultlist[1]),axis=1)
    if len(resultlist)>=1:
        for i in range(len(resultlist)):
            if i<=1:
                pass
            else:
                start=np.concatenate((start,resultlist[i]),axis=1)
    return start


def  AddResultArray(resultlist):
    start=resultlist[0]+resultlist[1]
    if len(resultlist)>=1:
        for i in range(len(resultlist)):
            if i<=1:
                pass
            else:
                start=resultlist[i]+start
    start = start / len(resultlist)
    return start

def getNegPos(resultlist):
    result=mergeResultArray(resultlist)
    neg = []
    pos = []
    for x in result:
        # print(x)
        temp1 = []
        temp2 = []
        for i in range(len(x)):
            # print(x[i])
            if i % 2 == 0:
                temp1.append(x[i])
            else:
                temp2.append(x[i])
            # find max
            if i == len(x) - 1:
                a = max(temp1)
                b = max(temp2)
                neg.append(a)
                pos.append(b)
                temp1.clear()
                temp2.clear()
    # print(neg)
    # print(pos)
    # result=np.array([neg,pos])
    result = []
    for i, j in zip(neg, pos):
        result.append([i, j])
    result = np.array(result)
    return result

def ensamble(prelist,model="avg"):
    #目前来看测试集合的数据应该是同样的

    #使用了几种提示：
    if model=="avg":
        pre=AddResultArray(prelist)
        print(pre.shape)
        return pre
    if model=="max-min":
        pre=getNegPos(prelist)
        print(pre.shape)
        return pre
    else:
        print("输入的什么勾⑧玩意")
        pass

def getTrainconfig(floder,name):
    train = floder + "/wcmtrain"
    test = floder + "/wcmtest"
    savehisname='logs/WCM/'+name+".txt"
    savemodel = "savedmodel/WCM/Prompt/" +name
    config = {"batch_size": 8, "epochs": 50, "checkpoint": "hfl/cino-small-v2", "trainpath": train, "testpath": test,
               "savemodelname": savemodel,"savehis":savehisname}
    return config

# data0="dataset/WCM/finaleditionprompt0"
data0="dataset/WCM/finaleditionprompt2"
# config0=getTrainconfig(data0,name="finalprompt0")
config0=getTrainconfig(data0,name="Nofinutedprompt2")
predics0,rellabels0=getPredictions(config0)
getTarget(predics0,rellabels0)

# data1="dataset/WCM/finaleditionprompt1"
# config1=getTrainconfig(data1,name="finalprompt1")
# # predics1,rellabels1=getPredictions(config1)
# # getTarget(predics1,rellabels1)
#
# data2="dataset/WCM/finaleditionprompt2"
# config2=getTrainconfig(data2,name="finalprompt2")
# predics2,rellabels2=getPredictions(config2)
# getTarget(predics2,rellabels2)


# # 得到集成学习的结果后，进行预测评价
# resultlist=[predics0,predics1,predics2]
# print("使用集成学习")
# ensambleresult1=ensamble(resultlist,model="avg")
# ensambleresult2=ensamble(resultlist,model="max-min")
# #评估
# print("avg")
# getTarget(ensambleresult1,rellabels1)
# print("max")
# getTarget(ensambleresult2,rellabels1)
# #仅仅是为测试，看看输出是否一样
# getTarget(ensambleresult1,rellabels2)
# getTarget(ensambleresult2,rellabels2)

"""
zero shot 不使用加载训练好的模型直接输出
"""

# #1准备数据，还是用不同prompt1的testdata
# # config0["checkpoint"]='hfl/cino-large-v2'
# print(config2)
# tokenzier = XLMRobertaTokenizer.from_pretrained(config2["checkpoint"])
# testds,tftestds=getTFDataset(config2,tokenzier=tokenzier)
#
# #2直接调用模型
# cinomodel=TFXLMRobertaModel.from_pretrained(config2["checkpoint"])
# model=Prompt(cinomodel,dropout=0)
# #3直接模型输出
# predics = model.predict(tftestds)
# rellabels = get_orlabel(testds["labels"])
# getTarget(predics,rellabels)



