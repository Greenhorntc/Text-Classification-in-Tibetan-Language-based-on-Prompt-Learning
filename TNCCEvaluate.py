import tensorflow as tf
from transformers import XLMRobertaTokenizer, TFXLMRobertaModel,AutoTokenizer
import numpy as np
from collections import Counter
from datasets import Dataset
from sklearn.metrics import recall_score,precision_score,f1_score,classification_report


#0 读取测试集合
def getTFDataset(config,tokenzier):
    testds = Dataset.load_from_disk(config["testpath"])
    print(testds)
    def tokenize_fun(example):
        return tokenzier(example["textfeatures"], truncation=True, padding="max_length", max_length=500)
    test_tokenized_ds = testds.map(tokenize_fun, batched=True)
    tf_test_dataset = test_tokenized_ds.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["labels"],
        shuffle=False,
        # collate_fn=data_collator,
        batch_size=12,
    )
    return testds,tf_test_dataset

def get_orlabel(labellist):
    labels = list(reversed(labellist))
    # 因为使用的后缀根据标签添加的，所以每条数据将会产生12条数据
    # 每次取bathcsize条数据，就能得到原始标签
    reallabels = []
    temp = []
    while len(labels) > 0:
        label = labels.pop()
        temp.append(label)
        if len(temp) == 12:
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
    # print("getpredic_label")
    # print(predicts)
    # print(predicts.shape)
    labellist=[]
    templist=[]
    for i in range(len(predicts)):
        templist.append(predicts[i])
        if len(templist)==12:
            prelabel=compare_lable(templist)
            labellist.append(prelabel[0])
            templist.clear()
        else:
            pass
    return labellist

def getPredictions(config):
    tokenzier = AutoTokenizer.from_pretrained(config["checkpoint"])
    test_dataset_reloaded, tf_test_dataset = getTFDataset(config=config, tokenzier=tokenzier)
    basemodelfilepath = config["savemodelname"]
    basemodel = tf.keras.models.load_model(basemodelfilepath)
    print("predict")
    # predics = basemodel.predict(tf_test_dataset.take(3))
    predics=basemodel.predict(tf_test_dataset)

    # 实际标签
    # rellabels = get_orlabel(test_dataset_reloaded["labels"][:36])
    rellabels=get_orlabel(test_dataset_reloaded["labels"])
    return predics,rellabels

def getReport(rellabels,pre_label,save):
    matrix = tf.keras.metrics.Accuracy()
    matrix.update_state(rellabels, pre_label)
    acc = matrix.result().numpy()
    print("测试ACC")
    print(acc)
    """

    """
    Precision = precision_score(rellabels, pre_label, average='weighted')
    print("测试Precision")
    print(Precision)
    recall = recall_score(rellabels, pre_label, average='weighted')
    print("测试Recall")
    print(recall)
    f1 = f1_score(rellabels, pre_label, average='weighted')
    print("测试F1")
    print(f1)
    # 返回模型的输出，用于集成学习计算概率。
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    print("report")
    print(classification_report(rellabels, pre_label, labels=labels))
    with open(file=save, encoding="UTF-8", mode='w') as f:
        f.write(str(float(acc)) + "测试ACC ")
        f.write("\n")
        f.write(str(float(Precision)) + "测试Precision ")
        f.write("\n")
        f.write(str(float(recall)) + " 测试Recall")
        f.write("\n")
        f.write(str(float(f1)) + " 测试F1")
        f.write("\n")
        f.close()


def getTarget(predics,rellabels,file):
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
    getReport(rellabels,pre_label,save=file)




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