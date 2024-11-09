import tensorflow as tf
from transformers import XLMRobertaTokenizer, TFXLMRobertaModel,AutoTokenizer
import numpy as np
from collections import Counter
from datasets import Dataset
from sklearn.metrics import recall_score,precision_score,f1_score,classification_report

"""
There is an example to illustrate the entire calculation process. Given a piece of content: "The tacos in Mexico are delicious...", suppose the dataset has three classification categories: Politics, Life, and Art. The proposed method will generate three sentences as follows:
index1: "The tacos in Mexico are delicious...; this is a piece of political news."
index2: "The tacos in Mexico are delicious...; this is a piece of life news."
index3: "The tacos in Mexico are delicious...; this is a piece of art news."
The model predicts the probability distribution for each sentence and its corresponding prompt, ranging from [0,1] to indicate the likelihood of an entailment relationship. Assume the output predictions for each sentence are as follows:index1: [0.8, 0.2],index2: [0.2, 0.8],index3: [0.4, 0.6].
For each individual prompt, the class with the highest confidence is determined as follows:
Use argmax[] to identify the index of the highest probability in each prediction array.
For index1, the highest probability index is 0 (indicating Politics).
For index2, the highest probability index is 1 (indicating Life).
For index3, the highest probability index is 1 (indicating Art).
The results thus classify the content under both Lifestyle and Art. If multiple samples are classified as category 1 (i.e., the highest probability corresponds to index 1), the sample with the highest probability for this category is chosen as the true label. Any other samples predicted as category 1 will be reclassified as not being in category 1.If no sample is classified as category 1, the sample with the highest probability is selected and labeled as category 1.
Given the assumed output, both Life and Art are predicted as entailment categories. However, the probability of 0.8 for Life in index2 exceeds the 0.6 for Art in index3, leading to the final classification as Life. When integrating multiple prompts, methods like averaging or max-min are applied to the probability distribution of each prompt for weighted calculations on this basis.

"""
#0 read testset
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
    # Since the suffixes are added according to the labels, each piece of data will generate 12 pieces of data.
    #By taking batchsize pieces of data each time, the original labels can be obtained.
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
    #To denote the indices and probabilities corresponding to the label '1' in this set of predictions:
    for i in range(len(labellist)):
        index=np.argmax(labellist[i])
        if index==1:
            temp_index.append(i)
            temp_pos.append(labellist[i][1])

    #To determine which of the multiple predictions for the label '1' has the smaller probability and then change the index of the one with the smaller probability (i.e., the "tempclass") to 0, 
    # print(temp_index)
    # print(temp_pos)
    if len(temp_pos) > 1:
        maxvalue=max(temp_pos)
        idx = temp_pos.index(maxvalue)
        index=temp_index[idx]
        #After obtaining the maximum index, all '1's at other positions should be set to '0'.
        for i in range(len(temp_class_pre)):
            if temp_class_pre[i]==1:
                if i !=index:
                    temp_class_pre[i]=0

    elif len(temp_pos)==0:
        #Another scenario is when there are no '1's in temp.
        temp_max_1=[]
        for i in  range(len(labellist)):
            temp_max_1.append(labellist[i][1])
        maxtemp=max(temp_max_1)
        maxidx=temp_max_1.index(maxtemp)
        #After obtaining the maximum index, set the corresponding index in the original prediction labellist to '1'.
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
    print("test Precision")
    print(Precision)
    recall = recall_score(rellabels, pre_label, average='weighted')
    print("testRecall")
    print(recall)
    f1 = f1_score(rellabels, pre_label, average='weighted')
    print("testF1")
    print(f1)
   
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    print("report")
    print(classification_report(rellabels, pre_label, labels=labels))
    with open(file=save, encoding="UTF-8", mode='w') as f:
        f.write(str(float(acc)) + "testACC ")
        f.write("\n")
        f.write(str(float(Precision)) + "testPrecision ")
        f.write("\n")
        f.write(str(float(recall)) + " testRecall")
        f.write("\n")
        f.write(str(float(f1)) + " testF1")
        f.write("\n")
        f.close()


def getTarget(predics,rellabels,file):
    # predics,rellabels=getPredictions(config)
    # [0,1,2,4,5,6,9]

    result = Counter(rellabels)
    print("Count the number of labels.")
    print(result)
    print("The labels of the test set after conversion.")
    print(rellabels)
    pre_label = getpredic_label(predics)
    print("Prediction of model")
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
 
    if model=="avg":
        pre=AddResultArray(prelist)
        print(pre.shape)
        return pre
    if model=="max-min":
        pre=getNegPos(prelist)
        print(pre.shape)
        return pre
    else:
        print("get wrong input")
        pass
