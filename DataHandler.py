#tc
#使用复旦的 TNCC数据集
#tncc的数据集包含两类 标题 Tourism	བྲག ཡེར པ 和 标题 文章 /t +passage 藏文的分隔符。被替换成了空格

# TC
#主要是用来构建数据集合
from datasets import Dataset,interleave_datasets
import random
class Data_handler():
    def __init__(self,promptfile,datafile,label2id):
        self.promptfile=promptfile
        self.datafile=datafile
        self.labeltoid=label2id

    def get_suffix(self,promptfile):
        lineslist = []
        with open(file=promptfile, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                lineslist.append(line.strip("\n"))
                # text=text.split('\u0F0B')
        return lineslist

    def getSelectd_suffix(self,file,num):
        linelist=self.get_suffix(file)
        suffixlist=[]
        for i in num:
            suffixlist.append(linelist[i])
        return suffixlist

    def load_data(self):
        D = []
        with open(self.datafile, encoding='utf-8') as f:
            for l in f:
                label, txt = l.strip().split('\t')
                label=self.labeltoid[label]
                #将空格换 停用词汇
                txt = ''.join([i.strip(' ') for i in txt])
                # print(txt)
                # \u0F0B
                # title = txt.split(" ")
                # tmp=""
                # for i in title:
                #     tmp=tmp+i
                D.append((txt,label))
        return D

    def select_data(self,datalist,n):
        chosendatalist=[]
        unchosendatalist=[]
        #从0-数据集里面个数随机采样n个数据
        chosenindex=random.sample(range(0,len(datalist)),n)
        #chosenindex 是np array 不是普通的list元素 in 这样不一定能找到的
        for i in range(len(datalist)):
            if i in chosenindex:
                chosendatalist.append(datalist[i])
            else:
                unchosendatalist.append(datalist[i])

        return chosendatalist,unchosendatalist

    def delete_key(self,dic,n):
        delkey=[]
        for key in dic:
            datalen = len(dic[key])
            if datalen <= n:
               delkey.append(key)
        for key in delkey:
            print(key)
            dic.pop(key)
        if len(delkey)==0:
            print("在取样过程中，不存在类别过少的数据，未删除")
        else:
            print("在取样过程中，存在类别过少的数据")
        return dic
    #从每个标签中抽样n条数据
    def sample_from_data(self,Datalist,sampledic,valdic):
        #使用dic来对已有的数据进行每个类进行标记
        datadic={}
        for data in Datalist:
            if data[1] not in datadic.keys():
                datadic[data[1]]=[data[0]]
            else:
                datadic[data[1]].append(data[0])
        # datadic=self.delete_key(datadic,n=n)
        #获得datadic后，创建traindic和testdic
        train = {}
        test = {}

        #如果改动每个类应当提取多少数据的时候
        for key in datadic:
            # print(key)
            # print(sampledic[key])
            sampledlist,unsampledlist=self.select_data(datadic[key],sampledic[key])
            train[key]=sampledlist
            test[key]=unsampledlist
        #再取40验证集合 剩下来做测试
        validationdata={}
        newtest={}
        for key in test:
            vallist,newtestlist=self.select_data(test[key],valdic[key])
            validationdata[key] = vallist
            newtest[key] = newtestlist

        return train,validationdata,newtest

    def get_fewshot(self,trainic,number):
        train = {}
        # 如果改动每个类应当提取多少数据的时候
        for key in trainic:
            # print(key)
            # print(sampledic[key])
            sampledlist, unsampledlist = self.select_data(trainic[key], n=number)
            train[key] = sampledlist
        return train

    def create_example(self,txt, label,promptfile):
        example = []
        #记录一下 选取的label和

        labelselected = [0,1,2,3,4,5,6,7,8,9,10,11]
        #取第一行到13行，第13行才是对应11标签的数据
        num=range(1,13)
        suffixs=self.getSelectd_suffix(file=promptfile,num=num)
        for i in range(len(suffixs)):
            # print(i)
            #采取截断输出400维度，直接单独存一个特征列
            index = labelselected.index(label)
            # print(index)
            if i == index:
                example.append((txt[:400],suffixs[i], 1,label))
            else:
                example.append((txt[:400],suffixs[i], 0,label))
        return example

    def datadic_2_ds(self,datadic,prompt):
        newfeatures = []
        featuresprompt=[]
        newlabels = []
        orlabels=[]
        for key in datadic:
            for i in range(len(datadic[key])):
                examples = self.create_example(datadic[key][i], key,prompt)
                for example in examples:
                    newfeatures.append(example[0])
                    featuresprompt.append(example[1])
                    newlabels.append(example[2])
                    orlabels.append(example[3])

        dic = {"textfeatures": newfeatures,"textprompt":featuresprompt, "labels": newlabels,"orlabels":orlabels}
        newds = Dataset.from_dict(dic)
        return newds

    #datadic直接转为普通数据集
    def nopromptdatatods(self,datadic):
        newfeatures = []
        newlabels = []
        for key in datadic:
            for i in range(len(datadic[key])):
                newfeatures.append(datadic[key][i])
                newlabels.append(key)
        dic = {"textfeatures": newfeatures, "labels": newlabels}
        newds = Dataset.from_dict(dic)
        return newds

    def getNopromptsds(self,ds1,ds2,ds3,name):
        trainds = self.nopromptdatatods(ds1)
        valds = self.nopromptdatatods(ds2)
        testds = self.nopromptdatatods(ds3)
        self.data_to_disk(trainds, valds, testds, name=name)

    def getAllds(self, ds1,ds2,ds3,name):
        i = 0
        for suffix in self.promptfile:
            #使用过采样对训练的样本进行平衡。neg为60% pos40%
            trainds = self.datadic_2_ds(ds1, prompt=suffix)
            valds=self.datadic_2_ds(ds2,prompt=suffix)
            testds = self.datadic_2_ds(ds3, prompt=suffix)
            oversamplingds=self.oversampling(trainds).shuffle(seed=41)
            oversamplingvalds=self.oversampling(valds).shuffle(seed=41)
            # return oversamplingds,oversamplingvalds,testds
            self.data_to_disk(oversamplingds,oversamplingvalds,testds,name=name + str(i))
            i = i + 1

    def data_to_disk(self,ds1,ds2,ds3,name):
        print(".....Data Save.......")
        path=name
        print(ds1)
        print(ds2)
        print(ds3)
        ds1.save_to_disk(path+"/train")
        ds2.save_to_disk(path+"/val")
        ds3.save_to_disk(path+"/test")
        print(".....Data has been Saved.......")

    def oversampling(self,ds):
        neg = ds.filter(lambda example: example["labels"]==0)
        pos = ds.filter(lambda example: example["labels"]==1)
        oversamplingds = interleave_datasets([neg,pos], probabilities=[0.6,0.4], seed=42,stopping_strategy="all_exhausted")

        return oversamplingds



