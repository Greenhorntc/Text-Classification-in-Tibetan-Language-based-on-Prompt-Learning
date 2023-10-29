# TC
#主要是用来构建数据集合
from datasets import load_dataset,Dataset,interleave_datasets
import random
class Data_handler():
    def __init__(self,promptfile,datafile):
        self.promptfile=promptfile
        self.datafile=datafile

    def get_suffix(self,file):
        lineslist = []
        with open(file=file, encoding="utf-8") as f:
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
                text, label = l.strip().split('\t')
                D.append((text, int(label)))
        return D

    def select_data(self,datalist,n):
        chosendatalist=[]
        unchosendatalist=[]
        #从0-数据集里面个数随机采样n个数据
        chosenindex=random.sample(range(0,len(datalist)),n)
        #chosenindex 是np array
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
        print("在取样过程中，存在类别过少的数据")
        return dic
    #从每个标签中抽样n条数据
    def sample_from_data(self,Datalist,n=50):
        #使用dic来对已有的数据进行每个类进行标记
        datadic={}
        for data in Datalist:
            if data[1] not in datadic.keys():
                datadic[data[1]]=[data[0]]
            else:
                datadic[data[1]].append(data[0])
        datadic=self.delete_key(datadic,n=n)
        #获得datadic后，创建traindic和testdic
        sampleddatadic = {}
        unsampleddatadic = {}
        for key in datadic:
            sampledlist,unsampledlist=self.select_data(datadic[key],n)
            sampleddatadic[key]=sampledlist
            unsampleddatadic[key]=unsampledlist

        return sampleddatadic,unsampleddatadic

    def create_example(self,txt, label,promptfile):
        example = []
        #记录一下 选取的label和
        labelselected = [0,1,2,4,5,6,9]
        suffixs=self.getSelectd_suffix(file=promptfile,num=[1,2,3,5,6,7,10])
        for i in range(len(suffixs)):
            # print(i)
            sententce = txt + suffixs[i]
            index = labelselected.index(label)
            # print(index)
            if i == index:
                example.append((sententce, 1))
            else:
                example.append((sententce, 0))
        return example

    def datadic_2_ds(self,datadic,prompt):
        newfeatures = []
        newlabels = []
        for key in datadic:
            for i in range(len(datadic[key])):
                examples = self.create_example(datadic[key][i], key,prompt)
                for example in examples:
                    newfeatures.append(example[0])
                    newlabels.append(example[1])

        dic = {"textfeatures": newfeatures, "labels": newlabels}
        newds = Dataset.from_dict(dic)
        return newds

    def getAllds(self, ds1,ds2):
        i = 0
        for suffix in self.promptfile:
            trainds = self.datadic_2_ds(ds1, prompt=suffix)
            testds = self.datadic_2_ds(ds2, prompt=suffix)
            oversamplingds = self.oversampling(trainds).shuffle(seed=41)
            self.data_to_disk(oversamplingds, testds, name="prompt" + str(i))
            i = i + 1

    def data_to_disk(self,ds1,ds2,name):
        print(".....Data Save.......")
        path="dataset/WCM/finaledition"+name
        print(ds1)
        print(ds2)
        ds1.save_to_disk(path+"/wcmtrain")
        ds2.save_to_disk(path+"/wcmtest")
        print(".....Data has been Saved.......")

    def oversampling(self,ds):
        neg = ds.filter(lambda example: example["labels"]==0)
        pos = ds.filter(lambda example: example["labels"]==1)
        oversamplingds = interleave_datasets([neg,pos], probabilities=[0.5,0.5], seed=42,stopping_strategy="all_exhausted")

        return oversamplingds

#1提示文件目前实现了3个不同的提示
suffixfile="dataset/WCM/prompt.txt"
suffixfile2="dataset/WCM/prompt2.txt"
suffixfile3="dataset/WCM/prompt3.txt"
suffixfilelist=[suffixfile,suffixfile2,suffixfile3]
#2数据集
datasetfile="dataset/WCM/minority/bo.txt"
#类
datahandler=Data_handler(promptfile=suffixfilelist,datafile=datasetfile)

#选取所在后缀的行
suffixlist=datahandler.getSelectd_suffix(file=suffixfilelist[0],num=[1,2,3,5,6,7,10])
suffixlist2=datahandler.getSelectd_suffix(file=suffixfilelist[1],num=[1,2,3,5,6,7,10])
suffixlist3=datahandler.getSelectd_suffix(file=suffixfilelist[2],num=[1,2,3,5,6,7,10])
print(suffixlist)
print(suffixlist2)
print(suffixlist3)
#原始数据
datalist=datahandler.load_data()
# print(datalist)
#fewshot使用50条数据作为记录
traindic,testdic=datahandler.sample_from_data(datalist,n=50)

# trainall=0
# for item in traindic.items():
#     trainall=trainall+item[1]
# print(str(trainall)+"条训练数据被使用占数据集" +str(trainall/1110))

testall=0
for key in testdic.keys():
    print(key)
    # print(testdic(key))
    # print(len(testdic(key)))
    testall=testall+len(testdic[key])
print(str(testall)+"测试集使用占数据集" +str(testall/1110))

datahandler.getAllds(ds1=traindic,ds2=testdic)
print("over")




