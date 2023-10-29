#tc
#使用复旦的 TNCC数据集
#tncc的数据集包含两类 标题 Tourism	བྲག ཡེར པ 和 标题 文章 /t +passage 藏文的分隔符。被替换成了空格
from DataHandler import Data_handler
#1提示文件目前实现了3个不同的提示,且全部加到了句末
suffixfile0="dataset/TNCC/TNCCprompt0.txt"
suffixfile1="dataset/TNCC/TNCCprompt1.txt"
suffixfile2="dataset/TNCC/TNCCprompt2.txt"
suffixfilelist=[suffixfile0,suffixfile1,suffixfile2]
#2 TNCC存在两个数据集 标题 和 文章内容
titlefile="dataset/TNCC/title-tibetan.txt"
passagefile="dataset/TNCC/document-tibetan.txt"
#对应的分类标签和数值
labeldic={'Politics': 0, 'Education': 1, 'Economics': 2, 'Environment': 3, 'Instruments': 4, 'Religion': 5, 'Medicine': 6, 'Tourism': 7, 'Arts': 8, 'Customs': 9, 'Literature': 10, 'Language': 11}

datahandler=Data_handler(promptfile=suffixfilelist,datafile=titlefile,label2id=labeldic)

# datahandler=Data_handler(promptfile=suffixfilelist,datafile=passagefile,label2id=labeldic)

#TNCC的数据集合每个类别的数据都很全，不用去掉用不到的提示，但是还是需要排除第一条 中文提示
#所以文件一共有13条数据，去掉第一条
# num=range(0,13)
# suffixlist=datahandler.getSelectd_suffix(file=suffixfilelist[0],num=num[1:])
# suffixlist2=datahandler.getSelectd_suffix(file=suffixfilelist[1],num=num[1:])
# suffixlist3=datahandler.getSelectd_suffix(file=suffixfilelist[2],num=num[1:])
# print(suffixlist)
# print(suffixlist2)
# print(suffixlist3)

#原始数据
datalist=datahandler.load_data()
print(datalist)

#fewshot使用n条数据作为记录,n使用的训练集合 数量，i 验证集数量,
# Counter({'0': 2132, '1': 1370, '2': 986, '3': 953, '4': 842, '5': 670, '6': 520, '7': 512, '8': 502, '9': 275, 'Literature': 259, 'Language': 255})
traindic = {0:1700,1:900,2:800,3:800,4:700, 5:550,6:410,7:410,8:420,9:220,10:200,11:200}
valdic=    {0:100, 1:100, 2:100,3:100,4:100,5:90,6:80,7:70,  8:60,9:20,10:20,11:20}


trainall=0
for item in traindic.items():
    trainall=trainall+item[1]
print(str(trainall)+"条训练数据被使用占数据集" +str(trainall/9276))

valall=0
for item in valdic.items():
    valall=valall+item[1]
print(str(valall)+"条训练数据被使用占数据集" +str(valall/9276))



traindic,valdic,testdic=datahandler.sample_from_data(datalist,sampledic=traindic,valdic=valdic)
titllesavedpath="dataset/TNCC/TNCCtitle/finalenditon2"
datahandler.getAllds(ds1=traindic,ds2=valdic,ds3=testdic,name=titllesavedpath)
print("title over")


