from DataHandler import Data_handler
from collections import Counter
#1提示文件目前实现了3个不同的提示,且全部加到了句末
suffixfile0="dataset/TNCC/TNCCprompt0.txt"
suffixfile1="dataset/TNCC/TNCCprompt1.txt"
suffixfile2="dataset/TNCC/TNCCprompt2.txt"
suffixfilelist=[suffixfile0,suffixfile1,suffixfile2]
#2 TNCC存在两个数据集 标题 和 文章内容
passagefile="dataset/TNCC/document-tibetan.txt"
#对应的分类标签和数值
labeldic={'Politics': 0, 'Education': 1, 'Economics': 2, 'Environment': 3, 'Instruments': 4, 'Religion': 5, 'Medicine': 6, 'Tourism': 7, 'Arts': 8, 'Customs': 9, 'Literature': 10, 'Language': 11}
datahandler=Data_handler(promptfile=suffixfilelist,datafile=passagefile,label2id=labeldic)

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
#统计标签
# lablelist=[x[1] for x in datalist]
# print(lablelist)
# print(Counter(lablelist))

#全量训练使用n条数据作为记录,n使用的训练集合 数量，i 验证集数量
# Counter( {0: 2117, 1: 1359, 2: 983, 3: 945, 4: 840, 5: 665, 6: 519, 7: 510, 8: 492, 9: 272, 10: 258, 11: 244})
traindic = {0:1700,  1:900,   2:800,  3:800,  4:700,  5:550,  6:410,  7:410,  8:420,  9:220,  10:200,  11:200}
valdic=    {0:100,   1:100,   2:100,  3:100,  4:100,  5:90,   6:80,   7:70,   8:60,   9:20,   10:20,   11:20}
#
# trainall=0
# for item in traindic.items():
#     trainall=trainall+item[1]
# print(str(trainall)+"条训练数据被使用占数据集" +str(trainall/9240))
#
# valall=0
# for item in valdic.items():
#     valall=valall+item[1]
# print(str(valall)+"条训练数据被使用占数据集" +str(valall/9240))
traindic,valdic,testdic=datahandler.sample_from_data(datalist,sampledic=traindic,valdic=valdic)

# #创建带提示的数据集，1——》3
# passagesavedpath="dataset/TNCC/TNCCpassage/Prompt/promptfinal"
# datahandler.getAllds(ds1         =traindic,ds2=valdic,ds3=testdic,name=passagesavedpath)
# print("Prompt dataset over")

#text label类型数据集，对比实验使用
# nopromptpath="dataset/TNCC/TNCCpassage/Noprompt/final"
# datahandler.getNopromptsds(ds1=traindic,ds2=valdic,ds3=testdic,name=nopromptpath)
# print("No-Prompt dataset over")

#few-shot prompt 数据集 仅仅是traindic的基础上进行切割
fewshot=[10,20,30,40,50]
for few in fewshot:
    fewdic=datahandler.get_fewshot(traindic,few)
    fewpath = "dataset/TNCC/TNCCpassage/FewShotPrompt/"+str(few)+"fewprompt"
    print(fewpath)
    datahandler.getAllds(ds1=fewdic, ds2=valdic, ds3=testdic, name=fewpath)
print("Few-shot Prompt  dataset over")

#textcnn rnn rcnn fewshot
# for few in fewshot:
#     fewdic=datahandler.get_fewshot(traindic,few)
#     fewpath = "dataset/TNCC/TNCCpassage/FewShotNOPrompt/fewshot"+str(few)
#     print(fewpath)
#     datahandler.getNopromptsds(ds1=fewdic, ds2=valdic, ds3=testdic, name=fewpath)
# print("Few-shot  Noprompt  dataset over")



