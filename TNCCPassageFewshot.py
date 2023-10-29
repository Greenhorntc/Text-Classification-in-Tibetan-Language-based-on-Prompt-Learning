"""
全量数据训练
few-shot 的实验内容
在创建好的数据集上 依次采用每个类别的数据i=10 ，i=20，i=30，i=40,i=50
prompt1 2 3 cino-base
prompt1 2 3 tibert
"""
import os
from TNCCPassageTrain import train,savehistory
from GetModel import TextCNN,RNN,TextRCNN,DPCNN
from transformers import TFXLMRobertaModel,AutoModel
from ResultPlot import plotLoss,plotACC
from matplotlib import pyplot as plt
from ComparativeExperiments import ctrain

#1 基于alldata  loaddataset


#修改config
def getFewshotconfig(floder,few):
    datasetpath = floder + "/" + few
    train = datasetpath + "/train"
    val = datasetpath + "/val"
    test = datasetpath + "/test"
    savehisname='logs/TNCCpassage/PromptFewshot/'+few+".txt"
    savemodel = "savedmodel/TNCC/TNCCpassage/PromptFewshot/" +few
    config = {"batch_size": 8, "epochs": 50, "checkpoint": "hfl/cino-small-v2", "trainpath": train, "testpath": test,
              "valpath": val, "savemodelname": savemodel,"savehis":savehisname}
    return config

# 根据config 训练对应的模型
def fewshotPrompt(floder):
    congfiglist=[]
    datafiles=os.listdir(floder)
    for i in range(len(datafiles)):
        #i i+1 i+2
        config=getFewshotconfig(floder=floder,few=datafiles[i])
        # print(config)
        congfiglist.append(config)
    return congfiglist


def getNopromptConfig(floder,file):
    dspath=floder+"/"+file
    train = dspath + "/train"
    val = dspath + "/val"
    test = dspath + "/test"
    savehisname = 'logs/TNCCpassage/Nopromptfewshot/'+file
    savemodel = "savedmodel/TNCC/TNCCpassage/Fewshot/" + file
    config = {"batch_size": 8, "epochs": 50, "checkpoint": "hfl/cino-small-v2", "trainpath": train, "testpath": test,
              "valpath": val, "savemodelname": savemodel, "savehis": savehisname}
    return config

def fewShotNoproTrain(floder,cinomodel):
    nofiles = os.listdir(floder)
    for i in range(len(nofiles)):
        config=getNopromptConfig(floder,nofiles[i])
        textcnn=TextCNN(cinomodel,dropout=0.4)
        ctrain(cmodel=textcnn,name="textCNN",config=config)
        rnn=RNN(cinomodel,dropout=0.2)
        ctrain(cmodel=rnn,name="RNN",config=config)
        rcnn=TextRCNN(cinomodel,dropout=0.4)
        ctrain(cmodel=rcnn,name="RCNN",config=config)
        dpcnn=DPCNN(cinomodel,dropout=0.2)
        ctrain(cmodel=dpcnn,name="DPCNN",config=config)
    print("No prompt over")

"""
#1prompt 全量数据训练
"""
# config1={"batch_size":8,"epochs":50,"checkpoint":"hfl/cino-small-v2","trainpath":"dataset/TNCC/TNCCpassage/edition1prompt0/train","testpath":"dataset/TNCC/TNCCpassage/edition1prompt0/test","valpath":"dataset/TNCC/TNCCpassage/edition1prompt0/val","savemodelname":"savedmodel/TNCC/TNCCpassage/editionprompt0-small"}
# config2={"batch_size":8,"epochs":50,"checkpoint":"hfl/cino-small-v2","trainpath":"dataset/TNCC/TNCCpassage/edition1prompt1/train","testpath":"dataset/TNCC/TNCCpassage/edition1prompt1/test","valpath":"dataset/TNCC/TNCCpassage/edition1prompt1/val","savemodelname":"savedmodel/TNCC/TNCCpassage/editionprompt1-small"}
# config3={"batch_size":8,"epochs":50,"checkpoint":"hfl/cino-small-v2","trainpath":"dataset/TNCC/TNCCpassage/edition1prompt2/train","testpath":"dataset/TNCC/TNCCpassage/edition1prompt2/test","valpath":"dataset/TNCC/TNCCpassage/edition1prompt2/val","savemodelname":"savedmodel/TNCC/TNCCpassage/editionprompt2-small"}
#
# config4={"batch_size":8,"epochs":50,"checkpoint":"sangjeedondrub/tibetan-roberta-base","trainpath":"dataset/TNCC/TNCCpassage/edition1prompt0/train","testpath":"dataset/TNCC/TNCCpassage/edition1prompt0/test","valpath":"dataset/TNCC/TNCCpassage/edition1prompt0/val","savemodelname":"savedmodel/TNCC/TNCCpassage/editionprompt0-tibert"}
# config5={"batch_size":8,"epochs":50,"checkpoint":"sangjeedondrub/tibetan-roberta-base","trainpath":"dataset/TNCC/TNCCpassage/edition1prompt1/train","testpath":"dataset/TNCC/TNCCpassage/edition1prompt1/test","valpath":"dataset/TNCC/TNCCpassage/edition1prompt1/val","savemodelname":"savedmodel/TNCC/TNCCpassage/editionprompt1-tibert"}
# config6={"batch_size":8,"epochs":50,"checkpoint":"sangjeedondrub/tibetan-roberta-base","trainpath":"dataset/TNCC/TNCCpassage/edition1prompt2/train","testpath":"dataset/TNCC/TNCCpassage/edition1prompt2/test","valpath":"dataset/TNCC/TNCCpassage/edition1prompt2/val","savemodelname":"savedmodel/TNCC/TNCCpassage/editionprompt2-tibert"}
#
# print("training")
# history1=train(config=config4)
# savehistory(history1,filename="logs/TNCCpassage/logs4.txt")
# history2=train(config=config5)
# savehistory(history2,filename="logs/TNCCpassage/logs5.txt")
# history3=train(config=config6)
# savehistory(history3,filename="logs/TNCCpassage/logs6.txt")

# modelhistorylist = [history1,history2,history3]
# namelist = ["Using-prompt1-base1","Using-Prompt2-base2","Using-Prompt3-base3"]
# namelist = ["Using-prompt1-tibert1","Using-Prompt2-tibert2","Using-Prompt3-tibert3"]
# #plot
# plotLoss(modelhistorylist, namelist=namelist)
# plt.show()
# plotACC(modelhistorylist, namelist=namelist)
# plt.show()

"""
# 2fewshot prompt
"""
fewshottnccPassagefloder='dataset/TNCC/TNCCpassage/FewShotPrompt'
#configlist0 -15 3个一组
configlist=fewshotPrompt(fewshottnccPassagefloder)
print(configlist[1])
his=train(configlist[1])
savehistory(his,filename=configlist[1]["savehis"])
print("=================================")


"""
#3全量数据对比实验 rcnn text cnn rnn

"""
# config={
#     "batch_size":8,
#     "epochs":50,
#     "checkpoint":"hfl/cino-small-v2",
#     "trainpath":"dataset/TNCC/TNCCpassage/Noprompt/final/train",
#     "valpath":"dataset/TNCC/TNCCpassage/Noprompt/final/val",
#     "testpath":"dataset/TNCC/TNCCpassage/Noprompt/final/test",
#     "savemodelname":"savedmodel/TNCC/TNCCpassage/",
#     "savehis":"logs/TNCCpassage/noprompt"
#
# }
# cinomodel = TFXLMRobertaModel.from_pretrained(config["checkpoint"])

"""
#训练模型
"""
# config["checkpoint"]="sangjeedondrub/tibetan-roberta-base"
# print(config)
# clsmodel=TFXLMRobertaModel.from_pretrained(config["checkpoint"],from_pt=True)
# # clsmodel=TFXLMRobertaModel.from_pretrained(config["checkpoint"],from_pt=True)
# trbout=Lmdicout(clsmodel,dropout=0.1)
# ctrain(trbout,name="TRB",config=config,fintued=False)

# cino=Lmdicout(cinomodel,dropout=0.1)
# ctrain(cmodel=cino,name="Cino",config=config)


# textcnn=TextCNN(cinomodel,dropout=0.4)
# ctrain(cmodel=textcnn,name="textCNN",config=config)
# rnn=RNN(cinomodel,dropout=0.2)
# ctrain(cmodel=rnn,name="RNN",config=config)
# rcnn=TextRCNN(cinomodel,dropout=0.4)
# ctrain(cmodel=rcnn,name="RCNN",config=config)
# dpcnn=DPCNN(cinomodel,dropout=0.2)
# ctrain(cmodel=dpcnn,name="DPCNN",config=config)

"""
# 4 fewshot no prompt 对比实现
"""
# Noprompt='dataset/TNCC/TNCCpassage/FewShotNOPrompt'
# fewShotNoproTrain(floder=Noprompt,cinomodel=cinomodel)


