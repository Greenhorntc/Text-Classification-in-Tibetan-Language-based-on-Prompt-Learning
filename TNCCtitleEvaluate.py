#tncc title 评估
from TNCCEvaluate import getPredictions,getTarget,ensamble
config= {"checkpoint":"hfl/cino-small-v2","testpath":"dataset/TNCC/final3prompt0/test","savemodelname":"savedmodel/TNCC/final3prompt0-small"}
config1={"checkpoint":"hfl/cino-small-v2","testpath":"dataset/TNCC/final3prompt1/test","savemodelname":"savedmodel/TNCC/final3prompt1-small"}
config2={"checkpoint":"hfl/cino-small-v2","testpath":"dataset/TNCC/final3prompt2/test","savemodelname":"savedmodel/TNCC/final3prompt2-small"}

# config3={"checkpoint":"hfl/cino-base-v2","testpath":"dataset/TNCC/prompt0/test","savemodelname":"savedmodel/TNCC/prompt0-base"}
# config4={"checkpoint":"hfl/cino-base-v2","testpath":"dataset/TNCC/prompt1/test","savemodelname":"savedmodel/TNCC/prompt1-base"}
# config5={"checkpoint":"hfl/cino-base-v2","testpath":"dataset/TNCC/prompt2/test","savemodelname":"savedmodel/TNCC/prompt2-base"}
#
# config6={"checkpoint":"hfl/cino-large-v2","testpath":"dataset/TNCC/prompt0/test","savemodelname":"savedmodel/TNCC/prompt0-large"}
# config7={"checkpoint":"hfl/cino-large-v2","testpath":"dataset/TNCC/prompt1/test","savemodelname":"savedmodel/TNCC/prompt1-large"}
# config8={"checkpoint":"hfl/cino-large-v2","testpath":"dataset/TNCC/prompt2/test","savemodelname":"savedmodel/TNCC/prompt2-large"}

predics0,rellabels0=getPredictions(config)
getTarget(predics0,rellabels0)
predics1,rellabels1=getPredictions(config1)
getTarget(predics1,rellabels1)
predics2,rellabels2=getPredictions(config2)
getTarget(predics2,rellabels2)
#得到集成学习的结果后，进行预测评价
resultlist=[predics0,predics1,predics2]
print("使用集成学习")
ensambleresult1=ensamble(resultlist,model="avg")
ensambleresult2=ensamble(resultlist,model="max-min")
#评估
print("avg")
getTarget(ensambleresult1,rellabels1)
print("max")
getTarget(ensambleresult2,rellabels1)