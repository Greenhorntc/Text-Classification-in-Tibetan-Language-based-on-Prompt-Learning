import pandas as pd
from sklearn.metrics import classification_report,accuracy_score

labeldic={'Politics': 0, 'Education': 1, 'Economics': 2, 'Environment': 3, 'Instruments': 4, 'Religion': 5, 'Medicine': 6, 'Tourism': 7, 'Arts': 8, 'Customs': 9, 'Literature': 10, 'Language': 11,'Culture':12}

datafile='result.xlsx'

df=pd.read_excel(datafile,sheet_name='all')
print(df.head(2))
rellabls=df['label']
gpt=df['chatgpt-3.5-turbo']
glm=df['chatglm']
lama=df['Llama-30b']


def getlabnls(datacoloumns):
    predicts=[]
    for predict in datacoloumns:
        # print(index)
        # print(predict)
        predicts.append(labeldic[predict])
    return predicts
y=getlabnls(rellabls)
gptpre=getlabnls(gpt)
glmpre=getlabnls(glm)
lamapre=getlabnls(lama)
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12]

print("gpt")
acc = accuracy_score(y, gptpre)
print(acc)
print(classification_report(y, gptpre, labels=labels,digits=5))

print("glm")
print(classification_report(y, glmpre, labels=labels,digits=5))

print("lama")
print(classification_report(y, lamapre, labels=labels,digits=5))