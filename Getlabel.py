def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D

#统计各个标签数量
train_data = load_data('dataset/WCM/minority/bo.txt')
print(len(train_data))
labeldic={}
for data in train_data:
    if data[1] not in labeldic:
        #当出现第一条记录时应记作1
        labeldic[data[1]]=1
    else:
        labeldic[data[1]]=labeldic[data[1]]+1
print(labeldic)


{1: 339, 2: 111, 6: 163, 4: 133, 0: 141, 5: 111, 9: 111, 7: 1}
{"艺术":0,"地理":1,"历史":2,"自然":3,"自然科学":4,"人物":5,"技术":6,"教育":7,"经济":8,"健康":9}

testprefix="འདི་ནི་xxx་གསར་འགྱུར་ཞིག་རེད།"
label1="འདི་ནི་སྒྱུ་རྩལ། ་གསར་འགྱུར་ཞིག་རེད།"
label=["","",]
"""
艺术	141
地理	339
历史	111
自然	0
自然科学	133
人物	111
技术	163
教育	1
经济	0
健康	111
"""
"""
སྒྱུ་རྩལ། 
ས་རྒྱུས། 
ལོ་རྒྱུས། 
རང་བྱུང། 
རང་བྱུང་ཚན་རིག། 
མི་སྣ། 
ལག་རྩལ། 
སློབ་གསོ། 
དཔལ་འབྱོར། 
བདེ་ཐང། 
"""
