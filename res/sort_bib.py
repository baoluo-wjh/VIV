import re

strDict = dict()

file_name = "../VIV.bib"

with open(file_name, 'r') as f:
    allFile = f.read()
    endPos = len(allFile)
    allAta = []
    pos = -1
    while pos < endPos:
        pos = allFile.find('@', pos + 1)
        if pos != -1:
            allAta.append(pos)
        else:
            break
    allAta.append(endPos)
    for i in range(len(allAta) - 1):
        item = allFile[allAta[i] : allAta[i + 1]]
        # pattern: @article{casas1994combined,
        searchObj = re.search(r"@.*\{(\D*?)(\d*?)(\D*?),", item)
        if searchObj:
            itemKey = (searchObj.group(1), searchObj.group(2), searchObj.group(3))
            strDict[itemKey] = item
        else:
            raise TypeError()

strKeyList = list(strDict.keys())

for i in range(len(strKeyList) - 1):
    for j in range(i + 1, len(strKeyList)):
        author_1, time_1, word_1 = strKeyList[i]
        author_2, time_2, word_2 = strKeyList[j]
        flag = False
        if time_1 < time_2:
            flag = True
        elif time_1 == time_2 and author_1 > author_2:
            flag = True
        elif time_1 == time_2 and author_1 == author_2 and word_1 > word_2:
            flag = True
        if flag:
            strKeyList[i], strKeyList[j] = strKeyList[j], strKeyList[i]

# print(strKeyList)

with open(file_name, 'w') as f:
    for key in strKeyList:
        f.write(strDict[key])
