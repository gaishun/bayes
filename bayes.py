# coding=utf-8
import math


import numpy as np
import random

def loadDataSet():
	'''数据加载函数。这里是一个小例子'''
	postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
					['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
					['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
					['stop', 'posting', 'stupid', 'worthless', 'garbage'],
					['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
					['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0, 1, 0, 1, 0, 1]   # 1代表侮辱性文字，0代表正常言论，代表上面6个样本的类别
	return postingList, classVec

def createVocabList(dataSet):
	'''
	创建所有文档中出现的不重复词汇列表
	Args:
		dataSet: 所有文档
	Return:
		包含所有文档的不重复词列表，即词汇表
	'''
	vocabSet = set([])
	# 创建两个集合的并集
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

# 词集模型(set-of-words model):词在文档中是否存在，存在为1，不存在为0
def setOfWord2Vec(vocabList, inputSet):
	'''
	依据词汇表，将输入文本转化成词集模型词向量
	Args:
		vocabList: 词汇表
		inputSet: 当前输入文档
	Return:
		returnVec: 转换成词向量的文档
	例子：
		vocabList = ['I', 'love', 'python', 'and', 'machine', 'learning']
		inputset = ['python', 'machine', 'learning']
		returnVec = [0, 0, 1, 0, 1, 1]
		长度与词汇表一样长，出现了的位置为1，未出现为0，如果词汇表中无该单词则print
	'''
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print("the word: %s is not in my vocabulary!" % word)
	return returnVec


def trainNB0(trainMatrix, trainCategory):#两个np.array[]
	'''
	朴素贝叶斯分类器训练函数，求：p(Ci),基于词汇表的p(w|Ci)
	Args:
		trainMatrix : 训练矩阵，即向量化表示后的文档（词条集合）
		trainCategory : 文档中每个词条的列表标注
	Return:
		p0Vect : 属于0类别的概率向量(p(w1|C0),p(w2|C0),...,p(wn|C0))
		p1Vect : 属于1类别的概率向量(p(w1|C1),p(w2|C1),...,p(wn|C1))
		pAbusive : 属于1类别文档的概率
	'''

	# 长度为训练数据维度
	numTrainDocs = len(trainMatrix)
	# 长度为词汇表长度
	numWords = len(trainMatrix[0])
	# 计算先验概率
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	# 拉普拉斯平滑处理
	p0Num = np.ones(numWords)
	p1Num = np.ones(numWords)
	p0Denom = numTrainDocs*1.0 + 1
	p1Denom = numTrainDocs*1.0 + 1
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]# p(wi | c1)
		else:
			p0Num += trainMatrix[i]# p(wi | c2)
	#取对数 运算优化操作
	p1Vect = np.log(p1Num/p1Denom)
	p0Vect = np.log(p0Num/p0Denom)

	return p0Vect, p1Vect, pAbusive
	
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	'''
	朴素贝叶斯分类器
	Args:
		vec2Classify : 待分类的文档向量（已转换成array）
		p0Vec : p(w|C0)
		p1Vec : p(w|C1)
		pClass1 : p(C1)
	Return:
		1 : 为侮辱性文档 (基于当前文档的p(w|C1)*p(C1)=log(基于当前文档的p(w|C1))+log(p(C1)))
		0 : 非侮辱性文档 (基于当前文档的p(w|C0)*p(C0)=log(基于当前文档的p(w|C0))+log(p(C0)))
	'''
	# print(vec2Classify * p1Vec)
	p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
	p0 = np.sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

#--------------------------------------使用小例子进行测试------------------------------------
def testingNB():
	'''测试'''
	listOPosts, lisClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
	p0Vec, p1Vec, pAb = trainNB0(np.array(trainMat), np.array(lisClasses))
	
	testEntry1 = ['love', 'my', 'dalmation']
	thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry1))
	print(testEntry1, 'classified as:', classifyNB(thisDoc, p0Vec, p1Vec, pAb))

	testEntry2 = ['stupid', 'garbage']
	thisDoc2 = np.array(setOfWord2Vec(myVocabList, testEntry2))
	print(testEntry2, 'classified as:', classifyNB(thisDoc2, p0Vec, p1Vec, pAb))

#--------------------------------------进行垃圾邮件测试------------------------------------
def textParse(bigString):
	'''
	分词函数
	Args:
		bigString: 待分词文档
	Return:
		listOfTokens: 删除标点符号空格符等，已被转换成小写的字符串列表(删去少于两个字符的字符串)
	'''
	import re
	ans = []
	listOfTokens = str.split(bigString,' ')
	for tok in listOfTokens :
		if len(tok) >2:
			ans.append(tok)
	return ans

def spamTest():
	docList = []
	classList = []
	for i in range(1,26):
		wordList = textParse(open('email/spam/%d.txt' % i).read())
		docList.append(wordList)
		classList.append(1)
		wordList = textParse(open('email/ham/%d.txt' % i).read())
		docList.append(wordList)
		classList.append(0)

	#get the result of all splited files
	vocabList = createVocabList(docList)
	# print("vocabList=",vocabList)


	# random select the trainSetIndex and the testSetIndex
	trainingSet = list(range(50))  # trainingSet = [0,1,2,...,49]
	testSet = []
	for i in range(10):
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del (trainingSet[randIndex])



	trainMat = []
	trainClasses = []
	for docIndex in trainingSet:
		#从单词列表转成向量（单词在单词表中存在是1 不再单词表中存在是0，维度跟单词表的维度是一样的。 ）
		trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])

	# all the size of trainmat is 830 ?
	# size = []
	# for temp in trainMat:
	# 	size.append(len(temp))
	# print(size)

	p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
	# print(p0V,'\n',p1V,'\n',pSpam)
	errorCount = 0
	for docIndex in testSet:
		wordVector = setOfWord2Vec(vocabList, docList[docIndex])
		if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1
	print('the error rate is:', float(errorCount/len(testSet)))
























