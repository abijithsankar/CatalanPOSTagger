import sys
from collections import OrderedDict
import os
import math

def loadTestData(testDataPath):
    """
    This function loads the test data and 
    creates the test data list
    Args:
        testDataPath: testing data file path
    Returns:
        testDataList: list of testing data
    """
    testDataList=[]
    with open(testDataPath) as testData:
        testDataList = testData.readlines()    
    testDataList = [x.strip() for x in testDataList]
    return testDataList

def loadModelData(modelDataPath):
    """
    This function loads the model parameters data and 
    creates model data list
    Args:
        modelDataPath: model data file path
    Returns:
        modelDataList: list of model parameter data
    """
    with open(modelDataPath) as modelData:        
        modelDataList= [x.strip("\r\n") for x in modelData.readlines()[1:]]
    return modelDataList
    
def processDataModel(modelDataList):
    """
    This function processes the model data file and
    retrieves the model parameters
    Args:
        modelDataList: list of model data
    Returns:
        tagCount: dictionary of tag and freequency
        tagToTag: transition probability dictionary
        tagAndWord: dictionary of tags and their tokens
        tagList: list of tags in the corpus
    """
        
    tagCount={}
    tagToTag=OrderedDict()
    tagAndWord=OrderedDict()
    tagList=[]
    for item in modelDataList:
        if(item.startswith("*")):
            value=item[1:]
            tagAndCount=value.split("|")
            tagCount[tagAndCount[0]]=tagAndCount[1]
        elif item.startswith("1") or item.startswith("2") or item.startswith("3"):
            continue
        elif item.startswith("#"):
            value=item[1:]
            inWord={}
            keyAndValue=value.split(":",1)
            valueItems=keyAndValue[1].strip().split()
            for items in valueItems:
                inKeyAndValue=items.rsplit("|")
                inWord[inKeyAndValue[0]]=inKeyAndValue[1]
            tagAndWord[keyAndValue[0]]=inWord
        elif item.startswith("!"):
            value=item[1:]
            tagItems=value.strip(",").split(",")
            for items in tagItems:
                tagList.append(items)    
            
        else:
            inTag={}
            keyAndValue=item.split(":")
            valueItems=keyAndValue[1].strip(",").split(",")
            for items in valueItems:
                inKeyAndValue=items.split("|")
                inTag[inKeyAndValue[0]]=inKeyAndValue[1]
            tagToTag[keyAndValue[0]]=inTag
    return tagCount,tagToTag,tagAndWord,tagList
    

        
def createEmissionProbs(tagCount,sentence,tagAndWord):
    """
    This function creates the emission probability matrix
    Args:
        tagCount: dictionary of tag and freequency
        sentence: test data sentence
        tagAndWord: dictionary of tags and their tokens
    Returns:
        emission: Emission probability matrix
    """
    emission=OrderedDict()
    observations=sentence.split()
    for word in observations:
        for key,items in tagAndWord.iteritems():
                inWord=OrderedDict()               
                if(word in items):
                    inWord[word]=(float)(items[word])/(float)(tagCount[key])
                    if(key in emission):
                        emission[key][word]=(float)(items[word])/(float)(tagCount[key])
                            
                    else:
                        emission[key]=inWord
                else:
                    if(key in emission):
                        emission[key][word]=0.00000001
                    else:
                        unseenInWord=OrderedDict()
                        unseenInWord[word]=0.00000001
                        emission[key]=unseenInWord
                        
        
    return emission 
    
                    
def createViterbi(tagToTag,tagList,testDataList,tagCount,tagAndWord):
    """
    This function creates the viterbi decoder and
    returns the predicted tag list for the test data
    Args:
        tagToTag: transition probability dictionary
        tagList: list of tags in the corpus
        testDataList: list of test data
        tagCount: dictionary of tag and freequency
        tagAndWord: dictionary of tag and their tokens
    Returns:
        predictedTagList: list of predicted tags for the test data
    """
    predictedTagList=[]
    for item in testDataList:
        viterbi=[{}]
        emission = createEmissionProbs(tagCount,item,tagAndWord)
        #print emission
        observations=item.split()
        for st in tagList:
            viterbi[0][st]={"prob":math.log((float)(tagToTag["q0"][st])) + math.log(emission[st][observations[0]]),"prev":None}
                                       
        for i in range(1,len(observations)):
            viterbi.append({})
            for st in tagList:
                max_trans_prob=max((viterbi[i-1][prev_state]["prob"]) + math.log(float(tagToTag[prev_state][st])) for prev_state in tagList)
                for prev_state in tagList:
                    if (viterbi[i-1][prev_state]["prob"]) + math.log(float(tagToTag[prev_state][st])) == max_trans_prob:
                        max_prob = max_trans_prob + math.log(emission[st][observations[i]])
                        viterbi[i][st] = {"prob": max_prob, "prev": prev_state}
                        break                 
        opt=[]        
        max_prob=max(value["prob"] for value in viterbi[-1].values())
        previous=None
        for st,data in viterbi[-1].items():
            if(data["prob"]==max_prob):
                opt.append(st)
                previous=st
                break
        for i in range(len(viterbi)-2,-1,-1):
            opt.insert(0,viterbi[i+1][previous]["prev"])
            previous=viterbi[i+1][previous]["prev"]
        predictedTagList.append(opt)
                
    
    return predictedTagList
  
def writeOutputFile(predictedTagList,testDataList):
    """
    This function creates the hmm output file
    Args:
        predictedTagList: list of predicted tags
        testDataList: list of test data
    Returns:
        None
    """
    hmmOut=open("Z:/MSBooks/NLP/HW5/hmmoutput.txt","w")
    count=0
    for item in testDataList:
        line=[]
        observations=item.split()
        for j in range(len(observations)):
            line.append(observations[j]+"/"+predictedTagList[count][j])
        hmmOut.write(' '.join(line)+os.linesep) 
        count=count+1  
    
    
def main(testDataPath):
    """
    This is the hmm decoder using viterbi algorithm
    for Catalan POS Tagger
    Args:
        testDataPath: testing file path
    """
    testDataList = loadTestData(testDataPath)
    modelDataList = loadModelData("Z:/MSBooks/NLP/HW5/hmmmodel.txt")
    tagCount,tagToTag,tagAndWord,tagList = processDataModel(modelDataList)
    predictedTagList = createViterbi(tagToTag,tagList,testDataList,tagCount,tagAndWord)
    writeOutputFile(predictedTagList,testDataList)
    
main(sys.argv[1])