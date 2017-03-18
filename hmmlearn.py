import sys
from collections import OrderedDict
import os

def loadTrainData(trainTextPath):
    """
    This function loads the training data and 
    creates the data list
    Args:
        trainTextPath: training data file path
    Returns:
        trainDataList: List of training data
    """
    trainDataList=[]
    with open(trainTextPath) as trainData:
        trainDataList = trainData.readlines()    
    trainDataList = [x.strip() for x in trainDataList]
    return trainDataList
    
def createTagCount(trainDataList):
    """
    This function creates the freequency dictionary of 
    tags in the training data
    Args:
        trainDataList: List of training data
    Returns:
        tagCount: dictionary with tag as key and freequency as value
    """
    tagCount={}
    for sentence in trainDataList:
        sentenceList=sentence.split()
        for i in range(len(sentenceList)):
            wordAndTag=sentenceList[i].rsplit("/",1)
            if(wordAndTag[1] in tagCount):
                tagCount[wordAndTag[1]]+=1
            else:
                tagCount[wordAndTag[1]]=1
    return tagCount
    
def getTrainsitionCounts(trainDataList):
    """
    This function creates the tag to tag freequencies in the 
    training data, complete tag list and 
    freequency of tokens under each tag
    Args:
        trainDataList: List of training data
    Returns:
        tagToTag: dictionary of tag bigram freequencies
        tagList: List of all the tags in the corpus
        tagAndWord: Freequency of tokens under each tag
    """
        
    tagList=[]
    tagToTag= OrderedDict()
    tagAndWord=OrderedDict()
    
    for sentence in trainDataList:
        
        sentenceList=sentence.split()
        
        for i in range(len(sentenceList)):
            inTag={}
            inWord={}
            wordAndTag=sentenceList[i].rsplit("/",1)
            
            if(wordAndTag[0] in inWord):
                inWord[wordAndTag[0]]+=1
            else:
                inWord[wordAndTag[0]]=1
                
            if(wordAndTag[1] in tagAndWord):
                if(wordAndTag[0] in tagAndWord[wordAndTag[1]]):
                    tagAndWord[wordAndTag[1]][wordAndTag[0]]+=1
                else:
                    tagAndWord[wordAndTag[1]][wordAndTag[0]]=1
            else:
                tagAndWord[wordAndTag[1]]=inWord
                    
                    
            
            
            
                    
            if(i==0):        
                if(wordAndTag[1] in inTag):
                    inTag[wordAndTag[1]]+=1
                    
                else:
                    inTag[wordAndTag[1]]=1
                    
                if("q0" in tagToTag):
                    if(wordAndTag[1] in tagToTag["q0"]):
                        tagToTag["q0"][wordAndTag[1]]+=1
                    else:
                        tagToTag["q0"][wordAndTag[1]]=1
                    
                else:
                    tagToTag["q0"]=inTag  
                
                
            else:
                prevWordAndTag=sentenceList[i-1].rsplit("/",1)
                if(wordAndTag[1] in inTag):
                    inTag[wordAndTag[1]]+=1                    
                else:
                    inTag[wordAndTag[1]]=1
                
                if(prevWordAndTag[1] in tagToTag):
                    if(wordAndTag[1] in tagToTag[prevWordAndTag[1]]):
                        tagToTag[prevWordAndTag[1]][wordAndTag[1]]+=1
                    else:
                        tagToTag[prevWordAndTag[1]][wordAndTag[1]]=1
                else:
                    tagToTag[prevWordAndTag[1]]=inTag     
                
                
                
            
            if(wordAndTag[1] not in tagList):
               tagList.append(wordAndTag[1])
            
            
           
    
               
    return tagToTag,tagList,tagAndWord
    
def createTransitionProbs(tagToTag,tagList):
    """
    This function creates the state transition probabilities
    Args:
        tagToTag: dictionary of bigram tag freequencies
        tagList: list of tags in the corpus
    Returns:
        tagToTag: dictionary of bigram tag transition probabilities
    """
    
    vocabSize=len(tagList)
    for key,items in tagToTag.iteritems():
        sum=0
        for inKey,item in items.iteritems():
            sum+=item
        for inKey,item in items.iteritems():
            tagToTag[key][inKey]=(tagToTag[key][inKey]+1)/(float)(sum+vocabSize)
        for inTag in tagList:
            if inTag not in tagToTag[key]:
                tagToTag[key][inTag]=1/(float)(sum+vocabSize)
        
    return tagToTag
    
def writeModelFile(tagCount,tagToTag,tagAndWord,tagList):
    """
    This file creates the model parameters required for the
    hmm decode module
    Args:
        tagCount: dictionary of tag and their freequency
        tagToTag: transition probability dictionary
        tagAndWord: dictionary of tags and the tokens it tagged
        tagList: list of tags in the corpus
    Returns:
        None
    """
    hmmModel=open("Z:/MSBooks/NLP/HW5/hmmmodel.txt","w")
    hmmModel.write("0Tag|Freequency"+os.linesep)
    tagCountModelSign="*"
    for key,items in tagCount.iteritems():
        line =  "{}|{}".format(tagCountModelSign+key,items)
        hmmModel.write(line+os.linesep)
    
    hmmModel.write("1State_TransitionProbabilities"+os.linesep)
    for key,items in tagToTag.iteritems():
        keyLine = "{}:".format(key)
        hmmModel.write(keyLine)
        for inKey,inItem in items.iteritems():
            line = "{}|{},".format(inKey,inItem)
            hmmModel.write(line)
        hmmModel.write(os.linesep)
    
    hmmModel.write("2Emission_Stats"+os.linesep)
    for key,items in tagAndWord.iteritems():
        keyLine="#{}:".format(key)
        hmmModel.write(keyLine)
        for inKey,inItem in items.iteritems():
            line = "{}|{} ".format(inKey,inItem)
            hmmModel.write(line)
        hmmModel.write(os.linesep)
    hmmModel.write("3tagList"+os.linesep)
    hmmModel.write("!")
    for item in tagList:
        hmmModel.write(item+",")
        

    
def main(trainTextPath):
    """
    This is the Catalan POS Tagger training module
    Args:
        Training Data path
    Returns:
        None
    """
    trainDataList = loadTrainData(trainTextPath)
    tagCount=createTagCount(trainDataList)
    tagToTag,tagList,tagAndWord = getTrainsitionCounts(trainDataList)
    transProbMatrix = createTransitionProbs(tagToTag,tagList)
    writeModelFile(tagCount,transProbMatrix,tagAndWord,tagList)
    
    
    
    
    
    
main(sys.argv[1])