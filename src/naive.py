# This Naive impelmentation works a preprocess matrix representation of data


from collections import defaultdict
import math


class NaiveBaye:
    def __init__(self,terms, classes = None, mtx = None) -> None:
        
        # this is a vocublary store every word in the form of {0: [<word>, repition in the document]}
        self.vocublary = defaultdict(lambda: ["", 0])
        with open(terms, 'r') as file:
            i = 0
            for line in file:
       
                self.vocublary[i][0] = line[:-1]    
                i += 1
        # this stores the 2225 text with their number and class lablel
        # the text are represent with number
        # 0 - businness
        # 1- entertainment
        # 2 - politics
        # 3 - sport
        # 4 - tech              
        self.textLabel = {}
        with open(classes, 'r') as file:
            for line in file:
                wo = line[:-1].split(" ")
                self.textLabel[ int(wo[0]) ] = int(wo[1])
                
                
        self.textTraining = defaultdict(lambda: defaultdict(int))
        self.textTest = defaultdict(lambda: defaultdict(int))
        self.classCount = defaultdict(set)
        
        # this is a vectorizer
        # for vectization we used dictionary instead of array to save some sace as most of the words are not present in the text takes useless space in array 
        # it assigns the word to the text
        # means  text - {word - repition}
        # we ignored the text file which are not in the range of 400 - 510, 810 - 896, 1196 - 1313, 1713 - 1824, 2144 - 2225
        # so we can use them as test data
        
        with open(mtx, 'r') as file:
            for line in file:
                wo = line[:-1].split(" ")
                wo[0] = int(wo[0]) - 1
                wo[1] = int(wo[1]) - 1
                wo[2] = int(float(wo[2]))
                if 400 < wo[1] < 510 or 810 < wo[1] < 896 or 1196 < wo[1] < 1313 or 1713 < wo[1] < 1824 or 2144 < wo[1] < 2225:
                    self.textTest[ wo[1] ][ wo[0] ] = wo[2]
                else:
                    self.textTraining[ wo[1] ][ wo[0] ] = wo[2]
                    self.classCount[ self.textLabel[ wo[1] ]].add(wo[1])
                    self.vocublary[wo[0]][1] += 1          
                    
        self.classProbablity = {}
        
        # Below are the feature  requirments for the naive bayes
        
        # feature 1 tools - bag of words
        self.classFeature1 = {}
        self.wordProbablityFeature1 = {}
        self.alphaValueFeature1 = 0
        
        # feature 2 tools - frequency of words
        self.wordProbablityFeature2 = {}
        self.alphaValueFeature2 = 0
        
        # feature 3 tools - tf - idf of words
        self.classFeaure3 = {}
        self.wordProbablityFeature3 = {}
        self.alphaValueFeature3 = 0
        
        
    def train(self, alpha = 0):
        self.calculateClassProbability()
        
        self.calculateWordProbabilityFeature1(alpha)
        self.calculateWordProbabilityFeature2(alpha)
        self.calculateWordProbabilityFeature3(alpha)
        
    def calculateClassProbability(self):
        total = 0
        for i in self.classCount:
            total += len(self.classCount[i])
        for classes in self.classCount:
            self.classProbablity[classes] = len(self.classCount[classes])
        for classes in self.classCount:
            self.classProbablity[classes] /= total
    
    
    def calculateWordProbabilityFeature1(self, alpha = 0, trainingData = None, outPutData = None, classWords = None):

        if trainingData == None:
            self.wordProbablityFeature1 = defaultdict( lambda: defaultdict(int))
            outPutData = self.wordProbablityFeature1
            trainingData = self.textTraining
            self.alphaValueFeature1 = alpha
            self.classFeature1 = defaultdict(int)
            classWords = self.classFeature1
        # count the repitition Word >>> class  >>> repition
        for text, words in trainingData.items():
            textClass = self.textLabel[text]
            for word, repetition in words.items():
                outPutData[word][textClass] += repetition
                classWords[textClass] += repetition
        # calculate probability Word >>> class  >>> relative wordcount
        self.classFeature1 = classWords
        for word, classes in outPutData.items():
            for clas, repitions in classes.items():
                numerator = repitions + alpha
                denominator = classWords[clas] + (alpha * len(self.vocublary))
                outPutData[word][clas] = numerator / denominator
                    
    
    def predictFeature1(self, textNumber, alpha = 0, wordFeature = None, classFeature = None):
        
        test = self.textTest[textNumber] if textNumber in self.textTest else self.textTraining[textNumber]
        if wordFeature == None:
            alpha = self.alphaValueFeature1
            wordFeature = self.wordProbablityFeature1
            classFeature = self.classFeature1
        
        answer = defaultdict(int)
         
        for word, repetition in test.items():
            for clas in self.classProbablity:
                if clas in wordFeature[word]:
                    answer[clas] += (math.log(wordFeature[word][clas]) * repetition)
                else:
                    answer[clas] += math.log(alpha / (classFeature[clas]  + (alpha * len(self.vocublary))))
        
        answer = list(answer.items())
        answer.sort(key = lambda x: x[1], reverse = True)
        return answer[0][0]
    
    
            
    def calculateWordProbabilityFeature2(self, alpha = 0):
                    
        # frequency Counter word probability calculator         
        self.wordProbablityFeature2 = defaultdict( lambda: defaultdict( lambda: defaultdict(int)))
        self.alphaValueFeature2 = alpha
        
        # count the repitition Word >>> class  >>> repition
        for text, words in self.textTraining.items():
            textClass = self.textLabel[text]
            for word, repetition in words.items():
                self.wordProbablityFeature2[word][textClass][repetition] += 1
        # calculate probability Word >>> class  >>> repition >> probability
        for word, classes in self.wordProbablityFeature2.items():
            for clas, repitions in classes.items():
                denominator = len(self.classCount[clas]) 
                for repition in repitions:
                    numerator = repetition + alpha
                    self.wordProbablityFeature2[word][clas][repition] = numerator / denominator
        
    def predictFeature2(self, textNumber, alpha = 0, testData = None):
        
        test = self.textTest[textNumber] if textNumber in self.textTest else self.textTraining[textNumber]
        if testData == None:
            alpha = self.alphaValueFeature2
            testData = self.wordProbablityFeature2
        
        answer = defaultdict(int)
         
        for word, repetition in test.items():
            for clas in self.classProbablity:
                if repetition in testData[word][clas]:
                    probability = testData[word][clas][repetition]
                else:
                    probability = alpha / len(self.classCount[clas])
                    
                answer[clas] += math.log(probability)
        
        answer = list(answer.items())
        answer.sort(key = lambda x: x[1], reverse = True)
        return answer[0][0]
    
    
    
    def calculateWordProbabilityFeature3(self, alpha = 0):
        
        def tfidfCalculate():
            document = len(self.textTraining)
            tfidf = defaultdict(lambda: defaultdict(int))
            for text, words in self.textTraining.items():
                sums = sum(words.values())
                textClass = self.textLabel[text]
                for word, repetition in words.items():
                    tfidf[text][word] += ((repetition / sums) *  (math.log(len(self.textTraining) / self.vocublary[word][1])) )
            return tfidf 
        
        
        self.alphaValueFeature3 = alpha
        self.wordProbablityFeature3 = defaultdict( lambda: defaultdict(int))
        self.classFeature3 = defaultdict(int)
        trainingData = tfidfCalculate()
        self.calculateWordProbabilityFeature1(alpha, trainingData, self.wordProbablityFeature3, self.classFeature3)
    
    def predictFeature3(self, textNumber):
        return self.predictFeature1(textNumber, self.alphaValueFeature3, self.wordProbablityFeature3, self.classFeature3)
        
classifier = NaiveBaye("./data/bbc.terms", "./data/bbc.classes", "./data/bbc.mtx")

# for predictFeature1ion we use the test data numbered from
# >> 400 to 510  -  0
# >> 810 to 896  -  1
# >> 1196 to 1313 - 2
# >> 1713 to 1824  - 3
# >> 2144 to 2224  - 4
# this classes were ignored in the training data so we can use them for testing

def accuracyCalc(alpha = 0.1, feature = classifier.predictFeature1):
    classifier.train(alpha)
    testing = [ _ for _ in range(400, 510)] + [ _ for _ in range(810, 896)] + [ _ for _ in range(1196, 1313)] + [ _ for _ in range(1713, 1824)] + [ _ for _ in range(2144, 2225)]
    
    feature1 = [0, 0]
    for i in testing:
        pred = feature(i)
        feature1[0] += 1 if pred == classifier.textLabel[i] else 0
        feature1[1] += 1
    return feature1[0] / feature1[1]

alphas = [0.01, 0.1, 0.5, 1, 2, 5, 10, 100]

print("\n")
print("Accuracy based on Bag of Words")
for alpha in alphas:
    print("\tAccuracy for lapalce smoothing = ", alpha, "is - ", accuracyCalc(alpha, classifier.predictFeature1))
    
print("\n")  
print("Accuracy based on Frequency word")
for alpha in alphas:
    print("\tAccuracy for lapalce smoothing = ", alpha, "is - ", accuracyCalc(alpha, classifier.predictFeature2))
    
print("\n")
print("Accuracy based on IF-IDF")
for alpha in alphas:
    print("\tAccuracy for lapalce smoothing = ", alpha, "is - ", accuracyCalc(alpha, classifier.predictFeature3))
print("\n")