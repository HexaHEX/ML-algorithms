import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from random import shuffle, seed
from sklearn.svm import SVC
def loadDataset(filename, split, trainingSet=[] , testSet=[],trainingclass = [] ,testclass = []):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        seed(split)
        shuffle(dataset)
        for x in range(len(dataset)):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if x < split*len(dataset):
                trainingSet.append(dataset[x][:4]) 
                trainingclass.append(dataset[x][4])
            else:
                testSet.append(dataset[x][:4])
                testclass.append(dataset[x][4])
 
 
 
 
def getAccuracy(testclass, predictions):
    correct = 0
    for x in range(len(testclass)):
        if testclass[x] == predictions[x]:
            correct += 1
    return (correct/float(len(testclass))) * 100.0
	
def main():
	# prepare data
    trainingSet=[]
    testSet=[]
    trainingclass = []
    testclass = []
    split = 0.67
    loadDataset('iris.data', split, trainingSet, testSet,trainingclass,testclass)
    #print 'Train set: ' + repr(len(trainingSet))
    #print 'Test set: ' + repr(len(testSet))
    #print 'Test set: ' + str(trainingclass)
    #print 'Test set: ' + repr(len(testSet))
	# generate predictions
    predictions=[]
    k = 10
    #neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
    #neigh = tree.DecisionTreeClassifier()
    neigh = SVC()
    neigh.fit(trainingSet,trainingclass)
    for x in range(len(testSet)):
        result = neigh.predict([testSet[x]])
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testclass[x]))


    accuracy = getAccuracy(testclass, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
	
main()
