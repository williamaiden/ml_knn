# -*- coding: UTF-8 -*- 
'''
Created on 2017年6月16日

@author: William Aiden
'''
import csv
import random
import math
import operator

def load_data_set(file_name, split, training_set=[], test_set=[]):
    with open(file_name,'rb') as csv_file:
        lines = csv.reader(csv_file)
        data_set = list(lines)
        for x in range(len(data_set)-1):
            for y in range(4):
                data_set[x][y] = float(data_set[x][y])
            if random.random() < split:
                training_set.append(data_set[x])
            else:
                test_set.append(data_set[x])
                
def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)

def get_k_neighbors(training_set,test_instance, k):
    distances = []
    length = len(test_instance)-1
    for x in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    k_neighbors = []
    for x in range(k):
        k_neighbors.append(distances[x][0])
    return k_neighbors

def get_response(k_neighbors):
    class_votes = {}
    for x in range(len(k_neighbors)):
        response = k_neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sorted_votes[0][0]

def get_accuracy(test_set,predictions):
    corrent = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            corrent += 1
    return (corrent/float(len(test_set))) * 100.0

def main():
    training_set = []
    test_set = []
    split = 0.67
    load_data_set(r"knn_iris.txt", split, training_set, test_set)
    print(len(training_set))
    print(training_set)
    print(len(test_set))
    print(test_set)
    predictions = []
    k = 3
    for x in range(len(test_set)):
        k_neighbors = get_k_neighbors(training_set, test_set[x], k)
        result = get_response(k_neighbors)
        predictions.append(result)
        print(x)
        print(result)
        print(test_set[x][-1])
    accuracy = get_accuracy(test_set, predictions)
    print(repr(accuracy)+'%')
    
if __name__ == '__main__':    
    main()

#http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


# 96
# [[5.1, 3.5, 1.4, 0.2, 'Iris-setosa'], [4.9, 3.0, 1.4, 0.2, 'Iris-setosa'], [4.7, 3.2, 1.3, 0.2, 'Iris-setosa'], [5.4, 3.9, 1.7, 0.4, 'Iris-setosa'], [4.6, 3.4, 1.4, 0.3, 'Iris-setosa'], [5.0, 3.4, 1.5, 0.2, 'Iris-setosa'], [4.4, 2.9, 1.4, 0.2, 'Iris-setosa'], [4.9, 3.1, 1.5, 0.1, 'Iris-setosa'], [5.4, 3.7, 1.5, 0.2, 'Iris-setosa'], [5.8, 4.0, 1.2, 0.2, 'Iris-setosa'], [5.7, 4.4, 1.5, 0.4, 'Iris-setosa'], [5.1, 3.5, 1.4, 0.3, 'Iris-setosa'], [5.7, 3.8, 1.7, 0.3, 'Iris-setosa'], [5.1, 3.8, 1.5, 0.3, 'Iris-setosa'], [5.1, 3.3, 1.7, 0.5, 'Iris-setosa'], [4.8, 3.4, 1.9, 0.2, 'Iris-setosa'], [5.0, 3.0, 1.6, 0.2, 'Iris-setosa'], [5.2, 3.5, 1.5, 0.2, 'Iris-setosa'], [5.2, 3.4, 1.4, 0.2, 'Iris-setosa'], [4.8, 3.1, 1.6, 0.2, 'Iris-setosa'], [5.4, 3.4, 1.5, 0.4, 'Iris-setosa'], [5.5, 4.2, 1.4, 0.2, 'Iris-setosa'], [4.9, 3.1, 1.5, 0.1, 'Iris-setosa'], [5.0, 3.2, 1.2, 0.2, 'Iris-setosa'], [5.5, 3.5, 1.3, 0.2, 'Iris-setosa'], [4.4, 3.0, 1.3, 0.2, 'Iris-setosa'], [5.1, 3.4, 1.5, 0.2, 'Iris-setosa'], [4.5, 2.3, 1.3, 0.3, 'Iris-setosa'], [5.0, 3.5, 1.6, 0.6, 'Iris-setosa'], [5.1, 3.8, 1.9, 0.4, 'Iris-setosa'], [4.8, 3.0, 1.4, 0.3, 'Iris-setosa'], [5.1, 3.8, 1.6, 0.2, 'Iris-setosa'], [4.6, 3.2, 1.4, 0.2, 'Iris-setosa'], [5.3, 3.7, 1.5, 0.2, 'Iris-setosa'], [5.0, 3.3, 1.4, 0.2, 'Iris-setosa'], [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor'], [6.9, 3.1, 4.9, 1.5, 'Iris-versicolor'], [5.5, 2.3, 4.0, 1.3, 'Iris-versicolor'], [6.5, 2.8, 4.6, 1.5, 'Iris-versicolor'], [4.9, 2.4, 3.3, 1.0, 'Iris-versicolor'], [6.6, 2.9, 4.6, 1.3, 'Iris-versicolor'], [5.2, 2.7, 3.9, 1.4, 'Iris-versicolor'], [5.0, 2.0, 3.5, 1.0, 'Iris-versicolor'], [5.9, 3.0, 4.2, 1.5, 'Iris-versicolor'], [6.1, 2.9, 4.7, 1.4, 'Iris-versicolor'], [5.6, 2.9, 3.6, 1.3, 'Iris-versicolor'], [6.7, 3.1, 4.4, 1.4, 'Iris-versicolor'], [5.6, 3.0, 4.5, 1.5, 'Iris-versicolor'], [5.8, 2.7, 4.1, 1.0, 'Iris-versicolor'], [6.2, 2.2, 4.5, 1.5, 'Iris-versicolor'], [5.9, 3.2, 4.8, 1.8, 'Iris-versicolor'], [6.1, 2.8, 4.0, 1.3, 'Iris-versicolor'], [6.3, 2.5, 4.9, 1.5, 'Iris-versicolor'], [6.4, 2.9, 4.3, 1.3, 'Iris-versicolor'], [6.6, 3.0, 4.4, 1.4, 'Iris-versicolor'], [6.7, 3.0, 5.0, 1.7, 'Iris-versicolor'], [6.0, 2.9, 4.5, 1.5, 'Iris-versicolor'], [5.5, 2.4, 3.7, 1.0, 'Iris-versicolor'], [5.8, 2.7, 3.9, 1.2, 'Iris-versicolor'], [6.0, 2.7, 5.1, 1.6, 'Iris-versicolor'], [6.7, 3.1, 4.7, 1.5, 'Iris-versicolor'], [6.3, 2.3, 4.4, 1.3, 'Iris-versicolor'], [5.6, 3.0, 4.1, 1.3, 'Iris-versicolor'], [5.8, 2.6, 4.0, 1.2, 'Iris-versicolor'], [5.6, 2.7, 4.2, 1.3, 'Iris-versicolor'], [5.7, 3.0, 4.2, 1.2, 'Iris-versicolor'], [5.1, 2.5, 3.0, 1.1, 'Iris-versicolor'], [6.3, 3.3, 6.0, 2.5, 'Iris-virginica'], [5.8, 2.7, 5.1, 1.9, 'Iris-virginica'], [7.1, 3.0, 5.9, 2.1, 'Iris-virginica'], [6.5, 3.0, 5.8, 2.2, 'Iris-virginica'], [7.6, 3.0, 6.6, 2.1, 'Iris-virginica'], [4.9, 2.5, 4.5, 1.7, 'Iris-virginica'], [7.2, 3.6, 6.1, 2.5, 'Iris-virginica'], [6.8, 3.0, 5.5, 2.1, 'Iris-virginica'], [6.5, 3.0, 5.5, 1.8, 'Iris-virginica'], [7.7, 3.8, 6.7, 2.2, 'Iris-virginica'], [7.7, 2.6, 6.9, 2.3, 'Iris-virginica'], [6.9, 3.2, 5.7, 2.3, 'Iris-virginica'], [7.7, 2.8, 6.7, 2.0, 'Iris-virginica'], [6.7, 3.3, 5.7, 2.1, 'Iris-virginica'], [6.4, 2.8, 5.6, 2.1, 'Iris-virginica'], [7.2, 3.0, 5.8, 1.6, 'Iris-virginica'], [6.4, 2.8, 5.6, 2.2, 'Iris-virginica'], [6.3, 2.8, 5.1, 1.5, 'Iris-virginica'], [6.1, 2.6, 5.6, 1.4, 'Iris-virginica'], [6.3, 3.4, 5.6, 2.4, 'Iris-virginica'], [6.4, 3.1, 5.5, 1.8, 'Iris-virginica'], [6.0, 3.0, 4.8, 1.8, 'Iris-virginica'], [6.7, 3.1, 5.6, 2.4, 'Iris-virginica'], [6.9, 3.1, 5.1, 2.3, 'Iris-virginica'], [5.8, 2.7, 5.1, 1.9, 'Iris-virginica'], [6.7, 3.3, 5.7, 2.5, 'Iris-virginica'], [6.7, 3.0, 5.2, 2.3, 'Iris-virginica'], [6.5, 3.0, 5.2, 2.0, 'Iris-virginica'], [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']]
# 54
# [[4.6, 3.1, 1.5, 0.2, 'Iris-setosa'], [5.0, 3.6, 1.4, 0.2, 'Iris-setosa'], [4.8, 3.4, 1.6, 0.2, 'Iris-setosa'], [4.8, 3.0, 1.4, 0.1, 'Iris-setosa'], [4.3, 3.0, 1.1, 0.1, 'Iris-setosa'], [5.4, 3.9, 1.3, 0.4, 'Iris-setosa'], [5.4, 3.4, 1.7, 0.2, 'Iris-setosa'], [5.1, 3.7, 1.5, 0.4, 'Iris-setosa'], [4.6, 3.6, 1.0, 0.2, 'Iris-setosa'], [5.0, 3.4, 1.6, 0.4, 'Iris-setosa'], [4.7, 3.2, 1.6, 0.2, 'Iris-setosa'], [5.2, 4.1, 1.5, 0.1, 'Iris-setosa'], [4.9, 3.1, 1.5, 0.1, 'Iris-setosa'], [5.0, 3.5, 1.3, 0.3, 'Iris-setosa'], [4.4, 3.2, 1.3, 0.2, 'Iris-setosa'], [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor'], [5.7, 2.8, 4.5, 1.3, 'Iris-versicolor'], [6.3, 3.3, 4.7, 1.6, 'Iris-versicolor'], [6.0, 2.2, 4.0, 1.0, 'Iris-versicolor'], [5.6, 2.5, 3.9, 1.1, 'Iris-versicolor'], [6.1, 2.8, 4.7, 1.2, 'Iris-versicolor'], [6.8, 2.8, 4.8, 1.4, 'Iris-versicolor'], [5.7, 2.6, 3.5, 1.0, 'Iris-versicolor'], [5.5, 2.4, 3.8, 1.1, 'Iris-versicolor'], [5.4, 3.0, 4.5, 1.5, 'Iris-versicolor'], [6.0, 3.4, 4.5, 1.6, 'Iris-versicolor'], [5.5, 2.5, 4.0, 1.3, 'Iris-versicolor'], [5.5, 2.6, 4.4, 1.2, 'Iris-versicolor'], [6.1, 3.0, 4.6, 1.4, 'Iris-versicolor'], [5.0, 2.3, 3.3, 1.0, 'Iris-versicolor'], [5.7, 2.9, 4.2, 1.3, 'Iris-versicolor'], [6.2, 2.9, 4.3, 1.3, 'Iris-versicolor'], [5.7, 2.8, 4.1, 1.3, 'Iris-versicolor'], [6.3, 2.9, 5.6, 1.8, 'Iris-virginica'], [7.3, 2.9, 6.3, 1.8, 'Iris-virginica'], [6.7, 2.5, 5.8, 1.8, 'Iris-virginica'], [6.5, 3.2, 5.1, 2.0, 'Iris-virginica'], [6.4, 2.7, 5.3, 1.9, 'Iris-virginica'], [5.7, 2.5, 5.0, 2.0, 'Iris-virginica'], [5.8, 2.8, 5.1, 2.4, 'Iris-virginica'], [6.4, 3.2, 5.3, 2.3, 'Iris-virginica'], [6.0, 2.2, 5.0, 1.5, 'Iris-virginica'], [5.6, 2.8, 4.9, 2.0, 'Iris-virginica'], [6.3, 2.7, 4.9, 1.8, 'Iris-virginica'], [7.2, 3.2, 6.0, 1.8, 'Iris-virginica'], [6.2, 2.8, 4.8, 1.8, 'Iris-virginica'], [6.1, 3.0, 4.9, 1.8, 'Iris-virginica'], [7.4, 2.8, 6.1, 1.9, 'Iris-virginica'], [7.9, 3.8, 6.4, 2.0, 'Iris-virginica'], [7.7, 3.0, 6.1, 2.3, 'Iris-virginica'], [6.9, 3.1, 5.4, 2.1, 'Iris-virginica'], [6.8, 3.2, 5.9, 2.3, 'Iris-virginica'], [6.3, 2.5, 5.0, 1.9, 'Iris-virginica'], [6.2, 3.4, 5.4, 2.3, 'Iris-virginica']]
# 0
# Iris-setosa
# Iris-setosa
# 1
# Iris-setosa
# Iris-setosa
# 2
# Iris-setosa
# Iris-setosa
# 3
# Iris-setosa
# Iris-setosa
# 4
# Iris-setosa
# Iris-setosa
# 5
# Iris-setosa
# Iris-setosa
# 6
# Iris-setosa
# Iris-setosa
# 7
# Iris-setosa
# Iris-setosa
# 8
# Iris-setosa
# Iris-setosa
# 9
# Iris-setosa
# Iris-setosa
# 10
# Iris-setosa
# Iris-setosa
# 11
# Iris-setosa
# Iris-setosa
# 12
# Iris-setosa
# Iris-setosa
# 13
# Iris-setosa
# Iris-setosa
# 14
# Iris-setosa
# Iris-setosa
# 15
# Iris-versicolor
# Iris-versicolor
# 16
# Iris-versicolor
# Iris-versicolor
# 17
# Iris-versicolor
# Iris-versicolor
# 18
# Iris-versicolor
# Iris-versicolor
# 19
# Iris-versicolor
# Iris-versicolor
# 20
# Iris-versicolor
# Iris-versicolor
# 21
# Iris-versicolor
# Iris-versicolor
# 22
# Iris-versicolor
# Iris-versicolor
# 23
# Iris-versicolor
# Iris-versicolor
# 24
# Iris-versicolor
# Iris-versicolor
# 25
# Iris-versicolor
# Iris-versicolor
# 26
# Iris-versicolor
# Iris-versicolor
# 27
# Iris-versicolor
# Iris-versicolor
# 28
# Iris-versicolor
# Iris-versicolor
# 29
# Iris-versicolor
# Iris-versicolor
# 30
# Iris-versicolor
# Iris-versicolor
# 31
# Iris-versicolor
# Iris-versicolor
# 32
# Iris-versicolor
# Iris-versicolor
# 33
# Iris-virginica
# Iris-virginica
# 34
# Iris-virginica
# Iris-virginica
# 35
# Iris-virginica
# Iris-virginica
# 36
# Iris-virginica
# Iris-virginica
# 37
# Iris-virginica
# Iris-virginica
# 38
# Iris-virginica
# Iris-virginica
# 39
# Iris-virginica
# Iris-virginica
# 40
# Iris-virginica
# Iris-virginica
# 41
# Iris-versicolor
# Iris-virginica
# 42
# Iris-virginica
# Iris-virginica
# 43
# Iris-versicolor
# Iris-virginica
# 44
# Iris-virginica
# Iris-virginica
# 45
# Iris-virginica
# Iris-virginica
# 46
# Iris-virginica
# Iris-virginica
# 47
# Iris-virginica
# Iris-virginica
# 48
# Iris-virginica
# Iris-virginica
# 49
# Iris-virginica
# Iris-virginica
# 50
# Iris-virginica
# Iris-virginica
# 51
# Iris-virginica
# Iris-virginica
# 52
# Iris-versicolor
# Iris-virginica
# 53
# Iris-virginica
# Iris-virginica
# 94.44444444444444%
