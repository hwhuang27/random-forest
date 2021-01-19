import sys
import numpy as np
import pandas as pd
import math
import random
from pprint import pprint

def entropy(feature):
    prob = feature.value_counts(normalize = True)
    entropy = -(np.sum(np.log2(prob) * prob))
    return round(entropy,5)

def info_gain(data, feature, target):
    target_entropy = entropy(data[target])
    entropy_list = []
    weight_list = []
    
    for element in data[feature].unique():
        feature_elem = data[data[feature] == element]
        entropy_elem = entropy(feature_elem[target])
        entropy_list.append(round(entropy_elem,5))
        
        weight_elem = len(feature_elem) / len(data)
        weight_list.append(round(weight_elem,5))
        
    feature_wrt_target = np.sum(np.array(entropy_list) * np.array(weight_list))
    info_gain = target_entropy - feature_wrt_target

    return round(info_gain,5)

def best_split(data, features, target):
    # find the best split feature via greatest information gain
    maxInfoGain = 0
    for feature in features:    
        infoGain = info_gain(data, feature, target)
        if infoGain >= maxInfoGain:
            maxInfoGain = infoGain
            maxFeature = feature

    return maxFeature
    
def learn_dtree(tree, data, features, target):
    count = data[target].value_counts()
    
    # termination conditions
    if len(count.index) == 1 or len(features) == 0 or (data.shape[1] == 1 and data.columns[0] == target):
        tree[(target, count.index[count.argmax()])] = {}
        return

    # get feature w/ max info gain.
    maxFeature = best_split(data, features, target)
    
    # build tree
    for elem in set(data[maxFeature]):
        tree[(maxFeature, elem)] = {}
        partition = data[data[maxFeature] == elem].drop(maxFeature, axis = 1)
        partition = partition.reset_index(drop = True)
        learn_dtree(tree[(maxFeature, elem)], partition, features - {maxFeature}, target)

def predict_row(tree, row, target):
    # predict label for a single row in test DataFrame
    size_tree = len(tree)
    size_row = len(row)
    if size_tree == 0 or size_row == 0:
        return None
    
    for feature, elem in tree:
        if feature == target:
            return elem
        if feature in row.index and row[feature] == elem:
            return predict_row(tree[(feature, elem)], row.drop(feature), target)

def predict_labels(data, tree, target):
    # predict labels for every row in test DataFrame
    return data.apply(lambda row: predict_row(tree, row, target), axis = 1)

def random_forest(train, test, features, numTrees, featureRatio, target):
    predict_df = pd.DataFrame()
    # note: feature == attribute
    for i in range(numTrees):
        dtree = dict()
        sampleSize = round(len(features)*featureRatio)
        selectedFeatures = set(random.sample(features, sampleSize))
        learn_dtree(dtree, train, selectedFeatures, target)
        prediction = predict_labels(test, dtree, target)
        predict_df[i] = prediction
    
    predict_df[target] = predict_df.mode(axis=1)[0]
    return predict_df

def save_predictions(result, target):
    result[target].to_csv('predictions.csv')
    return

def main():
    train = pd.read_csv("banks.csv")
    test = pd.read_csv("banks-test.csv")
    target = 'label'
    # single d-tree / prediction
    '''
    dtree = dict()
    learn_dtree(dtree, train, set(train.columns[:-1]), target)
    predict = predict_labels(test, dtree, target)
    #pprint(dtree)
    #print(predict)
    '''
    # random forest parameters
    numTrees = 12
    attributeRatio = 0.8
    result = random_forest(train, test, set(train.columns[:-1]), numTrees, attributeRatio, target)
    save_predictions(result, target)
    # 12 and 0.8 seem to give an average of around 80% accuracy
    
    # check accuracy of prediction
    testTarget = test[target]
    check = testTarget == result[target]
    check = check.value_counts()
    accuracy = check[1] / len(testTarget) * 100
    print('numberOfTrees =', numTrees)
    print('percentageOfAttributes =', attributeRatio)
    print('accuracy = ', accuracy, '%')

if __name__ == '__main__':
    main()
    

    