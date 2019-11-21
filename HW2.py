#!/usr/bin/env python
# coding: utf-8

# In[ ]:

def create_datasets(directory, encoding, out_bow=None, out_bern=None, write='N'):
    wordset = set([])
    features = [['Y']]
    for folder in os.listdir(directory):
        class_val = 1 if folder == "ham" else 0
        for filename in os.listdir(directory+folder+"/"):
            features.append([class_val] + [0]*len(wordset))
            with open(directory+folder+"/"+filename,"r", encoding=encoding) as file:
                for word in file.read().split():
                    if word not in wordset:
                        wordset.add(word)
                        features[0].append(word)
                        features[-1].append(1)
                    else:
                        features[-1][features[0].index(word)] += 1
                        
    for row in features[1:]:
        row += [0]*(len(features[0])-len(row))
        
    bow_ds = np.array(features)
    bern_ds = bow_ds.copy()
    bern_ds[1:,:] = np.where(bern_ds[1:,:] > '0' , '1', '0')
    
    if write == 'Y':
        with open(out_bow, "w", newline='', encoding=encoding) as f:
            writer = csv.writer(f, lineterminator = '\n')
            writer.writerows(bow_ds)

        with open(out_bern, "w", newline='', encoding=encoding) as f:
            writer = csv.writer(f, lineterminator = '\n')
            writer.writerows(bern_ds)
    
    return bern_ds, bow_ds
        
def create_test_datasets(directory, V, encoding, out_bow=None, out_bern=None, write='N'):
    wordset = set(V)
    features = [list(np.hstack((['Y'],V)))]
    for folder in os.listdir(directory):
        class_val = 1 if folder == "ham" else 0
        for filename in os.listdir(directory+folder+"/"):
            features.append([class_val] + [0]*len(wordset))
            with open(directory+folder+"/"+filename,"r", encoding=encoding) as file:
                for word in file.read().split():
                    if word in wordset:
                        features[-1][features[0].index(word)] += 1
                        
    bow_ds = np.array(features)
    bern_ds = bow_ds.copy()
    bern_ds[1:,:] = np.where(bern_ds[1:,:] > '0' , '1', '0')
    
    if write == 'Y':
        with open(out_bow, "w", newline='', encoding=encoding) as f:
            writer = csv.writer(f, lineterminator = '\n')
            writer.writerows(bow_ds)

        with open(out_bern, "w", newline='', encoding=encoding) as f:
            writer = csv.writer(f, lineterminator = '\n')
            writer.writerows(bern_ds)
    
    return bern_ds, bow_ds

def TrainBernoulliNB(dataset):
    V = dataset[0,1:]
    N = np.size(dataset, 0)-1
    C, count = np.unique(dataset[1:, 0], return_counts = True)
    prior = count/N
    pass
    class0 = dataset[dataset[:,0] == '0']
    class1 = dataset[dataset[:,0] == '1']
    class0_sum = np.sum(class0[:,1:].astype(int), axis=0) + 1
    class1_sum = np.sum(class1[:,1:].astype(int), axis=0) + 1
    condprob = np.concatenate(([class0_sum/(count[0]+2)], [class1_sum/(count[1]+2)]), axis=0)
    return C, V, prior, condprob

def TrainMultinomialNB(dataset):
    V = dataset[0,1:]
    N = np.size(dataset, 0)-1
    C, count = np.unique(dataset[1:, 0], return_counts = True)
    prior = count/N
    class0 = dataset[dataset[:,0] == '0']
    class1 = dataset[dataset[:,0] == '1']
    class0_sum = np.sum(class0[:,1:].astype(int), axis=0) + 1
    class1_sum = np.sum(class1[:,1:].astype(int), axis=0) + 1
    condprob = np.concatenate(([class0_sum/np.sum(class0_sum)], [class1_sum/np.sum(class1_sum)]), axis=0)
    return C, V, prior, condprob

def ApplyMultinomialNB(C, V, prior, condprob, directory):
    correct = 0
    total = 0
    actual = []
    prediction = []
    V = list(V)
    for folder in os.listdir(directory):
        class_val = 1 if folder == "ham" else 0
        for filename in os.listdir(directory+folder+"/"):
            total += 1
            score = np.log(prior)
            with open(directory+folder+"/"+filename,"r", encoding=encoding) as file:
                for word in file.read().split():
                    if word in V:
                        score += np.log(condprob[:,V.index(word)])
            if class_val == np.argmax(score):
                correct += 1
            actual.append(class_val)
            prediction.append(np.argmax(score))
            
    print("ApplyMultinomialNB")
    print(accuracy_score(actual,prediction))
    print(precision_score(actual,prediction))
    print(recall_score(actual,prediction))
    print(f1_score(actual,prediction))
    return correct*100/total

def ApplyBernoulliNB(C, V, prior, condprob, directory):
    correct = 0
    total = 0
    actual = []
    prediction = []
    V = list(V)
    for folder in os.listdir(directory):
        class_val = 1 if folder == "ham" else 0
        for filename in os.listdir(directory+folder+"/"):
            total += 1
            score = np.log(prior)
            with open(directory+folder+"/"+filename,"r", encoding=encoding) as file:
                Vd = file.read().split()
                for word in V:
                    if word in Vd:
                        score += np.log(condprob[:,V.index(word)])
                    else:
                        score += np.log(1-condprob[:,V.index(word)])
            if class_val == np.argmax(score):
                correct += 1
            actual.append(class_val)
            prediction.append(np.argmax(score))
    print("ApplyBernoulliNB")
    print(accuracy_score(actual,prediction))
    print(precision_score(actual,prediction))
    print(recall_score(actual,prediction))
    print(f1_score(actual,prediction))
    return correct*100/total

def exp(W, X):
    return np.exp(np.dot(X,W))

def ProbY0(W, X):
    return 1/(1+exp(W, X))

def ProbY1(W, X):
    return 1-ProbY0(W, X)

def z(W, X):
    return np.dot(X,W)

def LogCondLikelihood(W, X, Y, L=0): 
    return np.sum((Y*z(W, X))+np.log(ProbY0(W, X)))-np.dot(W.T,W)*L/2

def GradientDescent(W, X, Y, L=0.1, n=0.01, iterations=300):
    for i in range(iterations):
        W += n*(np.dot(X.T,(Y-ProbY1(W, X)))-(L*W))
        if i%100 == 0:
            pass
            #print("LL:"+str(LogCondLikelihood(W, X, Y, L)))
            #print("Accy:"+str(Accuracy(W, X, Y)))
    return W

def PredictTrueAs1(W, X):
    return (z(W, X)>0).astype(int)
    
def Accuracy(W, X, Y):
    result = PredictTrueAs1(W, X)
    return np.sum(np.equal(result,Y))/np.size(Y)

def TestLogisticRegression(V, W, directory, ds_type):
    correct = 0
    total = 0
    V = list(V)
    prediction = []
    actual = []
    for folder in os.listdir(directory):
        class_val = 1 if folder == "ham" else 0
        for filename in os.listdir(directory+folder+"/"):
            total += 1
            features = np.zeros((1,np.size(V)+1))
            features[0,0] = 1
            with open(directory+folder+"/"+filename,"r", encoding=encoding) as file:
                for word in file.read().split():
                    if word in V:
                        if ds_type == "Bag-of-words":
                            features[0,V.index(word)+1] += 1
                        else:
                            features[0,V.index(word)+1] = 1
            result = PredictTrueAs1(W, features)
            prediction.append(result[0][0])
            actual.append(class_val)
            correct += 1 if np.equal(result, class_val) else 0
    print("TestLogisticRegression with "+ds_type)
    print(accuracy_score(actual,prediction))
    print(precision_score(actual,prediction))
    print(recall_score(actual,prediction))
    print(f1_score(actual,prediction))
    return correct*100/total
       
if __name__ == '__main__':
    import numpy as np
    import os
    import time
    import csv
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV
    import sys
    
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
        
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    representation = sys.argv[3]
    algo = sys.argv[4]
    encoding = "latin1"
    
    bern_ds, bow_ds = create_datasets(train_dir, encoding, "bow_ds.csv", "bern_ds.csv", 'N')
    
    if algo == "-D":
        C, V, prior, condprob = TrainBernoulliNB(bern_ds)
        ApplyBernoulliNB(C, V, prior, condprob, test_dir)
    elif algo == "-M":
        C, V, prior, condprob = TrainMultinomialNB(bow_ds)
        ApplyMultinomialNB(C, V, prior, condprob, test_dir)
    elif algo == "-L":
        if representation == "-B":
            index=np.where(np.round(np.sum(bern_ds[1:,:].astype(int), axis=0),1)<bern_ds[1:,:].shape[0]*1.0)[0]
            bern_ds = bern_ds[:,index]
            X=np.array(bern_ds[1:,1:]).astype(int)
            X0=np.ones((X.shape[0],1), dtype=int)
            X = np.hstack((X0, X))
            Y=np.array(bern_ds[1:,0]).astype(int)
            Y.shape=(X.shape[0],1)
            V = bern_ds[0,1:]
            W=np.zeros((X.shape[1], 1))
            W = GradientDescent(W, X, Y)
            TestLogisticRegression(V, W, test_dir, "Bernoulli")  
        else:
            index=np.where(np.round(np.sum(bern_ds[1:,:].astype(int), axis=0),1)<bern_ds[1:,:].shape[0]*1.0)[0]
            bow_ds = bow_ds[:,index]
            X=np.array(bow_ds[1:,1:]).astype(int)/10
            X0=np.ones((X.shape[0],1), dtype=int)
            X = np.hstack((X0, X))
            Y=np.array(bow_ds[1:,0]).astype(int)
            Y.shape=(X.shape[0],1)
            V = bow_ds[0,1:]
            W=np.zeros((X.shape[1], 1))
            W = GradientDescent(W, X, Y)
            TestLogisticRegression(V, W, test_dir, "Bag-of-words")   
    else:
        if representation == "-B":
            X=np.array(bern_ds[1:,1:]).astype(int)
            Y=np.array(bern_ds[1:,0]).astype(int)
            Y.shape=(X.shape[0],1)
            V = bern_ds[0,1:]
            sgdc = SGDClassifier()
            params = {"alpha": [ 0.001, 0.01, 0.1, 1],
                      "max_iter": [50, 100, 300],
                      "tol" : [1e-1, 1e-2, 1e-3],
                      "penalty": ["l2"],
                      "loss" : ["log"]}
            grid_search = GridSearchCV(sgdc, param_grid=params)
            grid_search.fit(X, Y) 
            print(grid_search.best_estimator_)
            
            bern_ds_test, bow_ds_test = create_test_datasets(test_dir, V, encoding, "bow_ds.csv", "bern_ds.csv", 'N')
            X_test = np.array(bern_ds_test[1:,1:]).astype(int)
            Y_test = np.array(bern_ds_test[1:,0]).astype(int)
            
            prediction = grid_search.predict(X_test)
            actual = Y_test
            print("SGD with Bernoulli")
            print(accuracy_score(actual,prediction))
            print(precision_score(actual,prediction))
            print(recall_score(actual,prediction))
            print(f1_score(actual,prediction))
        else:
            X=np.array(bow_ds[1:,1:]).astype(int)
            Y=np.array(bow_ds[1:,0]).astype(int)
            Y.shape=(X.shape[0],1)
            V = bow_ds[0,1:]
            sgdc = SGDClassifier()
            params = {"alpha": [ 0.0001, 0.001, 0.01, 0.1, 1],
                      "max_iter": [100, 300, 500],
                      "tol" : [1e-1, 1e-2, 1e-3],
                      "penalty": ["l2"],
                      "loss" : ["log"]}
            grid_search = GridSearchCV(sgdc, param_grid=params)
            grid_search.fit(X, Y) 
            print(grid_search.best_estimator_)
            
            bern_ds_test, bow_ds_test = create_test_datasets(test_dir, V, encoding, "bow_ds.csv", "bern_ds.csv", 'N')
            X_test = np.array(bow_ds_test[1:,1:]).astype(int)
            Y_test = np.array(bow_ds_test[1:,0]).astype(int)
            
            prediction = grid_search.predict(X_test)
            actual = Y_test
            print("SGD with Bag-of-words")
            print(accuracy_score(actual,prediction))
            print(precision_score(actual,prediction))
            print(recall_score(actual,prediction))
            print(f1_score(actual,prediction))