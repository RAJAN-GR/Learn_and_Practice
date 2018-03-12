import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree


iris = load_iris()
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]


clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)  # Train the dada.

#print(test_target)  # testing data's lable
#print(clf.predict(test_data))  # predicted data's lable.

#print(test_data[1], test_target[1])
#print(iris.feature_names, iris.target_names)



"""
print("IrisFeatureNames:", iris.feature_names)
print("Target Names(Lable Names):", iris.target_names)
print(iris.data[0])
print(iris.target[0])
"""