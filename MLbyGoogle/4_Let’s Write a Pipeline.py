# import a dataset
from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

# split the data by half
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)


# use models to classifie
"""
from sklearn import tree
my_classifire = tree.DecisionTreeClassifier()
"""

# change the model to done the work.
from sklearn.neighbors import KNeighborsClassifier
my_classifire = KNeighborsClassifier()

my_classifire.fit(x_train, y_train)

predictions = my_classifire.predict(x_test)


# mesure the accuracy of predication
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))