from scipy.spatial import distance


def euc(a, b):
	return distance.euclidean(a, b)  # find out how does it mesures the distance between features!


class ScrappyKNN():
	def fit(self, x_train, y_train):  # "fit" means store the data and labels.
		self.x_train = x_train
		self.y_train = y_train

	# "predict" meanse compare the test(example) data with stored data and with most the example data matches,
	#  retern that stored data's(features) label.
	def predict(self, x_test):
		predictions = []
		for row in x_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions

	def closest(self, row):
		best_dist = euc(row, self.x_train[0])
		best_index = 0
		for i in range(1, len(self.x_train)):
			dist = euc(row, self.x_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.y_train[best_index]

# import a dataset
from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

# split the data by half
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)


# change the model to done the work.
#from sklearn.neighbors import KNeighborsClassifier
my_classifire = ScrappyKNN()

my_classifire.fit(x_train, y_train)

predictions = my_classifire.predict(x_test)


# mesure the accuracy of predication
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))