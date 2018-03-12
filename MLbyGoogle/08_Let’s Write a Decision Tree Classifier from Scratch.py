
# coding: utf-8

# In[16]:


# Dataset
training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon']
]


# In[17]:


# These are used only to pring the tree.
header = ["color", "diamete", "label"]


# In[18]:


# Find the unique values from a column in  a datase.
def unique_vals(rows, col):
    # In set dataset it will not store duplicate values.
    return set([row[col] for row in rows])


# In[19]:


# Test and see how this function works.
unique_vals(training_data, 0)


# In[46]:


class Decision_Node:
    """ A Decition Node asks a 
        This holds a reference to the question, and to the two child nodes.
    """
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


# In[21]:


def info_gain(left, right, current_uncertainty):
    """ Information Gain.
        The uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
    """
    p = float(len(left) / len(left) + len(right))  # Don't know what does this mean, and why?
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


# In[238]:


# In[22]:


def partition(rows, question):
    """ Partition the dataset.        
        For each row in the dataset, Check if Question's answer is True,
        then add it to 'true_rows', else add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


# In[23]:


def is_numeric(value):
    # Check if a values is numeric or not.
    return isinstance(value, int) or isinstance(value, float)


# In[24]:


class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question."""
    
    def __init__(self, column, value):
        self.column = column
        self.value = value
    
    def match(self, example):
        # Compare the freature value in an example to the feature value in the question.
        val = example[self.column]  # I don't know how this line works?, specialy "example".
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper mehod to print the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" %(header[self.column], condition, str(self.value))


# In[25]:


def class_counts(rows):
    # Counts the bumber of each type of example in dataset.
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts
    


# In[26]:


# See the out of "class_counts()"function.
class_counts(training_data)


# In[27]:


def gini(rows):
    # Calculate the Gini Ipurity for a list or rows.
    # there are many ways to find the Gini. 
    # See: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


# In[44]:


def find_best_split(rows):
    # Find the best question to ask by iterating over every feature/value
    # and calculating the information gain.
    best_gain = 0  # keep track of the best infromration gain.
    best_question = None  # keep train of the feature/value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # gives the number of columns except "label column".
    
    for col in range (n_features):  # for each feature
        # unique values in the column
        values = set([row[col]for row in rows])        
        # for each value
        
        for val in values:
            question = Question(col, val)            
            # split the dataset as an anser of "Question".
            true_row, false_row = partition(rows, question)            
            
            # skip this split if it doesn't divide the dataset.
            if len(true_row) == 0 or len(false_row) == 0:
                continue
            
            # Calculate the information gain from this split
            gain = info_gain(true_row, false_row, current_uncertainty)
            
            # can be use '>' instead of '>=' here, used '>' to look the tree certain way for dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question


# In[59]:


class Leaf:
    """ A Leaf node classifies data.
        This holds a dictionary of class (e.g., "Apple") ->number of times
        it appears in the rows from the training data that reach this leaf.
    """
    def __init__(self, rows):
        self.predictions = class_counts(rows)


# In[30]:


# Build the tree.
def build_tree(rows):
    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)
    
    
    # Since there ara no more questons, going to return the leaf.
    if gain == 0:
        return Leaf(rows)
    
    # now we can find useful feature/value to partition on.
    true_rows, false_rows = partition(rows, question)
    
    # Recursively build the true branch.
    true_branch = build_tree(true_rows)
    
    # Recursively build the false branch.
    false_branch = build_tree(false_rows)
    
    # Return a Question node.
    # This records the best feature/value to ask at this pont, as well as the branches to follow depending on the ansewr.
    return Decision_Node(question, true_branch, false_branch)


# In[61]:


def print_tree(node, spacing=""):
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return
    
    # Print the question at this node
    print(spacing + str(node.question))
    
    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")
    
    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


# In[47]:


my_tree = build_tree(training_data)


# In[63]:


print_tree(my_tree)

