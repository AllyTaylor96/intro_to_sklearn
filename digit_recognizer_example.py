"""
This is an example classification problem completed using the
tutorial in the link. It looks at the identification of hand-written digits.
https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html
"""

# import packages
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# first load in the digit-recognition dataset from sklearn
from sklearn.datasets import load_digits
digits = load_digits()  # shape (1797, 8, 8)

"""
The dataset consists of 1797 samples of hand-drawn digits, consisting of 8x8 
pixels each. To work with this in sklearn, we need to reduce it to a 2D 
[n_samples, n_features] representation. We can do this by flattening out the 
pixel arrays into a 64-long array of values (so every pixel is a feature).
This representation is built into the digit dataset by default.
"""

X = digits.data  # flattens pixel rep so data size is now (1797, 64)
y = digits.target  # gets target so shape is (1797, )

"""
A good first step is to visualize the data on a 2D graph. We can reduce the 
dimensionality by using manifold learning algorithm 'Isomap' to transform the
data to 2D.
"""

from sklearn.manifold import Isomap  # import model
iso = Isomap(n_components=2)  # instantiate model with hyperparams
iso.fit(digits.data)  # fit to data
data_projected = iso.transform(digits.data)  # projects data to 2d repr

# data is now shape (1797, 2) so can plot and have a look
plt.figure(1)
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('rainbow', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)
plt.savefig('graphs/digits_2d_repr.png')

"""
By peeking at the projected data, we can see there is some clear division in
how the numbers are divided in the 64-dimensional space. This means that a
simple supervised classification algorithm should be able to work.

We will do this in a similar way to the Iris example, splitting the data into 
training and testing sets and fitting a naive Gaussian model.
"""

# split data into respective segments
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.naive_bayes import GaussianNB  # import Gaussian model
model = GaussianNB()  # instantiate model with hyperparams
model.fit(X_train, y_train)  # fit model using supervised data
y_model = model.predict(X_test)  # predict on test data

# use accuracy score to see how it performed
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_model))  # 83%

"""
This is working relatively well, but it isn't telling us where it went wrong. 
We can try to figure this out by using a calculation matrix.
"""

# import and calculate confusion matrix for our model
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_model)

# plot the matrix using matplotlib and seaborn
plt.figure(2)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.savefig('graphs/confusion_matrix.png')

"""
The matrix shows us where it went wrong: for eg. a number of twos have been 
falsely classified as either ones or eights. This helps explain where the
ambiguity is coming from and allows us to focus on how to improve our model.
"""