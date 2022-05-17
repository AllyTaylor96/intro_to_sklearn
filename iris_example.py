"""
This is an example classification problem completed using the
tutorial in the link. It looks at different ways of classifying the classic
Iris dataset (both supervised and unsupervised).
https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html
"""

# import appropriate packages
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load example iris dataset as dataframe (150 x 5 size) with various attributes
# of the flowers (sepal_length, petal_length etc.) and which species
iris = sns.load_dataset('iris')

# take the four attributes in the X_iris (as a features matrix)
X_iris = iris.drop('species', axis=1)  # size (150, 4)
# take the species axis as a target array (what we are classifying around)
y_iris = iris['species'] # size (150, )

"""
Now the task formally is: we are trying to build a model that if trained on a 
portion of the Iris dataset, can it correctly classify the remaining labels?

To do this, we'll use the ultra-simple Gaussian naive Bayes model. First,
we need to split the data into test and train groups. Then, we follow the
recipe specified in sklearn's documentation:
1. Choose class of model + import appropriate estimator class from sklearn.
2. Choose model hyperparams by instantiating class with appropriate values.
3. Arrange data into features matrix + target vector.
4. Fit model to data by calling fit() method of model instance.
5. Apply model to new data (for supervised learning, this is usually predict():
for unsupervised, oftentimes data is transformed or inferred using either 
transform() or predict().
"""

# step 4 - use train_test_split function to automatically split the data
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris,
                                                random_state=1)

# step 1 - import chosen model (Gaussian)
from sklearn.naive_bayes import GaussianNB

# step 2 - instantiate model with hyperparams (none needed here)
model = GaussianNB()

# step 3 - fit to data (train on test data with supervised data)
model.fit(X_train, y_train)

# step 4 - predict it on new data
y_model = model.predict(X_test)

# can use accuracy score utility to see amount of predicted labels that match
# their true value
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_model))  # 97%

"""
We can see supervised training is yielding almost 97% accuracy, which is great.

Let's now try unsupervised training and see if it will also work. The first 
solution we will try is reducing the dimensionality of the data (at present, it
has four features for each sample). If we reduced the dimensions of the data to
2D, we could plot the data on a graph!

We can use principal component analysis (PCA) to return a 2-component 
representation of the data.
"""

from sklearn.decomposition import PCA  # import model
unsup_model = PCA(n_components=2)  # instantiate model with 2 components
unsup_model.fit(X_iris)  # fit to data (note not using y data)
X_2D = unsup_model.transform(X_iris)  # transform data to 2D

# insert results into original iris dataframe
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]

# plot using lmplot
graph = sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)
plt.savefig('graphs/reduce_dim_graph.png')

"""
The graph shows a good separation between the different species, even without
using the labelled data in training.

The next task is to try using clustering to find distinct groups in the data.
We will use a GMM here to do this.
"""

from sklearn.mixture import GaussianMixture  # import GMM model
gmm_model = GaussianMixture(n_components=3,
            covariance_type='full')  # instantiate model as class
gmm_model.fit(X_iris)  # again fit to the data, not using y
y_gmm = gmm_model.predict(X_iris)  # determine cluster labels

# add cluster label to iris dataframe and use seaborn
iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='species',
           col='cluster', fit_reg=False)
plt.savefig('graphs/clustering_graph.png')

"""
Examining the graph shows that the Setosa species has been perfectly separated
in the first cluster, while the Versicolor and Virginica have small amounts of 
overlap. This highlights how we could automatically identify the different
groups of species with the clustering algorithm.
"""