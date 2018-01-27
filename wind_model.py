from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier


# The prediction model for the wind
# Takes as input the wind forecasts and outputs a new prediction
# The prediction will be used by the pathfinding algorithm
# Usually, the output would be the probability of being over 15 but it
# can be anything as long as the algo knows how to use it

model = LogisticRegression()
modelName = 'Logit'

# The threshold that will be used to plot the predictions
# in the comparison (on the map and on the graph)
evalThreshold = 0.5

# None if no PCA is to be performed, otherwise the number of components to keep
red = None
