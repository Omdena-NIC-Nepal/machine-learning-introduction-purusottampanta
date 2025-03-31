import data_preprocessing

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = data_preprocessing.preprocessAndSplit()

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation on training set
train_score = model.score(X_train, y_train)
print(f"Training R-squared: {train_score:.2f}")

