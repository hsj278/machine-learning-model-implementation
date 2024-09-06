from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import Linear_Regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
import argparse

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import Logistic_Regression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import Decision_Tree
from sklearn.datasets import load_iris

parser = argparse.ArgumentParser(description="Choose which model you want to test")
parser.add_argument('-m', '--model', type=str, help='model you want to use')
args = parser.parse_args()

regressions = ['linear_regression','regression_tree']
classifications = ['logistic_regression']
decision_tree = ['decision_tree']

if args.model in regressions:
    X, y = make_regression(n_samples=1000, n_features=10, noise=20)
elif args.model in classifications:
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=20)
elif args.model in decision_tree:
    iris = load_iris()
    X, y = iris.data, iris.target
    
# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if args.model == 'linear_regression':    
    # Initialize and train model
    custom_model = Linear_Regression.LinearRegression(learning_rate=0.01, iterations=1000, regularization='L2', alpha=0.01, loss_function='MSE', fit_method='GD')
    custom_model.fit(X_train, y_train)

    LR_model = Ridge(alpha = 0.01)
    LR_model.fit(X_train, y_train)

    custom_no_reg = Linear_Regression.LinearRegression(learning_rate=0.01, iterations=1000, regularization = None, alpha=0.01, loss_function='MSE', fit_method='GD')
    custom_no_reg.fit(X_train, y_train)

    LR_no_reg = LinearRegression()
    LR_no_reg.fit(X_train, y_train)
    # Make predictions
    predictions = custom_model.predict(X_test)
    LR_pred = LR_model.predict(X_test)

    no_reg_pred = custom_no_reg.predict(X_test)
    LR_no_reg_pred = LR_no_reg.predict(X_test)

    # Evaluate model
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, predictions)
    LR_mse = mean_squared_error(y_test, LR_pred)
    no_reg_mse = mean_squared_error(y_test, no_reg_pred)
    LR_no_reg_mse = mean_squared_error(y_test, LR_no_reg_pred)
    print(f"Mean Squared Error on test data for custom model: {mse}")
    print(f"Mean Squared Error on test data for Ridge regression in sklearn: {LR_mse}")
    print(f"Mean Squared Error on test data for custom no reg: {no_reg_mse}")
    print(f"Mean Squared Error on test data for linear regression in sklearn: {LR_no_reg_mse}")

elif args.model == 'logistic_regression':
    custom_model = Logistic_Regression.LogisticRegression(learning_rate = 0.01, iterations=1000, regularization=None, alpha=0.01, loss_function='BCE', fit_method='GD')
    custom_model.fit(X_train, y_train)
    
    Log_model = LogisticRegression(random_state=42)
    Log_model.fit(X_train, y_train)
    
    custom_pred = custom_model.predict(X_test)
    Log_pred = Log_model.predict(X_test)

    custom_accuracy = accuracy_score(y_test, custom_pred)
    Log_accuracy = accuracy_score(y_test, Log_pred)
    print(f"Custom Accuracy: {custom_accuracy}")
    print(f"Logistic Regression Accuracy: {Log_accuracy}")

    custom_conf_matrix = confusion_matrix(y_test, custom_pred)
    Log_conf_matrix = confusion_matrix(y_test, Log_pred)
    print("Confusion Matrix:")
    print(custom_conf_matrix)
    print(Log_conf_matrix)
    
    custom_class_report = classification_report(y_test, custom_pred)
    Log_class_report = classification_report(y_test, Log_pred)
    print("Classification Report:")
    print(custom_class_report)
    print(Log_class_report)
    
elif args.model == 'decision_tree':
    custom_tree = Decision_Tree.DecisionTree(max_depth=10)#, regression=False, threshold_method='unique')
    custom_tree.fit(X_train, y_train)
    
    custom_pred = custom_tree.predict(X_test)

    DT_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    DT_tree.fit(X_train, y_train)
    
    DT_pred = DT_tree.predict(X_test)
    
    custom_accuracy = accuracy_score(y_test, custom_pred)
    DT_accuracy = accuracy_score(y_test, DT_pred)
    
    print(f"Custom accuracy: {custom_accuracy}")
    print(f"DecisionTreeClassifier accuracy: {DT_accuracy}")
    
elif args.model == 'regression_tree':
    custom_tree = Decision_Tree.DecisionTree(max_depth=10, regression=True, threshold_method='unique')
    custom_tree.fit(X_train, y_train)
    
    custom_pred = custom_tree.predict(X_test)
    
    DT_tree = DecisionTreeRegressor(max_depth=10, random_state=42)
    DT_tree.fit(X_train, y_train)
    
    DT_pred = DT_tree.predict(X_test)
    
    custom_mse = mean_squared_error(y_test, custom_pred)
    mse = mean_squared_error(y_test, DT_pred)
    
    print(f"Custom MSE: {custom_mse}")
    print(f"DecisionTreeRegressor MSE: {mse}")