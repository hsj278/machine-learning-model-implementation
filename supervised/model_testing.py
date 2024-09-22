from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import argparse

from sklearn.datasets import make_classification

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser(description="Choose which model you want to test")
parser.add_argument('-m', '--model', type=str, help='model you want to use')
parser.add_argument('-b', '--boost', type=str, help='boost method you want to use')
args = parser.parse_args()

regressions = ['linear_regression','regression_tree','random_forest_reg','KNN_reg']
classifications = ['logistic_regression','decision_tree','random_forest_cls','KNN_cls','SVM']

boosts_reg = ['adaboost_reg']
boosts_cls = ['adaboost_cls']

if args.model in regressions:
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
elif args.model in classifications:
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, n_clusters_per_class=1,flip_y=0.1,class_sep=0.5,random_state=42)
    
# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if args.model == 'linear_regression':    
    import Linear_Regression
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    
    if args.boost == 'adaboost_reg':
        import AdaBoost
        from sklearn.ensemble import AdaBoostRegressor
        
        custom_adaboost = AdaBoost.AdaBoost(n_estimators=50, max_depth=1, learning_rate=0.01, regression=True, learner='linear_regression')
        custom_adaboost.fit(X_train, y_train)
        custom_pred = custom_adaboost.predict(X_test)
        
        ABR = AdaBoostRegressor(base_estimator=LinearRegression(), n_estimators=50, learning_rate=0.01, random_state=42)
        ABR.fit(X_train, y_train)
        ABR_pred = ABR.predict(X_test)
        
        custom_accuracy = mean_squared_error(y_test, custom_pred)
        ABR_accuracy = mean_squared_error(y_test, ABR_pred)
        
        print(f'Custom mse: {custom_accuracy}')
        print(f'AdaBoost Regressor mse: {ABR_accuracy}')
    else:        
        # Initialize and train model
        custom_model = Linear_Regression.LinearRegression(learning_rate=0.01, iterations=1000, regularization='L2', alpha=0.01/X_train.shape[0], loss_function='MSE', fit_method='GD')
        custom_model.fit(X_train, y_train)

        LR_model = Ridge(alpha = 0.01)
        LR_model.fit(X_train, y_train)

        custom_no_reg = Linear_Regression.LinearRegression(learning_rate=0.01, iterations=1000, regularization = None, alpha=0.01/X_train.shape[0], loss_function='MSE', fit_method='GD')
        custom_no_reg.fit(X_train, y_train)

        LR_no_reg = LinearRegression()
        LR_no_reg.fit(X_train, y_train)
        # Make predictions
        predictions = custom_model.predict(X_test)
        LR_pred = LR_model.predict(X_test)

        no_reg_pred = custom_no_reg.predict(X_test)
        LR_no_reg_pred = LR_no_reg.predict(X_test)

        # Evaluate model
        mse = mean_squared_error(y_test, predictions)
        LR_mse = mean_squared_error(y_test, LR_pred)
        no_reg_mse = mean_squared_error(y_test, no_reg_pred)
        LR_no_reg_mse = mean_squared_error(y_test, LR_no_reg_pred)
        print(f"Mean Squared Error on test data for custom Ridge regression model: {mse}")
        print(f"Mean Squared Error on test data for Ridge regression in sklearn: {LR_mse}")
        print(f"Mean Squared Error on test data for custom regression with no regularization: {no_reg_mse}")
        print(f"Mean Squared Error on test data for linear regression in sklearn: {LR_no_reg_mse}")

elif args.model == 'logistic_regression':
    import Logistic_Regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    import Decision_Tree
    from sklearn.tree import DecisionTreeClassifier

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
    import Decision_Tree
    from sklearn.tree import DecisionTreeRegressor


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
    
elif args.model == 'random_forest_cls':
    import Random_Forest
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    custom_forest = Random_Forest.RandomForest(n_tree=10, max_depth=5, n_features=5)
    custom_forest.fit(X_train, y_train)
    custom_pred = custom_forest.predict(X_test)
    
    RF = RandomForestClassifier(n_estimators=10, max_depth=5, max_features=5, random_state=42)
    RF.fit(X_train,y_train)
    RF_pred = RF.predict(X_test)
        
    custom_accuracy = accuracy_score(y_test, custom_pred)
    RF_accuracy = accuracy_score(y_test, RF_pred)
    
    print(f'Custom accuracy: {custom_accuracy}')
    print(f'Random Forest accuracy: {RF_accuracy}')
    
elif args.model == 'random_forest_reg':
    import Random_Forest
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    
    custom_forest = Random_Forest.RandomForest(n_tree=10, max_depth=5, n_features=5, regression=True)
    custom_forest.fit(X_train, y_train)
    custom_pred = custom_forest.predict(X_test)
    
    RF = RandomForestRegressor(n_estimators=10, max_depth=5, max_features=5, random_state=42)
    RF.fit(X_train,y_train)
    RF_pred = RF.predict(X_test)
        
    custom_accuracy = mean_squared_error(y_test, custom_pred)
    RF_accuracy = mean_squared_error(y_test, RF_pred)
    
    print(f'Custom MSE: {custom_accuracy}')
    print(f'Random Forest MSE: {RF_accuracy}')
    
elif args.model == 'KNN_cls':
    import KNN
    from sklearn.neighbors import KNeighborsClassifier
    
    custom_KNN = KNN.KNN(k=3,dist='euclidean',regression=False)
    custom_KNN.fit(X_train,y_train)
    custom_pred = custom_KNN.predict(X_test)
    
    KNC = KNeighborsClassifier(n_neighbors=3)
    KNC.fit(X_train,y_train)
    KNC_pred = KNC.predict(X_test)
 
    custom_accuracy = accuracy_score(y_test, custom_pred)
    KNC_accuracy = accuracy_score(y_test, KNC_pred)
    
    print(f'Custom accuracy: {custom_accuracy}')
    print(f'KNN Classifier accuracy: {KNC_accuracy}')
    
elif args.model == 'KNN_reg':
    import KNN
    from sklearn.neighbors import KNeighborsRegressor
    
    custom_KNN = KNN.KNN(k=3,dist='euclidean',regression=True)
    custom_KNN.fit(X_train,y_train)
    custom_pred = custom_KNN.predict(X_test)
    
    KNR = KNeighborsRegressor(n_neighbors=3)
    KNR.fit(X_train,y_train)
    KNR_pred = KNR.predict(X_test)

    custom_mse = mean_squared_error(y_test, custom_pred)
    KNR_mse = mean_squared_error(y_test, KNR_pred)
    
    print(f'Custom MSE: {custom_mse}')
    print(f'KNN Classifier MSE: {KNR_mse}')
    
elif args.model == 'adaboost_cls':
    import AdaBoost
    from sklearn.ensemble import AdaBoostClassifier
    
    custom_adaboost = AdaBoost.AdaBoost(n_estimators=50, max_depth=1)
    custom_adaboost.fit(X_train, y_train)
    custom_pred = custom_adaboost.predict(X_test)
    
    ABC = AdaBoostClassifier(n_estimators=50, random_state=42)
    ABC.fit(X_train, y_train)
    ABC_pred = ABC.predict(X_test)
    
    custom_accuracy = accuracy_score(y_test, custom_pred)
    ABR_accuracy = accuracy_score(y_test, ABC_pred)
    
    print(f'Custom accuracy: {custom_accuracy}')
    print(f'AdaBoost Classifier accuracy: {ABR_accuracy}')
    
elif args.model == 'adaboost_reg':
    import AdaBoost
    from sklearn.ensemble import AdaBoostRegressor
    
    custom_adaboost = AdaBoost.AdaBoost(n_estimators=50, max_depth=1, learning_rate=0.1, regression=True)
    custom_adaboost.fit(X_train, y_train)
    custom_pred = custom_adaboost.predict(X_test)
    
    ABR = AdaBoostRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
    ABR.fit(X_train, y_train)
    ABR_pred = ABR.predict(X_test)
    
    custom_accuracy = mean_squared_error(y_test, custom_pred)
    ABR_accuracy = mean_squared_error(y_test, ABR_pred)
    
    print(f'Custom mse: {custom_accuracy}')
    print(f'AdaBoost Regressor mse: {ABR_accuracy}')
    
elif args.model == 'SVM':
    import SVM
    from sklearn.svm import SVC
    
    custom_linear_SVM = SVM.SVM(C=1.0, kernel='linear')
    custom_linear_SVM.fit(X_train,y_train)
    custom_linear_SVM_pred = custom_linear_SVM.predict(X_test)
    
    custom_poly_SVM = SVM.SVM(C=1.0, kernel='poly', degree=3, gamma=1, coef0=1)
    custom_poly_SVM.fit(X_train,y_train)
    custom_poly_SVM_pred = custom_poly_SVM.predict(X_test)
    
    custom_rbf_SVM = SVM.SVM(C=1.0, gamma=2, kernel='rbf')
    custom_rbf_SVM.fit(X_train,y_train)
    custom_rbf_SVM_pred = custom_rbf_SVM.predict(X_test)
    
    custom_linear_SVM_acc = accuracy_score(y_test, custom_linear_SVM_pred)
    custom_poly_SVM_acc = accuracy_score(y_test, custom_poly_SVM_pred)
    custom_rbf_SVM_acc = accuracy_score(y_test, custom_rbf_SVM_pred)
    
    
    SVC_linear = SVC(C=1.0, kernel='linear')
    SVC_linear.fit(X_train,y_train)
    SVC_linear_pred = SVC_linear.predict(X_test)    

    SVC_poly = SVC(C=1.0, kernel='poly', degree=3, gamma=1, coef0=1)
    SVC_poly.fit(X_train,y_train)
    SVC_poly_pred = SVC_poly.predict(X_test)
    
    SVC_rbf = SVC(C=1.0, kernel='rbf', gamma=2)
    SVC_rbf.fit(X_train,y_train)
    SVC_rbf_pred = SVC_rbf.predict(X_test)
    
    SVC_linear_acc = accuracy_score(y_test, SVC_linear_pred)
    SVC_poly_acc = accuracy_score(y_test, SVC_poly_pred)
    SVC_rbf_acc = accuracy_score(y_test, SVC_rbf_pred)    
    
    
    print(f'Custom linear SVM accuracy: {custom_linear_SVM_acc}')
    print(f'Sklearn linear SVC accuracy: {SVC_linear_acc}')
    print(f'Custom polynomial SVM accuracy: {custom_poly_SVM_acc}')
    print(f'Sklearn polynomial SVC accuracy: {SVC_poly_acc}')
    print(f'Custom rbf SVM accuracy: {custom_rbf_SVM_acc}')
    print(f'Sklearn rbf SVC accuracy: {SVC_rbf_acc}')
    