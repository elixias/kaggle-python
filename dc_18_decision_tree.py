# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier
# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = DecisionTreeClassifier(max_depth=6, random_state=SEED)
# Fit dt to the training set
dt.fit(X_train, y_train)
# Predict test set labels
y_pred = dt.predict(X_test)
print(y_pred[0:5])

# Import accuracy_score
from sklearn.metrics import accuracy_score
# Predict test set labels
y_pred = dt.predict(X_test)
# Compute test set accuracy  
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))

"""custom plotting of boundaries for classifiers"""
# Import LogisticRegression from sklearn.linear_model
from sklearn.linear_model import LogisticRegression
# Instatiate logreg
logreg = LogisticRegression(random_state=1)
# Fit logreg to the training set
logreg.fit(X_train, y_train)
# Define a list called clfs containing the two classifiers logreg and dt
clfs = [logreg, dt]
# Review the decision regions of the two classifiers
plot_labeled_decision_regions(X_test, y_test, clfs)

#tree tries to maximise information gain (IG) at each split
#IG(f, sp) = I(parent) - (Nl/Nt * I(left) + Nr/Nt * I(right))
#I is impurity of the node (gini index or entropy)
#If at node the information gain IG(node) is 0, it is declared as a leaf
#DecisionTreeClassifier(criterion='gini'...)
# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier
# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=1)
# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)

# Import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score
# Use dt_entropy to predict test set labels
y_pred= dt_entropy.predict(X_test)
# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_test,y_pred)
# Print accuracy_entropy
print('Accuracy achieved by using entropy: ', accuracy_entropy)
# Print accuracy_gini
print('Accuracy achieved by using the gini index: ', accuracy_gini)


#min_samples_leaf=0.1 means each leaf should contain at least 10% of training data

#RMSE for computing impurity of a node
MSE(node) = 1/Nnode sum of impurities across all nodes

"""tree for regression"""
# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor
# Instantiate dt
dt = DecisionTreeRegressor(max_depth=8,
             min_samples_leaf=0.13,
            random_state=3)
# Fit dt to the training set
dt.fit(X_train, y_train)

"""calculating the RMSE"""
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE
# Compute y_pred
y_pred = dt.predict(X_test)
# Compute mse_dt
mse_dt = MSE(y_test,y_pred)
# Compute rmse_dt
rmse_dt = mse_dt**0.5
# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))

"""comparison with Linear Regression"""
# Predict test set labels 
y_pred_lr = lr.predict(X_test)
# Compute mse_lr
mse_lr = MSE(y_test,y_pred_lr)
# Compute rmse_lr
rmse_lr = mse_lr**0.5
# Print rmse_lr
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))
# Print rmse_dt
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))

"""generalization error OF f_hat = bias^2+variance+irreducible_error_(noise)"""

#high bias = underfitting
#high variance = overfitting
#bias & variance tradeoff
#model complexity / flexibility of f_hat
#complexity increases, bias decreases and variance increases

###generalisation error of f_hat is _approximated_ by test set error of f_hat
#solution: CV KFold CV (mean of each fold's error) or Holdout CV
#use the CV KFold err and compare with training set error
#higher ~ high variance > overfitting has happened > decrease model complexity ie decrease max depth, inc max samples per leaf, or gather more data
#if CVFold err approx training err and larger than desired error, then bias problem or underfitting. inc model complexity or getting more relevant features

# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split
# Set SEED for reproducibility
SEED = 1
# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
# Instantiate a DecisionTreeRegressor dt
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=SEED)

"""!Note that since cross_val_score has only the option of evaluating the negative MSEs, its output should be multiplied by negative one to obtain the MSEs."""
# Compute the array containing the 10-folds CV MSEs
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10, 
                       scoring='neg_mean_squared_error',
                       n_jobs=-1) #use all cores
# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean())**(0.5)
# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))
"""Note: Perform CV on the train set and still withholding the test set. In the past we perform CV on the entire set without splitting"""

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE
# Fit dt to the training set
dt.fit(X_train, y_train)
# Predict the labels of the training set
y_pred_train = dt.predict(X_train)
# Evaluate the training set RMSE of dt
RMSE_train = (MSE(y_train, y_pred_train))**(0.5)
# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))

"""Tree Ensemble"""
#CART sensitive to small variations in datasets, suffers from high variance (overfitting)
#Solution is : ensemble learning, voting classifiers - hard voting
# Set seed for reproducibility
SEED=1
# Instantiate lr
lr = LogisticRegression(random_state=SEED)
# Instantiate knn
knn = KNN(n_neighbors=27)
# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)
# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]
# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:    
    # Fit clf to the training set
    clf.fit(X_train, y_train)    
    # Predict y_pred
    y_pred = clf.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) 
    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))

# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier
# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)     
# Fit vc to the training set
vc.fit(X_train, y_train)   
# Evaluate the test set predictions
y_pred = vc.predict(X_test)
# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Voting Classifier: {:.3f}'.format(accuracy))

"""bootstrap aggregation"""
#or bagging, uses the same algorithm
#but not trained on the entire trainingset but on a subset of training
#reduces variance of individual model in ensemble
#***draws samples with replacement*** from your existing data
#quite different from CV which partitions
#draws n different samples, used to train m models using same algorithm
#BaggingClassifier - classification uses voting
#BaggingRegressor - regression uses aggregation
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier
# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)
# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)

# Fit bc to the training set
bc.fit(X_train, y_train)
# Predict test set labels
y_pred = bc.predict(X_test)
# Evaluate acc_test
acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of bc: {:.2f}'.format(acc_test)) 

"""some data may not be sampled at all, they become Out of Bag (OOB) instances"""
#they can be used to validate the model (thus not having to use cross validation)
#this is called OOB evaluation
#OOB score is the average of the OOB scores for each model in the ensemble
#set parameter oob_score to True, and extract oob_score_
#classifiers uses accuracy score
#regressors uses R^2
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier
# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=8, random_state=1)
# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, 
            n_estimators=50,
            oob_score=True,
            random_state=1)
# Fit bc to the training set 
bc.fit(X_train, y_train)
# Predict test set labels
y_pred = bc.predict(X_test)
# Evaluate test set accuracy
acc_test = accuracy_score(y_test, y_pred)
# Evaluate OOB accuracy
acc_oob = bc.oob_score_
# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))

"""random tree, uses decision tree as estimator"""
#each estimator trained on a different bootstrap sample (of same size as training set)
#introduces randomization
#d features sampled at each node without replacement
#ie: each estimator learn using different subsets(independent variable) of the training data
#RandomForestClassifier and RandomForestRegressor
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# Instantiate rf
rf = RandomForestRegressor(n_estimators=25,
            random_state=2)
# Fit rf to the training set    
rf.fit(X_train, y_train) 
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE
# Predict the test set labels
y_pred = rf.predict(X_test)
# Evaluate the test set RMSE
rmse_test = MSE(y_test,y_pred).mean()**0.5
# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)
# Sort importances
importances_sorted = importances.sort_values()
# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()

"""ada boosting"""
#combining weak learners (only slightly better than random guessing i.e: tree stump with depth 1) to form strong learner
#predictors are trained sequentially, trying to correct its predecessor
#AdaBoost (Adaptive Boost) and GradientBoost
#n predictors: the error from 1st is passed as input to the 2nd so on so forth, with more weightage on those that are incorrect
#learning rate (if low), compensate with more predictors
#AdaBoostClassifier vs AdaBoostRegressor
#dataset is imbalanced(?) hence using roc_auc_score ?But in this case I see it as: y_test is binary, y_pred_proba is %prob?

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
# Instantiate ada
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)
# Fit ada to the training set
ada.fit(X_train, y_train)
# Compute the probabilities of obtaining the positive class
y_pred_proba = ada.predict_proba(X_test)[:,1]
# Import roc_auc_score
from sklearn.metrics import roc_auc_score
# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)
# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))

"""gradient boost"""
#does not tweak weights
#uses predecessors' residual errors as labels
#learning rate (n) as shrinkage
#regression: ypred = y1 + nr1 + nr2 + .... + nrN
#GradientBoostingRegressor, GradientBoostingClassifier
# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Instantiate gb
gb = GradientBoostingRegressor(max_depth=4, 
            n_estimators=200,
            random_state=2)
# Fit gb to the training set
gb.fit(X_train, y_train)
# Predict test set labels
y_pred = gb.predict(X_test)
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE
# Compute MSE
mse_test = MSE(y_test, y_pred)
# Compute RMSE
rmse_test = mse_test.mean()**0.5
# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))

"""stochastic gradient boosting"""
#to prevent the trees from relearning same split points and same features
#each tree trained on random row subsets of data without replacement
#in addition, columns/features without replacement chosen at split points
#creates ensemble diversity -> higher complexity -> higher variance to ensemble
# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=4,  #not much diff still using GradientBoostingRegressor
            subsample=0.9, #sample 90% of rows
            max_features=0.75, #75% of features
            n_estimators=200,                                
            random_state=2)
# Fit sgbr to the training set
sgbr.fit(X_train, y_train)
# Predict test set labels
y_pred = sgbr.predict(X_test)
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE
# Compute test set MSE
mse_test = MSE(y_test, y_pred)
# Compute test set RMSE
rmse_test = mse_test.mean()**0.5
# Print rmse_test
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))

"""hyperparameter tuning for CART"""
#dt.get_params()
# Define params_dt
params_dt = {'max_depth':[2,3,4],'min_samples_leaf':[0.12,0.14,0.16,0.18]}
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='roc_auc',
                       cv=5,
                       n_jobs=-1)
# Import roc_auc_score from sklearn.metrics
from sklearn.metrics import roc_auc_score
# Extract the best estimator
best_model = grid_dt.best_estimator_
# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:,1]
# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))

# Define the dictionary 'params_rf'
params_rf = {
    'n_estimators':[100,350,500],
    'max_features':['log2','auto','sqrt'],
    'min_samples_leaf':[2,10,30]
}

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=3,
                       verbose=1,
                       n_jobs=-1)

# Import mean_squared_error from sklearn.metrics as MSE 
from sklearn.metrics import mean_squared_error as MSE
# Extract the best estimator
best_model = grid_rf.best_estimator_
# Predict test set labels
y_pred = best_model.predict(X_test)
# Compute rmse_test
rmse_test = MSE(y_test,y_pred).mean()**0.5
# Print rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test)) 