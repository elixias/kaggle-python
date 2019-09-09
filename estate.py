import pandas as pd

df = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv', header=0, na)
df.describe()
print(df.columns)
df = df.dropna(axis=0)

y = df.Price #prediction/target
x_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = df[x]
X.describe()

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X,train_y)
forest_pred = forest_model.predict(valX)
print(mean_absolute_error(val_y,forest_pred))

#n_estimators=50, random_state=0, criterion='mae', min_samples_split=20, max_depth=7


#X = X_full.select_dtypes(exclude=['object'])
"""using imputers for data cleaning"""
"""imputing with extension"""
for col in cols_with_missing:
	X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
	X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
"""Without extension"""
from sklearn.impute import SimpleImputer
# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns


"""kaggle code"""
final_X_test = X_test.copy()
for col in emptycol:
    final_X_test[col + '_was_missing'] = final_X_test[col].isnull()
imputed_final_X_test = pd.DataFrame(my_imputer.fit_transform(final_X_test))
imputed_final_X_test.columns = final_X_test.columns

final_X_test = imputed_final_X_test.copy()

# Fill in the line below: get test predictions
preds_test = model.predict(imputed_final_X_test)

step_4.b.check()


"""categorical data"""
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])
#We set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data, and
#setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).


from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
