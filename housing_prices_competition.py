import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score as cvs
from sklearn.ensemble.partial_dependence import plot_partial_dependence

#Retrieve data.
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#Drop rows from training set that do not have 'SalePrice' values.
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

#Store target values in y variable.
y = train_data['SalePrice']

#Remove target variables column from feature set.
x_train = train_data.drop(['Id', 'SalePrice'], axis=1)
x_test = test_data.drop(['Id'], axis=1)

#One hot encode categorical data.
encoded_x_train = pd.get_dummies(x_train)
encoded_x_test = pd.get_dummies(x_test)

#Find columns that have missing values.
x_train_cols_with_missing = [col for col in encoded_x_train.columns
                     if encoded_x_train[col].isnull().any()]

x_test_cols_with_missing = [col for col in encoded_x_test.columns
                     if encoded_x_test[col].isnull().any()]

#Add extra columns to indenify which rows had missing values
encoded_x_train_plus = encoded_x_train.copy()
for col in x_train_cols_with_missing:
    encoded_x_train_plus[col + '_was_missing'] = encoded_x_train_plus[col].isnull()
    
encoded_x_test_plus = encoded_x_test.copy()
for col in x_test_cols_with_missing:
    encoded_x_test_plus[col + '_was_missing'] = encoded_x_test_plus[col].isnull()
    
#Use imputer to fill missing values
imputer = SimpleImputer()
imputed_encoded_x_train_plus = pd.DataFrame(imputer.fit_transform(encoded_x_train_plus))
imputed_encoded_x_train_plus.columns = encoded_x_train_plus.columns

imputed_encoded_x_test_plus = pd.DataFrame(imputer.fit_transform(encoded_x_test_plus))
imputed_encoded_x_test_plus.columns = encoded_x_test_plus.columns

#Align testing and training data sets
final_train, final_test = imputed_encoded_x_train_plus.align(imputed_encoded_x_test_plus, join='inner', axis=1)

#Create model and fit
my_model = GradientBoostingRegressor()
my_model.fit(final_train, y)

#Use cross validation to evaluate model
scores = cvs(my_model, final_train, y, scoring='neg_mean_absolute_error')
print('Mean Absolute Error using Cross Validation is: ', (-1 * scores.mean()))

#Plot some partial dependences
my_graphs = plot_partial_dependence(my_model,
                                    X = final_train,
                                    features = [2, 5],
                                    feature_names = final_train.columns, 
                                    grid_resolution=10)

#Predict saleprice of test data
predictions = my_model.predict(final_test)

#Create a submission dataframe and export to a csv file
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': predictions})
output.to_csv('submission.csv', index=False)