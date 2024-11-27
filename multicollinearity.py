# import pandas as pd
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.impute import SimpleImputer
# from statsmodels.stats.outliers_influence import variance_inflation_factor

# # Define the ReduceVIF class (paste your provided code here)

# class ReduceVIF(BaseEstimator, TransformerMixin):
#     def __init__(self, thresh=10.0, impute=True, impute_strategy='median'):
#         self.thresh = thresh
#         if impute:
#             self.imputer = SimpleImputer(strategy=impute_strategy)

#     def fit(self, X, y=None):
#         if hasattr(self, 'imputer'):
#             self.imputer.fit(X)
#         return self

#     def transform(self, X, y=None):
#         columns = X.columns.tolist()
#         if hasattr(self, 'imputer'):
#             X = pd.DataFrame(self.imputer.transform(X), columns=columns)
#         return ReduceVIF.calculate_vif(X, self.thresh)

#     @staticmethod
#     def calculate_vif(X, thresh=10.0):
#         dropped = True
#         while dropped:
#             variables = X.columns
#             dropped = False
#             vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

#             max_vif = max(vif)
#             if max_vif > thresh:
#                 maxloc = vif.index(max_vif)
#                 print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
#                 X = X.drop([X.columns.tolist()[maxloc]], axis=1)
#                 dropped = True
#         return X

# # Load your dataset
# df = pd.read_csv('day_28_flg_random_oversampled_data.csv')

# # Drop the target variable
# X = df.drop(columns=['day_28_flg'])

# # Handle categorical variables (optional)
# # X = pd.get_dummies(X, drop_first=True)

# # Initialize and apply ReduceVIF
# vif_reducer = ReduceVIF(thresh=10.0)  # Set VIF threshold
# X_reduced = vif_reducer.fit_transform(X)

# # Display the remaining features
# print("Remaining features after reducing multicollinearity:")
# print(X_reduced.columns)

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import SimpleImputer

# Load your dataset
df = pd.read_csv('day_28_flg_random_oversampled_data.csv')

X = df[['sofa_first', 'age', 'stroke_flg', 'aline_flg', 'icu_los_day', 'hospital_los_day', 'gender_num',
        'service_num', 'renal_flg', 'liver_flg', 'copd_flg', 'cad_flg', 'mal_flg', 'resp_flg', 'abg_count',
        'wbc_first', 'platelet_first', 'bun_first', 'creatinine_first', 'po2_first', 'pco2_first', 'iv_day_1']]

# # Select only the SOFA and age columns
# Y = df[['sofa_first', 'age']]  # Replace with the exact column names in your dataset

# # Calculate VIF for each variable
# vif_data = pd.DataFrame()
# vif_data['Feature'] = Y.columns
# vif_data['VIF'] = [variance_inflation_factor(Y.values, i) for i in range(Y.shape[1])]

# # Display the VIF results
# print(vif_data)


# Impute missing values if necessary (using median by default)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X_imputed.columns
vif_data['VIF'] = [variance_inflation_factor(X_imputed.values, i) for i in range(X_imputed.shape[1])]

# Identify columns with 'inf' VIF (indicating perfect multicollinearity)
columns_to_drop = vif_data[vif_data['VIF'] == float('inf')]['Feature'].tolist()

# Drop columns with high multicollinearity (inf VIFs)
X_reduced = X_imputed.drop(columns=columns_to_drop)

# Calculate the VIF for the remaining features after dropping collinear variables
vif_data_reduced = pd.DataFrame()
vif_data_reduced['Feature'] = X_reduced.columns
vif_data_reduced['VIF'] = [variance_inflation_factor(X_reduced.values, i) for i in range(X_reduced.shape[1])]

# Display the new VIF results after removing collinear variables
print("\nVIF Results:")
print(vif_data_reduced)
