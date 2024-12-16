import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


# 1. zadatak
data = pd.read_csv("Housing.csv")

rows, cols = (data.shape)
print(f"Skup podataka sastoji se od {rows} redova i {cols} stupaca")
print(f"\nVrste podataka u skupu:\n{data.dtypes}")
print(f"\nNedostajuće vrijednosti u skupu:\n{data.isnull().sum()}")


# 2. zadatak
print(f"Osnovna statistika numeričkih varijabli:\n{data.describe()}")

plt.figure(figsize = (10, 6))
plt.scatter(data['area'], data['price'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs. Price')
plt.show()

numeric_cols = data.select_dtypes(include=np.number).columns
correlation_matrix = (data[numeric_cols].corr())
plt.figure(figsize = (10, 6))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Matrica korelacija svih numeričkih varijabli')
plt.show()

sns.pairplot(data[numeric_cols])
plt.show()


# 3. zadatak
data_binary = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
for column in data_binary:
    data[column] = LabelEncoder().fit_transform(data[column])

data = pd.get_dummies(data,columns = ['furnishingstatus'], drop_first = True)

data['furnishingstatus_semi-furnished'] = LabelEncoder().fit_transform(data['furnishingstatus_semi-furnished'])
data['furnishingstatus_unfurnished'] = LabelEncoder().fit_transform(data['furnishingstatus_unfurnished'])

print(data)

categorical_cols = data.select_dtypes(exclude=np.number).columns

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = ohe.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(categorical_cols))
data = pd.concat([data, encoded_df], axis=1)
data = data.drop(categorical_cols, axis=1)

numeric_cols = data.select_dtypes(include=np.number).columns

def check_vif(data):
  vif = pd.DataFrame()
  vif['Features'] = data.columns
  vif['VIF'] = [variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
  return vif

vif_data = check_vif(data.drop('price',axis=1))
print("\n", vif_data)

vif_data = check_vif(data.drop(['price','bedrooms'],axis=1))
print("\n", vif_data)

vif_data = check_vif(data.drop(['price','bedrooms','bathrooms'],axis=1))
print("\n", vif_data)

vif_data = check_vif(data.drop(['price','bedrooms','stories'],axis=1))
print("\n", vif_data)

vif_data = check_vif(data.drop(['price','bedrooms','mainroad'],axis=1))
print("\n", vif_data)

numerical_data_for_scaling = ['area']
scaler = StandardScaler()
scaler.fit(data[numerical_data_for_scaling])

data[numerical_data_for_scaling] = scaler.transform(data[numerical_data_for_scaling])

print("\n", data)


# 4. zadatak
y = data['price']
X = data.drop(['price','bedrooms','mainroad'], axis=1)

X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()

print("\n", results.summary())


# 5. zadatak
r_squared = results.rsquared
print(f"R-squared: {r_squared}")

residuals = results.resid
fitted_values = results.fittedvalues

plt.figure(figsize=(8, 6))
sns.residplot(x=fitted_values, y=residuals)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(data['area'], results.fittedvalues, color='blue', label='Predicted')
plt.scatter(data['area'], data['price'], color='red', label='Actual')
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Predviđene vrijednosti")
plt.title("Stvarne vs. Predviđene vrijednosti")
plt.legend()

plt.subplot(1, 2, 2)
sns.residplot(x=fitted_values, y=residuals, color='red', label='Residuals')
plt.xlabel("Predviđene vrijednosti")
plt.ylabel("Reziduali")
plt.title("Graf rezidualnih vrijednosti")
plt.legend()

plt.tight_layout()
plt.show()
