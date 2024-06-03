import numpy as np  #If you have not installed one of these packages on your working directory, go to File/Settings/Project:.../Python Interpreter and add this here
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EI = pd.read_excel("Veenkampen_Rollesbroich_data_Lysikeer2.xlsx") #Loading the VK and EI data
EI['IR_correct'] = EI['IR'] # In your Excel file, IR was already corrected for the angle.
EI = EI[EI['location'] == 'Simmerath'] #Here, I selected from all the data only the Rollesbroich to be used
# So, now you are left over with only the Eifel data

X1 = np.column_stack((EI['IR_correct'].values, EI['T_soil'].values, EI['T_air_x'], EI['RH_x'], EI['wind']))
X2 = np.column_stack((EI['IR_correct'].values, EI['T_soil2'].values, EI['T_air_x'], EI['RH_x'], EI['wind']))
X3 = np.column_stack((EI['IR_correct'].values, EI['T_soil3'].values, EI['T_air_x'], EI['RH_x'], EI['wind']))
X4 = np.column_stack((EI['IR_correct'].values, EI['T_soil4'].values, EI['T_air_x'], EI['RH_x'], EI['wind']))
X5 = np.column_stack((EI['IR_correct'].values, EI['T_soil5'].values, EI['T_air_x'], EI['RH_x'], EI['wind']))
X6 = np.column_stack((EI['IR_correct'].values, EI['T_soil6'].values, EI['T_air_x'], EI['RH_x'], EI['wind']))

y1 = EI.lysim_ET
y2 = EI.lysim_ET2
y3 = EI.lysim_ET3
y4 = EI.lysim_ET4
y5 = EI.lysim_ET5
y6 = EI.lysim_ET6

#These empty lists below enable you to save the results from the bootstrap runs in order to get a good overview of the model performances
coefficients_hh_all1, coefficients_hh_all2, coefficients_hh_all3, coefficients_hh_all4, coefficients_hh_all5, coefficients_hh_all6 = [],[],[],[],[],[]
intercepts_hh_all1, intercepts_hh_all2, intercepts_hh_all3, intercepts_hh_all4,intercepts_hh_all5,intercepts_hh_all6 = [],[],[],[],[],[]
RMSE_hh_all1, RMSE_hh_all2, RMSE_hh_all3, RMSE_hh_all4, RMSE_hh_all5, RMSE_hh_all6 = [],[],[],[],[],[]

num_iterations = 2000 #number of bootstrap runs
for iteration in range(num_iterations): #Each lysimeter has its own model with its own input. This model training is done 'num_iteration' times
    # Split the data into training and validation sets using bootstrapping
    X_train1, X_val1, y_train1, y_val1 = train_test_split(X1, y1, test_size=0.33, random_state=iteration)
    X_train2, X_val2, y_train2, y_val2 = train_test_split(X2, y2, test_size=0.33, random_state=iteration)
    X_train3, X_val3, y_train3, y_val3 = train_test_split(X3, y3, test_size=0.33, random_state=iteration)
    X_train4, X_val4, y_train4, y_val4 = train_test_split(X4, y4, test_size=0.33, random_state=iteration)
    X_train5, X_val5, y_train5, y_val5 = train_test_split(X5, y5, test_size=0.33, random_state=iteration)
    X_train6, X_val6, y_train6, y_val6 = train_test_split(X6, y6, test_size=0.33, random_state=iteration)

    # Make linear regression model for the first model (y)
    model_hh1 = LinearRegression()
    model_hh2 = LinearRegression()
    model_hh3 = LinearRegression()
    model_hh4 = LinearRegression()
    model_hh5 = LinearRegression()
    model_hh6 = LinearRegression()
    model_hh1.fit(X_train1, y_train1)
    model_hh2.fit(X_train2, y_train2)
    model_hh3.fit(X_train3, y_train3)
    model_hh4.fit(X_train4, y_train4)
    model_hh5.fit(X_train5, y_train5)
    model_hh6.fit(X_train6, y_train6)

    coefficients_hh_all1.append(model_hh1.coef_)
    intercepts_hh_all1.append(model_hh1.intercept_)
    coefficients_hh_all2.append(model_hh2.coef_)
    intercepts_hh_all2.append(model_hh2.intercept_)
    coefficients_hh_all3.append(model_hh3.coef_)
    intercepts_hh_all3.append(model_hh3.intercept_)
    coefficients_hh_all4.append(model_hh4.coef_)
    intercepts_hh_all4.append(model_hh4.intercept_)
    coefficients_hh_all5.append(model_hh5.coef_)
    intercepts_hh_all5.append(model_hh5.intercept_)
    coefficients_hh_all6.append(model_hh6.coef_)
    intercepts_hh_all6.append(model_hh6.intercept_)


    y_pred1 = model_hh1.predict(X_val1)
    rmse_hh1 = math.sqrt(mean_squared_error(y_val1, y_pred1))
    RMSE_hh_all1.append(rmse_hh1)
    y_pred2 = model_hh2.predict(X_val2)
    rmse_hh2 = math.sqrt(mean_squared_error(y_val2, y_pred2))
    RMSE_hh_all2.append(rmse_hh2)
    y_pred3 = model_hh3.predict(X_val3)
    rmse_hh3 = math.sqrt(mean_squared_error(y_val3, y_pred3))
    RMSE_hh_all3.append(rmse_hh3)
    y_pred4 = model_hh4.predict(X_val4)
    rmse_hh4 = math.sqrt(mean_squared_error(y_val4, y_pred4))
    RMSE_hh_all4.append(rmse_hh4)
    y_pred5 = model_hh5.predict(X_val5)
    rmse_hh5 = math.sqrt(mean_squared_error(y_val5, y_pred5))
    RMSE_hh_all5.append(rmse_hh5)
    y_pred6 = model_hh6.predict(X_val6)
    rmse_hh6 = math.sqrt(mean_squared_error(y_val6, y_pred6))
    RMSE_hh_all6.append(rmse_hh6)

# These lines below reshape the results in order to be used
avg_values_hh = [EI['IR_correct'].mean(), EI['T_soil'].mean(), EI['T_air_x'].mean(), EI['RH_x'].mean(), EI['wind'].mean()]
avg_values_hh = np.array(avg_values_hh)
coefficients_hh_all1 = np.array(coefficients_hh_all1)
coefficients_hh_all2 = np.array(coefficients_hh_all2)
coefficients_hh_all3 = np.array(coefficients_hh_all3)
coefficients_hh_all4 = np.array(coefficients_hh_all4)
coefficients_hh_all5 = np.array(coefficients_hh_all5)
coefficients_hh_all6 = np.array(coefficients_hh_all6)
intercepts_hh_all1 = np.array(intercepts_hh_all1).reshape(-1, 1)
intercepts_hh_all2 = np.array(intercepts_hh_all2).reshape(-1, 1)
intercepts_hh_all3 = np.array(intercepts_hh_all3).reshape(-1, 1)
intercepts_hh_all4 = np.array(intercepts_hh_all4).reshape(-1, 1)
intercepts_hh_all5 = np.array(intercepts_hh_all5).reshape(-1, 1)
intercepts_hh_all6 = np.array(intercepts_hh_all6).reshape(-1, 1)
value_hh_coeffs1 = coefficients_hh_all1 * avg_values_hh
value_hh_coeffs2 = coefficients_hh_all2 * avg_values_hh
value_hh_coeffs3 = coefficients_hh_all3 * avg_values_hh
value_hh_coeffs4 = coefficients_hh_all4 * avg_values_hh
value_hh_coeffs5 = coefficients_hh_all5 * avg_values_hh
value_hh_coeffs6 = coefficients_hh_all6 * avg_values_hh
value_hh1 = np.hstack((value_hh_coeffs1, intercepts_hh_all1))
value_hh2 = np.hstack((value_hh_coeffs2, intercepts_hh_all2))
value_hh3 = np.hstack((value_hh_coeffs3, intercepts_hh_all3))
value_hh4 = np.hstack((value_hh_coeffs4, intercepts_hh_all4))
value_hh5 = np.hstack((value_hh_coeffs5, intercepts_hh_all5))
value_hh6 = np.hstack((value_hh_coeffs6, intercepts_hh_all6))

headers = ['IR_correct', 'T_soil', 'T_air', 'RH', 'wind'] #Add ', 'Lysimter'' to the headers to incorporate the intercept in the boxplot
df_hh1 = pd.DataFrame(value_hh_coeffs1, columns=headers) #change 'value_hh_coefs1' for 'value_hh1' to also incorporate the intercept in the boxplot
df_hh1['source'] = 'Lysimeter 1'
df_hh2 = pd.DataFrame(value_hh_coeffs2, columns=headers)
df_hh2['source'] = 'Lysimeter 2'
df_hh3 = pd.DataFrame(value_hh_coeffs3, columns=headers)
df_hh3['source'] = 'Lysimeter 3'
df_hh4 = pd.DataFrame(value_hh_coeffs4, columns=headers)
df_hh4['source'] = 'Lysimeter 4'
df_hh5 = pd.DataFrame(value_hh_coeffs5, columns=headers)
df_hh5['source'] = 'Lysimeter 5'
df_hh6 = pd.DataFrame(value_hh_coeffs6, columns=headers)
df_hh6['source'] = 'Lysimeter 6'

cdf_EI = pd.concat([df_hh1, df_hh2, df_hh3, df_hh4, df_hh5, df_hh6], ignore_index=True)
mdf_EI = pd.melt(cdf_EI, id_vars='source', var_name='variable', value_name='value')
# Boxplot of the component importance
plt.figure(figsize=(12, 6))
sns.boxplot(x='variable', y='value', hue='source', data=mdf_EI)
plt.xlabel('Constituents')
plt.ylabel('Contribution to ET (mm/h)')
plt.grid(True)
plt.show()

headers_single = ['RMSE_value']

RMSE1 = pd.DataFrame(RMSE_hh_all1, columns=headers_single)
RMSE1['source'] = 'Lysimeter 1'
RMSE2 = pd.DataFrame(RMSE_hh_all2, columns=headers_single)
RMSE2['source'] = 'Lysimeter 2'
RMSE3 = pd.DataFrame(RMSE_hh_all3, columns=headers_single)
RMSE3['source'] = 'Lysimeter 3'
RMSE4 = pd.DataFrame(RMSE_hh_all4, columns=headers_single)
RMSE4['source'] = 'Lysimeter 4'
RMSE5 = pd.DataFrame(RMSE_hh_all5, columns=headers_single)
RMSE5['source'] = 'Lysimeter 5'
RMSE6 = pd.DataFrame(RMSE_hh_all6, columns=headers_single)
RMSE6['source'] = 'Lysimeter 6'

cdf_EI_RMSE = pd.concat([RMSE1, RMSE2, RMSE3, RMSE4, RMSE5, RMSE6], ignore_index=True)
mdf_EI_RMSE = pd.melt(cdf_EI_RMSE, id_vars='source', var_name='variable', value_name='value')
# Boxplot of the model performances
plt.figure(figsize=(12, 6))
sns.boxplot(x='source', y='RMSE_value', data=cdf_EI_RMSE)
plt.xlabel('Lysimeter')
plt.ylabel('RMSE (mm/h)')
plt.grid(True)
plt.show()

