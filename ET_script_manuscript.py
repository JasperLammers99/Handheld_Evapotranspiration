import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



# Load data
RB = pd.read_excel("Rietholzbach_data.xlsx")

RB['IR_correct'] = RB['IR']*RB['Cos(pitch)']

avg_values_hh = [RB['IR_correct'].mean(), RB['T_soil'].mean(), RB['T_air_x'].mean(), RB['RH_x'].mean(), RB['wind'].mean()]
avg_values_hh = np.array(avg_values_hh)

# Calibrate and validate on hh data
X = np.column_stack((RB['IR_correct'].values, RB['T_soil'].values, RB['T_air_x'].values, RB['RH_x'], RB['wind']))
y = RB["ET"]
y2 = RB["LE_ET"]

# Number of iterations
num_iterations = 2000

# Lists to store coefficients and intercepts for each iteration
coefficients_hh_all = []
intercepts_hh_all = []
coefficients_hhLEET_all = []
intercepts_hhLEET_all = []
RMSE_hh_all = []
RMSE_hh_LEET_all = []

for iteration in range(num_iterations):
    # Split the data into training and validation sets using bootstrapping
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=iteration)
    X_train, X_val, y2_train, y2_val = train_test_split(X, y2, test_size=0.33, random_state=iteration)

    # Make linear regression model for the first model (y)
    model_hh = LinearRegression()
    model_hh.fit(X_train, y_train)

    coefficients_hh_all.append(model_hh.coef_)
    intercepts_hh_all.append(model_hh.intercept_)

    model_hhLEET = LinearRegression()  # Define a new model for y2
    model_hhLEET.fit(X_train, y2_train)

    coefficients_hhLEET_all.append(model_hhLEET.coef_)
    intercepts_hhLEET_all.append(model_hhLEET.intercept_)

    y_pred = model_hh.predict(X_val)
    rmse_hh = math.sqrt(mean_squared_error(y_val, y_pred))
    RMSE_hh_all.append(rmse_hh)

    y2_pred = model_hhLEET.predict(X_val)
    rmse_hhLEET = math.sqrt(mean_squared_error(y2_val, y2_pred))
    RMSE_hh_LEET_all.append(rmse_hhLEET)

# Calculate the average coefficients, intercepts, and RMSE
avg_RMSE_hh = np.mean(RMSE_hh_all)
avg_RMSE_hh_LEET = np.mean(RMSE_hh_LEET_all)

# Combine coefficients and intercepts for both models
coefficients_hh_all = np.array(coefficients_hh_all)
intercepts_hh_all = np.array(intercepts_hh_all)
coefficients_hh_LEET = np.array(coefficients_hhLEET_all)
intercepts_hh_LEET = np.array(intercepts_hhLEET_all)

# Calculate values with coefficients and intercepts
value_hh_coeffs = coefficients_hh_all * avg_values_hh
value_hh = np.hstack((value_hh_coeffs, intercepts_hh_all.reshape(-1, 1)))

value_hh_LEET_coeffs = coefficients_hh_LEET * avg_values_hh
value_hh_LEET = np.hstack((value_hh_LEET_coeffs, intercepts_hh_LEET.reshape(-1, 1)))

headers = ['IR_correct', 'T_soil', 'T_air', 'RH', 'wind', 'intercept']
df_hh = pd.DataFrame(value_hh, columns=headers)
df_hh_LEET = pd.DataFrame(value_hh_LEET, columns=headers)

df_hh['Data'] = 'Lysimeter'
df_hh_LEET['Data'] = 'Eddy Covariance'

# Combine the dataframes
cdf = pd.concat([df_hh, df_hh_LEET])

# Melt the dataframe
mdf = pd.melt(cdf, id_vars=['Data'], var_name=['Variable'])
plt.figure(figsize=(12, 8))
sns.boxplot(x='Variable', y='value', hue='Data', data=mdf, notch=True, orient='v')
plt.xlabel('Constituents')
plt.ylabel('Contribution to ET (mm/h)')
plt.grid(True)
plt.show()

# Print average results
print("\nAverage results for the first model (Lysimeter):")
print("Average RMSE:", avg_RMSE_hh)

print("\nAverage results for the second model (Eddy Covariance):")
print("Average RMSE:", avg_RMSE_hh_LEET)





# Calibrate and validate on Buel data with handheld device T_soil
X = np.column_stack((RB['ISR'].values, RB['T_soil'].values, RB['T_air_y'].values, RB['RH_y'], RB['v_wind']))
y = RB["ET"]
y2 = RB["LE_ET"]

avg_values_Buel = [RB['ISR'].mean(), RB['T_soil'].mean(), RB['T_air_y'].mean(), RB['RH_y'].mean(), RB['v_wind'].mean()]
avg_values_Buel = np.array(avg_values_Buel)

# Number of iterations
num_iterations = 2000

# Lists to store coefficients and intercepts for each iteration
coefficients_Buel_all = []
intercepts_Buel_all = []
coefficients_BuelLEET_all = []
intercepts_BuelLEET_all = []
RMSE_Buel_all = []
RMSE_Buel_LEET_all = []


for iteration in range(num_iterations):
    # Split the data into training and validation sets using bootstrapping
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=iteration)
    X_train, X_val, y2_train, y2_val = train_test_split(X, y2, test_size=0.33, random_state=iteration)

    # Make linear regression model for the first model (y)
    model_Buel = LinearRegression()
    model_Buel.fit(X_train, y_train)

    # Store coefficients and intercept
    coefficients_Buel_all.append(model_Buel.coef_)
    intercepts_Buel_all.append(model_Buel.intercept_)

    # Make linear regression model for the second model (y2)
    model_BuelLEET = LinearRegression()
    model_BuelLEET.fit(X_train, y2_train)

    # Store coefficients and intercept
    coefficients_BuelLEET_all.append(model_BuelLEET.coef_)
    intercepts_BuelLEET_all.append(model_BuelLEET.intercept_)

    y_pred = model_Buel.predict(X_val)
    rmse_Buel = math.sqrt(mean_squared_error(y_val, y_pred))
    RMSE_Buel_all.append(rmse_Buel)

    y2_pred = model_BuelLEET.predict(X_val)
    rmse_BuelLEET = math.sqrt(mean_squared_error(y2_val, y2_pred))
    RMSE_Buel_LEET_all.append(rmse_BuelLEET)

# Calculate the average coefficients and intercepts
avg_RMSE_Buel = np.mean(RMSE_Buel_all)
avg_RMSE_Buel_LEET = np.mean(RMSE_hh_LEET_all)

avg_values_Buel_Ts = [RB['ISR'].mean(),RB['T_soil'].mean(), RB['T_air_y'].mean(), RB['RH_y'].mean(), RB['v_wind'].mean()]
avg_values_Buel_Ts = np.array(avg_values_Buel_Ts)

coefficients_Buel_all_Ts = np.array(coefficients_Buel_all)
intercepts_Buel_Ts = np.array(intercepts_Buel_all)
coefficients_Buel_LEET_Ts = np.array(coefficients_BuelLEET_all)
intercepts_BuelLEET_Ts = np.array(intercepts_BuelLEET_all)

value_Buel_Ts_coefs = coefficients_Buel_all*avg_values_Buel_Ts
value_Buel_Ts = np.hstack((value_Buel_Ts_coefs, intercepts_Buel_Ts.reshape(-1,1)))
value_Buel_LEET_Ts_coefs = coefficients_Buel_LEET_Ts * avg_values_Buel_Ts
value_Buel_LEET_Ts = np.hstack((value_Buel_LEET_Ts_coefs, intercepts_BuelLEET_Ts.reshape(-1,1)))


headers = ['ISR', 'T_soil', 'T_air', 'RH', 'wind', 'intercept']
df_Buel_Ts = pd.DataFrame(value_Buel_Ts, columns=headers)
df_Buel_LEET_Ts = pd.DataFrame(value_Buel_LEET_Ts, columns=headers)

df_Buel_Ts['Data'] = 'Lysimeter'
df_Buel_LEET_Ts['Data'] = 'Eddy Covariance'

# Combine the dataframes
cdf = pd.concat([df_Buel_Ts, df_Buel_LEET_Ts])

# Melt the dataframe
mdf = pd.melt(cdf, id_vars=['Data'], var_name=['Variable'])
plt.figure(figsize=(12, 8))
sns.boxplot(x='Variable', y='value', hue='Data', data=mdf, notch=True, orient='v')
plt.xlabel('Constituents')
plt.ylabel('Contribution to ET (mm/h)')
plt.grid(True)
plt.show()


# Print average results
print("\nAverage results for the first model (y):")
print("Average RMSE:", avg_RMSE_Buel)

print("\nAverage results for the second model (y2):")
print("Average RMSE:", avg_RMSE_Buel_LEET)






# Calibrate and validate on Buel data without handheld device T_soil
X = np.column_stack((RB['ISR'].values, RB['T_air_y'].values, RB['RH_y'], RB['v_wind']))
y = RB["ET"]
y2 = RB["LE_ET"]

# Number of iterations
num_iterations = 2000

# Lists to store coefficients and intercepts for each iteration
coefficients_Buel_no_Ts_all = []
intercepts_Buel_no_Ts_all = []
coefficients_Buel_no_TsLEET_all = []
intercepts_Buel_no_TsLEET_all = []
RMSE_Buel_no_Ts_all = []
RMSE_Buel_no_Ts_LEET_all = []

for iteration in range(num_iterations):
    # Split the data into training and validation sets using bootstrapping
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=iteration)
    X_train, X_val, y2_train, y2_val = train_test_split(X, y2, test_size=0.33, random_state=iteration)

    # Make linear regression model for the first model (y)
    model_Buel_no_Ts = LinearRegression()
    model_Buel_no_Ts.fit(X_train, y_train)

    # Store coefficients and intercept
    coefficients_Buel_no_Ts_all.append(model_Buel_no_Ts.coef_)
    intercepts_Buel_no_Ts_all.append(model_Buel_no_Ts.intercept_)

    # Make linear regression model for the second model (y2)
    model_Buel_no_TsLEET = LinearRegression()
    model_Buel_no_TsLEET.fit(X_train, y2_train)

    # Store coefficients and intercept
    coefficients_Buel_no_TsLEET_all.append(model_Buel_no_TsLEET.coef_)
    intercepts_Buel_no_TsLEET_all.append(model_Buel_no_TsLEET.intercept_)

    y_pred = model_Buel_no_Ts.predict(X_val)
    rmse_Buel_no_Ts = math.sqrt(mean_squared_error(y_val, y_pred))
    RMSE_Buel_no_Ts_all.append(rmse_Buel_no_Ts)

    y2_pred = model_Buel_no_TsLEET.predict(X_val)
    rmse_Buel_no_TsLEET = math.sqrt(mean_squared_error(y2_val, y2_pred))
    RMSE_Buel_no_Ts_LEET_all.append(rmse_Buel_no_TsLEET)

# Calculate the average coefficients and intercepts
avg_coefficients_Buel_no_Ts = np.mean(coefficients_Buel_no_Ts_all, axis=0)
avg_intercept_Buel_no_Ts = np.mean(intercepts_Buel_no_Ts_all)
avg_RMSE_Buel_no_Ts = np.mean(RMSE_Buel_no_Ts_all)

avg_coefficients_Buel_no_TsLEET = np.mean(coefficients_Buel_no_TsLEET_all, axis=0)
avg_intercept_Buel_no_TsLEET = np.mean(intercepts_Buel_no_TsLEET_all)
avg_RMSE_Buel_no_Ts_LEET = np.mean(RMSE_Buel_no_Ts_LEET_all)

avg_values_Buel_no_Ts = [RB['ISR'].mean(), RB['T_air_y'].mean(), RB['RH_y'].mean(), RB['v_wind'].mean()]
avg_values_Buel_no_Ts = np.array(avg_values_Buel_no_Ts)


coefficients_Buel_all_no_Ts = np.array(coefficients_Buel_no_Ts_all)
intercepts_Buel_no_Ts = np.array(intercepts_Buel_no_Ts_all)
coefficients_Buel_LEET_no_Ts = np.array(coefficients_Buel_no_TsLEET_all)
intercepts_Buel_LEET_no_TS = np.array(intercepts_Buel_no_TsLEET_all)


value_Buel_no_Ts_coefficients = coefficients_Buel_all_no_Ts*avg_values_Buel_no_Ts
value_Buel_no_Ts = np.hstack((value_Buel_no_Ts_coefficients, intercepts_Buel_no_Ts.reshape(-1,1)))
value_Buel_LEET_Ts_coefficients = coefficients_Buel_LEET_no_Ts * avg_values_Buel_no_Ts
value_Buel_LEET_Ts = np.hstack((value_Buel_LEET_Ts_coefficients, intercepts_Buel_LEET_no_TS.reshape(-1,1)))

headers = ['ISR', 'T_air', 'RH', 'wind','intercept']
df_Buel_no_Ts = pd.DataFrame(value_Buel_no_Ts, columns=headers)
df_Buel_LEET_no_Ts = pd.DataFrame(value_Buel_LEET_Ts, columns=headers)

df_Buel_no_Ts['Data'] = 'Lysimeter'
df_Buel_LEET_no_Ts['Data'] = 'Eddy Covariance'

# Combine the dataframes
cdf = pd.concat([df_Buel_no_Ts, df_Buel_LEET_no_Ts])

# Melt the dataframe
mdf = pd.melt(cdf, id_vars=['Data'], var_name=['Variable'])
plt.figure(figsize=(12, 8))
sns.boxplot(x='Variable', y='value', hue='Data', data=mdf, notch=True, orient='v')
plt.xlabel('Constituents')
plt.ylabel('Contribution to ET (mm/h)')
plt.grid(True)
plt.show()

# Print average results
print("\nAverage results for the first model (Lysimeter):")
print("Average RMSE:", avg_RMSE_Buel_no_Ts)

print("\nAverage results for the second model (Eddy Covariance):")
print("Average RMSE:", avg_RMSE_Buel_no_Ts_LEET)




# Make predictions and plots from the average coefficients
# First handheld from the lysimeter and the EC tower

y1 = RB.ET
y2 = RB.LE_ET
X = np.column_stack((RB['IR_correct'].values, RB['T_soil'].values, RB['T_air_x'], RB['RH_x'], RB['wind']))
y_prediction_lysi = np.sum(coefficients_hh_all[16]*X, axis = 1) + intercepts_hh_all[16] # Index 16 is a representative RMSE for hh lysi
y_prediction_EC = np.sum(coefficients_hh_LEET[3]*X, axis = 1) + intercepts_hh_LEET[3] # Index 3 is a representative RMSE for hh EC

plt.figure()
plt.scatter(y_prediction_lysi, y1, alpha=0.5, label = f'Lysimeter data \nValidation RMSE: {round(RMSE_hh_all[16],2)} mm/h', color = 'g')
plt.scatter(y_prediction_EC, y2, alpha = 0.5, label =  f'EC data \nValidation RMSE: {round(RMSE_hh_LEET_all[3],2)} mm/h', color = 'b')
plt.plot([0, 0.6], [0, 0.6], label = '1:1 line', linestyle = '--', color = 'red')
plt.xlabel('Smartphone derived evapotranspiration (mm/h)')
plt.ylabel('Lysimeter or EC evapotranspiration (mm/h)')
plt.legend()
plt.show()

ET_comp_mod = LinearRegression()
x = RB.ET.values.reshape(-1, 1)
y = RB.LE_ET.values.reshape(-1, 1)
ET_comp_mod.fit(x, y)
ypred = ET_comp_mod.predict(x)
RMSE_ET = math.sqrt(mean_squared_error(y, ypred))


plt.figure()
plt.scatter(y1, y2, label = f'Data\nRMSE: {round(RMSE_ET,2)} mm/h')
plt.plot([0, 0.6],[0,0.6], label = '1:1 line', linestyle = '--', color = 'red')
plt.xlabel('Lysimeter evapotranspiration (mm/h)')
plt.ylabel('EC evapotranspiration (mm/h)')
plt.legend()
plt.show()

