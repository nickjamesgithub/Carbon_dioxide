import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv("/Users/tassjames/Desktop/carbon_dioxide_emissions/owid-co2-data.csv")

countries = ["China", "United States", "India", "Russia", "Japan", "Iran", "Germany", "Indonesia", "South Korea", "Saudi Arabia", "Canada",
             "South Africa", "Brazil", "Mexico", "Australia", "Turkey", "United Kingdom", "Italy", "France", "Poland", "Kazakhstan", "Thailand",
             "Taiwan", "Spain", "Malaysia", "Pakistan", "Vietnam", "Egypt", "Ukraine", "Iraq", "United Arab Emirates", "Argentina", "Algeria",
             "Netherlands", "Philippines", "Nigeria", "Venezuela", "Uzbekistan", "Qatar", "Kuwait", "Colombia", "Bangladesh", "Czechia", "Belgium",
             "Turkmenistan", "Chile", "Romania", "Morocco", "Oman", "Austria", "Greece", "Mongolia", "Israel", "Belarus", "Serbia", "Peru", "Hungary"]
countries.sort()

slope_list = []
params_list = [] # Parameters list
for i in range(len(countries)):

    # Country C02 - country i
    country_c02 = data.loc[(data['country'] == countries[i]) & (data['year'] >= 1970), "co2"]

    # First difference time series
    c02_diff = np.diff(country_c02)

    # Fit OLS to first difference time series
    y = np.array(c02_diff).reshape(-1, 1)
    x1_diff = np.reshape(np.linspace(1971, 2019, len(y)), (len(y), 1))
    x1_ones = sm.tools.tools.add_constant(x1_diff)

    # Model 1 statsmodels
    model_m = sm.OLS(y, x1_ones)
    results_m = model_m.fit()

    # print(results_m.summary())

    # AIC/BIC/Adjusted R2
    params_m = results_m.params
    params_int = results_m.params[0]
    params_slope = results_m.params[1]
    p_value_slope = results_m.pvalues[1]
    params_list.append([countries[i], params_int, params_slope, p_value_slope])

    # # Plot first differences
    # plt.plot(x1_diff, results_m.fittedvalues, label="OLS")
    # plt.scatter(x1_diff, y, label="data")
    # plt.title("CO2_Acceleration_"+countries[i])
    # plt.legend()
    # plt.savefig("Acceleration_"+countries[i])
    # plt.show()

# Sort parameter list
sorted_slopes = sorted(params_list, key = lambda x: x[0])
print(sorted_slopes)
# Sorted slope as a dataframe
sorted_slopes_df = pd.DataFrame(sorted_slopes)
sorted_slopes_df.to_csv("/Users/tassjames/Desktop/hydrology/acceleration.csv")