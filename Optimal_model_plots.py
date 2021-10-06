import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

# Import datasets
data = pd.read_csv("/Users/tassjames/Desktop/carbon_dioxide_emissions/owid-co2-data.csv")
optimal_models = pd.read_csv("/Users/tassjames/Desktop/carbon_dioxide_emissions/optimal_models.csv")

countries = ["China", "United States", "India", "Russia", "Japan", "Iran", "Germany", "Indonesia", "South Korea",
             "Saudi Arabia", "Canada",
             "South Africa", "Brazil", "Mexico", "Australia", "Turkey", "United Kingdom", "Italy", "France", "Poland",
             "Kazakhstan", "Thailand",
             "Taiwan", "Spain", "Malaysia", "Pakistan", "Vietnam", "Egypt", "Ukraine", "Iraq", "United Arab Emirates",
             "Argentina", "Algeria",
             "Netherlands", "Philippines", "Nigeria", "Venezuela", "Uzbekistan", "Qatar", "Kuwait", "Colombia",
             "Bangladesh", "Czechia", "Belgium",
             "Turkmenistan", "Chile", "Romania", "Morocco", "Oman", "Austria"]

countries.sort()

for i in range(len(countries)):

    # Country C02 - country i
    country_c02 = data.loc[(data['country'] == countries[i]) & (data['year'] >= 1970), "co2"]
    country_c02 = np.array(country_c02)

    # Get model information and change points
    model_information = optimal_models.loc[optimal_models.iloc[:,1] == countries[i]]
    cp21 = np.int(model_information.iloc[1,2])
    cp31 = np.int(model_information.iloc[2, 2])
    cp32 = np.int(model_information.iloc[2, 3])

    # MODEL 1
    y = np.array(country_c02).reshape(-1, 1)
    x1 = np.reshape(np.linspace(1970, 2019, len(y)), (len(y), 1))
    x1_ones = sm.tools.tools.add_constant(x1)

    # Model 1 statsmodels
    model_m = sm.OLS(y, x1_ones)
    results_m = model_m.fit()

    # # Plot data and fit
    # plt.plot(x1, results_m.fittedvalues)
    # plt.scatter(x1, y, alpha=0.25)
    # plt.xlabel("Time")
    # plt.ylabel("Emissions")
    # plt.savefig("Optimal_" + countries[i] + "_" + "changepoint_" + str(0))
    # plt.show()

    # MODEL 2
    x_grid = np.reshape(np.linspace(1970,2019, len(y)), (len(y),1))

    # Partition 1
    y_21 = y[0:cp21]
    x21 = x_grid[0:cp21]
    x21_ones = sm.tools.tools.add_constant(x21)
    model_21 = sm.OLS(y_21, x21_ones)
    results_21 = model_21.fit()

    # Partition 2
    y_22 = y[cp21:]
    x22 = x_grid[cp21:]
    x22_ones = sm.tools.tools.add_constant(x22)
    model_22 = sm.OLS(y_22, x22_ones)
    results_22 = model_22.fit()

    # Plot data and fit
    plt.plot(x21, results_21.fittedvalues)
    plt.scatter(x21, y_21, alpha=0.25)
    plt.plot(x22, results_22.fittedvalues)
    plt.scatter(x22, y_22, alpha=0.25)
    plt.axvline(x=x_grid[cp21], alpha=0.25)
    plt.xlabel("Time")
    plt.ylabel("Emissions")
    plt.savefig("Optimal_"+countries[i]+"_"+"changepoint_"+str(cp21))
    plt.show()

    # MODEL 3
    x_grid = np.reshape(np.linspace(1970, 2019, len(y)), (len(y), 1))

    # Partition 1
    y_31 = y[0:cp31]
    x31 = x_grid[0:cp31]
    x31_ones = sm.tools.tools.add_constant(x31)
    model_31 = sm.OLS(y_31, x31_ones)
    results_31 = model_31.fit()

    # Partition 2
    y_32 = y[cp31:cp32]
    x32 = x_grid[cp31:cp32]
    x32_ones = sm.tools.tools.add_constant(x32)
    model_32 = sm.OLS(y_32, x32_ones)
    results_32 = model_32.fit()

    # Partition 3
    y_33 = y[cp32:]
    x33 = x_grid[cp32:]
    x33_ones = sm.tools.tools.add_constant(x33)
    model_33 = sm.OLS(y_33, x33_ones)
    results_33 = model_33.fit()

    # # Plot data and fit
    # plt.plot(x31, results_31.fittedvalues) # Model 1
    # plt.scatter(x31, y_31, alpha=0.15)
    # plt.plot(x32, results_32.fittedvalues) # Model 2
    # plt.scatter(x32, y_32, alpha=0.15)
    # plt.plot(x33, results_33.fittedvalues) # Model 3
    # plt.scatter(x33, y_33, alpha=0.15)
    # plt.axvline(x=x_grid[cp31], alpha=0.25)
    # plt.axvline(x=x_grid[cp32], alpha=0.25)
    # plt.xlabel("Time")
    # plt.ylabel("Emissions")
    # plt.savefig("Optimal_"+countries[i] + "_" + "changepoint_" + str(cp31) + "," + str(cp32))
    # plt.show()

