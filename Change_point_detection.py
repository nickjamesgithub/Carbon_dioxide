import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv("/Users/tassjames/Desktop/carbon_dioxide_emissions/owid-co2-data.csv")

countries = ["China", "United States", "India", "Russia", "Japan", "Iran", "Germany", "Indonesia", "South Korea",
             "Saudi Arabia", "Canada",
             "South Africa", "Brazil", "Mexico", "Australia", "Turkey", "United Kingdom", "Italy", "France", "Poland",
             "Kazakhstan", "Thailand",
             "Taiwan", "Spain", "Malaysia", "Pakistan", "Vietnam", "Egypt", "Ukraine", "Iraq", "United Arab Emirates",
             "Argentina", "Algeria",
             "Netherlands", "Philippines", "Nigeria", "Venezuela", "Uzbekistan", "Qatar", "Kuwait", "Colombia",
             "Bangladesh", "Czechia", "Belgium",
             "Turkmenistan", "Chile", "Romania", "Morocco", "Oman", "Austria", "Austria", "Greece", "Mongolia",
             "Israel", "Belarus", "Serbia", "Peru", "Hungary"]
countries.sort()

params_list = []  # Parameters list
for i in range(len(countries)):
    # Country C02 - country i
    country_c02 = data.loc[(data['country'] == countries[i]) & (data['year'] >= 1970), "co2"]
    country_c02 = np.array(country_c02)

    k_set = [0,1,2] # Set of possibilities for k
    for k in range(len(k_set)):
        r_square_list = []
        model = k_set[k]

        if model == 0:
            y = np.array(country_c02).reshape(-1, 1)
            x1 = np.reshape(np.linspace(1970, 2019, len(y)), (len(y), 1))
            x1_ones = sm.tools.tools.add_constant(x1)

            # Model 1 statsmodels
            model_m = sm.OLS(y, x1_ones)
            results_m = model_m.fit()
            r2_1 = results_m.rsquared
            r_square_list.append(r2_1)

        if model == 1:
            y = np.array(country_c02).reshape(-1, 1)
            x_grid = np.reshape(np.linspace(1970,2019, len(y)), (len(y),1))
            model_parameters = []
            for i in range(3,len(country_c02)-3):
                # Partition 1
                y_1 = y[0:i]
                x1 = x_grid[0:i]
                x1_ones = sm.tools.tools.add_constant(x1)
                model_1 = sm.OLS(y_1, x1_ones)
                results_1 = model_1.fit()
                r_sq_1 = results_1.rsquared

                # Partition 2
                y_2 = y[i:]
                x2 = x_grid[i:]
                x2_ones = sm.tools.tools.add_constant(x2)
                model_2 = sm.OLS(y_2, x2_ones)
                results_2 = model_2.fit()
                r_sq_2 = results_2.rsquared

                # Compute average R^2
                avg_r_sq = (r_sq_1 + r_sq_2)/2
                model_parameters.append([i, avg_r_sq])
                print(i)

            # Model parameters
            model_parameters_array = np.array(model_parameters)
            cp_ordered = model_parameters_array[model_parameters_array[:, 1].argsort()]
            cpd = cp_ordered[-1][0]
            r2_2 = cp_ordered[-1][0]
            r_square_list.append(r2_2)

        if model == 2:
            y = np.array(country_c02).reshape(-1, 1)
            x_grid = np.reshape(np.linspace(1970, 2019, len(y)), (len(y), 1))
            model_parameters = []
            for i in range(3, len(country_c02) - 3):
                for j in range(i+2, len(country_c02) - 3):
                    # Partition 1
                    y_1 = y[0:i]
                    x1 = x_grid[0:i]
                    x1_ones = sm.tools.tools.add_constant(x1)
                    model_1 = sm.OLS(y_1, x1_ones)
                    results_1 = model_1.fit()
                    r_sq_1 = results_1.rsquared

                    # Partition 2
                    y_2 = y[i:j]
                    x2 = x_grid[i:j]
                    x2_ones = sm.tools.tools.add_constant(x2)
                    model_2 = sm.OLS(y_2, x2_ones)
                    results_2 = model_2.fit()
                    r_sq_2 = results_2.rsquared

                    # Partition 3
                    y_3 = y[j:]
                    x3 = x_grid[j:]
                    x3_ones = sm.tools.tools.add_constant(x3)
                    model_3 = sm.OLS(y_3, x3_ones)
                    results_3 = model_3.fit()
                    r_sq_3 = results_3.rsquared

                    # Compute average R^2
                    avg_r_sq = (r_sq_1 + r_sq_2 + r_sq_3) / 3
                    model_parameters.append([i, avg_r_sq])
                    print(i)
