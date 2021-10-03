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

# Make plots?
make_plots = False

# Optimal model list and parameter list
params_list = []  # Parameters list
optimal_model_list = [] # Store all 3 country models
for i in range(len(countries)):

    # Country C02 - country i
    country_c02 = data.loc[(data['country'] == countries[i]) & (data['year'] >= 1970), "co2"]
    country_c02 = np.array(country_c02)

    # Compare all possible models
    k_set = [0,1,2] # Set of possibilities for k


    for k in range(len(k_set)):
        # Access model number
        model = k_set[k]

        if model == 0:
            y = np.array(country_c02).reshape(-1, 1)
            x1 = np.reshape(np.linspace(1970, 2019, len(y)), (len(y), 1))
            x1_ones = sm.tools.tools.add_constant(x1)

            # Model 1 statsmodels
            model_m = sm.OLS(y, x1_ones)
            results_m = model_m.fit()
            r2_1 = results_m.rsquared
            optimal_model_list.append([countries[i], r2_1])

            if make_plots:
                # Plot data and fit
                plt.plot(x1, results_m.fittedvalues)
                plt.scatter(x1, y)
                plt.show()

        if model == 1:
            y = np.array(country_c02).reshape(-1, 1)
            x_grid = np.reshape(np.linspace(1970,2019, len(y)), (len(y),1))
            model_parameters = []
            for cpd_1 in range(5,len(country_c02)-5):
                # Partition 1
                y_1 = y[0:cpd_1]
                x1 = x_grid[0:cpd_1]
                x1_ones = sm.tools.tools.add_constant(x1)
                model_1 = sm.OLS(y_1, x1_ones)
                results_1 = model_1.fit()
                r_sq_1 = results_1.rsquared

                # Partition 2
                y_2 = y[cpd_1:]
                x2 = x_grid[cpd_1:]
                x2_ones = sm.tools.tools.add_constant(x2)
                model_2 = sm.OLS(y_2, x2_ones)
                results_2 = model_2.fit()
                r_sq_2 = results_2.rsquared

                # Compute average R^2
                avg_r_sq = (r_sq_1 + r_sq_2)/2
                model_parameters.append([cpd_1, avg_r_sq])
                print(cpd_1)

                if make_plots:
                    # Plot data and fit
                    plt.plot(x1, results_1.fittedvalues)
                    plt.scatter(x1, y_1)
                    plt.plot(x2, results_2.fittedvalues)
                    plt.scatter(x2, y_2)
                    plt.axvline(x=x_grid[cpd_1], alpha=0.25)
                    plt.savefig(countries[i]+"_"+"change_point_"+str(cpd_1))
                    plt.show()

            # Model parameters
            model_parameters_array = np.array(model_parameters)
            cp_ordered = model_parameters_array[model_parameters_array[:, 1].argsort()]
            cpd = cp_ordered[-1][0]
            r2_2 = cp_ordered[-1][1]
            optimal_model_list.append([countries[i], cpd, r2_2])

        if model == 2:
            y = np.array(country_c02).reshape(-1, 1)
            x_grid = np.reshape(np.linspace(1970, 2019, len(y)), (len(y), 1))
            model_parameters = []
            for cpd_i in range(5, len(country_c02) - 5): # minimum model length is 5
                for cpd_j in range(cpd_i+5, len(country_c02) - 10):
                    # Partition 1
                    y_1 = y[0:cpd_i]
                    x1 = x_grid[0:cpd_i]
                    x1_ones = sm.tools.tools.add_constant(x1)
                    model_1 = sm.OLS(y_1, x1_ones)
                    results_1 = model_1.fit()
                    r_sq_1 = results_1.rsquared

                    # Partition 2
                    y_2 = y[cpd_i:cpd_j]
                    x2 = x_grid[cpd_i:cpd_j]
                    x2_ones = sm.tools.tools.add_constant(x2)
                    model_2 = sm.OLS(y_2, x2_ones)
                    results_2 = model_2.fit()
                    r_sq_2 = results_2.rsquared

                    # Partition 3
                    y_3 = y[cpd_j:]
                    x3 = x_grid[cpd_j:]
                    x3_ones = sm.tools.tools.add_constant(x3)
                    model_3 = sm.OLS(y_3, x3_ones)
                    results_3 = model_3.fit()
                    r_sq_3 = results_3.rsquared

                    if make_plots:
                        # Plot data and fit
                        plt.plot(x1, results_1.fittedvalues) # Model 1
                        plt.scatter(x1, y_1, alpha=0.15)
                        plt.plot(x2, results_2.fittedvalues) # Model 2
                        plt.scatter(x2, y_2, alpha=0.15)
                        plt.plot(x3, results_3.fittedvalues) # Model 3
                        plt.scatter(x3, y_3, alpha=0.15)
                        plt.axvline(x=x_grid[cpd_i], alpha=0.25)
                        plt.axvline(x=x_grid[cpd_j], alpha=0.25)
                        plt.savefig(countries[i] + "_" + "change_point_" + str(cpd_i) + "," + str(cpd_j))
                        plt.show()

                    # Compute average R^2
                    avg_r_sq = (r_sq_1 + r_sq_2 + r_sq_3) / 3
                    model_parameters.append([cpd_i, cpd_j, avg_r_sq])
                    print("Iteration", cpd_i, cpd_j)

            # Model parameters
            model_parameters_array = np.array(model_parameters)
            cp_ordered = model_parameters_array[model_parameters_array[:, 2].argsort()]
            cpd_1 = cp_ordered[-1][0]
            cpd_2 = cp_ordered[-1][1]
            r2_2 = cp_ordered[-1][2]
            optimal_model_list.append([countries[i], cpd_1, cpd_2, r2_2])

optimal_model_df = pd.DataFrame(optimal_model_list)
optimal_model_df.to_csv("/Users/tassjames/Desktop/carbon_dioxide_emissions/optimal_models.csv")


