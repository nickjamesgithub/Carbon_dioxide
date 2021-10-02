import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Utilities import dendrogram_plot_labels

data = pd.read_csv("/Users/tassjames/Desktop/carbon_dioxide_emissions/owid-co2-data.csv")

countries = ["China", "United States", "India", "Russia", "Japan", "Japan", "Iran", "Germany", "Indonesia", "South Korea", "Saudi Arabia", "Canada",
             "South Africa", "Brazil", "Mexico", "Australia", "Turkey", "United Kingdom", "Italy", "France", "Poland", "Kazakhstan", "Thailand",
             "Taiwan", "Spain", "Malaysia", "Pakistan", "Vietnam", "Egypt", "Ukraine", "Iraq", "United Arab Emirates", "Argentina", "Algeria",
             "Netherlands", "Philippines", "Nigeria", "Venezuela", "Uzbekistan", "Qatar", "Kuwait", "Colombia", "Bangladesh", "Czechia", "Belgium",
             "Turkmenistan", "Chile", "Romania", "Morocco"]
countries.sort()
# "Oman", "Austria", "Austria", "Greece", "Mongolia", "Israel", "Belarus", "Serbia", "Peru", "Hungary"

length_country_vector = len(countries)
print("# Countries", length_country_vector)

counter = 1970
trajectory_matrix_list = [] # Trajectory matrix list
while counter < 2020:
    trajectories = np.zeros((len(countries),len(countries)))
    for i in range(len(countries)):
        for j in range(len(countries)):
            # Country C02 - country i
            country_c02_i = data.loc[(data['country'] == countries[i]) & (data['year'] >= counter) & (data['year'] < counter+10), "co2"]
            country_co2_i = np.array(country_c02_i)

            # Country C02 - country i
            country_c02_j = data.loc[(data['country'] == countries[j]) & (data['year'] >= counter) & (data['year'] < counter + 10), "co2"]
            country_co2_j = np.array(country_c02_j)

            # Make L1 Norm trajectory for country i
            co2_norm_i = np.sum(np.abs(country_co2_i))
            co2_trajectory_i = country_co2_i / co2_norm_i

            # Make L1 Norm trajectory for country j
            co2_norm_j = np.sum(np.abs(country_co2_j))
            co2_trajectory_j = country_co2_j / co2_norm_j

            # Trajectory matrix
            trajectories[i,j] = np.sum(np.abs(co2_trajectory_i - co2_trajectory_j))

        # Print iteration through the loop
        print("Iteration", i)

    # Append Trajectory matrix to overall list
    trajectory_matrix_list.append(trajectories)
    counter += 10

# Plot labels list
plot_labels_list = ["1970-1979", "1980-1989", "1990-1999", "2000-2009", "2010-2019"]

# Loop over trajectory matrix list
for i in range(len(trajectory_matrix_list)):
    matrix = trajectory_matrix_list[i]
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)
    fig.colorbar(im)
    # X ticks and y ticks
    ax.set_xticks(np.arange(len(countries)))
    ax.set_yticks(np.arange(len(countries)))
    # Tick labels
    ax.set_xticklabels(countries, fontsize=6, rotation=90)
    ax.set_yticklabels(countries, fontsize=6)
    plt.title(plot_labels_list[i])
    plt.savefig("Trajectory_"+plot_labels_list[i])
    plt.show()

    # Plot dendrogram
    dendrogram_plot_labels(matrix, plot_labels_list[i], "_Trajectory_", labels=countries)

# # Country GDP - country i
# country_gdp_i = data.loc[(data['country'] == countries[i]) & (counter <= data['year'] < counter+10), "gdp"]
# # Country Population - country i
# country_population_i = data.loc[(data['country'] == countries[i]) & (counter <= data['year'] < counter+10), "population"]

# # Country C02 - country j
# country_c02_j = data.loc[(data['country'] == countries[j]) & (counter <= data['year'] < counter + 10), "co2"]
# # Country GDP - country j
# country_gdp_j = data.loc[(data['country'] == countries[j]) & (counter <= data['year'] < counter + 10), "gdp"]
# # Country Population - country j
# country_population_j = data.loc[(data['country'] == countries[j]) & (counter <= data['year'] < counter + 10), "population"]





