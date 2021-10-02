import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("/Users/tassjames/Desktop/carbon_dioxide_emissions/owid-co2-data.csv")

countries = ["China", "United States", "India", "Russia", "Japan", "Japan", "Iran", "Germany", "Indonesia", "South Korea", "Saudi Arabia", "Canada",
             "South Africa", "Brazil", "Mexico", "Australia", "Turkey", "United Kingdom", "Italy", "France", "Poland", "Kazakhstan", "Thailand",
             "Taiwan", "Spain", "Malaysia", "Pakistan", "Vietnam", "Egypt", "Ukraine", "Iraq", "United Arab Emirates", "Argentina", "Algeria",
             "Netherlands", "Philippines", "Nigeria", "Venezuela", "Uzbekistan", "Qatar", "Kuwait", "Colombia", "Bangladesh", "Czechia", "Belgium",
             "Turkmenistan", "Chile", "Romania", "Morocco", "Oman", "Austria", "Austria", "Greece", "Mongolia", "Israel", "Belarus", "Serbia", "Peru", "Hungary"]
countries.sort()

# Length
length_country_vector = len(countries)
print("# Countries", length_country_vector)

# Loop over countries
for i in range(len(countries)):
    # Country C02, population and GDP
    country_c02 = data.loc[(data['country'] == countries[i]) & (data['year'] >= 1970), "co2"]
    country_population = data.loc[(data['country'] == countries[i]) & (data['year'] >= 1970), "population"]
    country_gdp = data.loc[(data['country'] == countries[i]) & (data['year'] >= 1970), "gdp"]
    date_axis = np.linspace(1970, 2019, len(country_c02))

    # C02
    plt.plot(date_axis, country_c02)
    plt.xlabel("Time")
    plt.ylabel("Carbon dioxide")
    plt.title(countries[i]+"_CO2")
    plt.savefig(countries[i] + "_CO2")
    plt.show()

    # Population
    plt.plot(date_axis, country_population)
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title(countries[i] + "_population")
    plt.savefig(countries[i] + "_population")
    plt.show()

    # GDP
    plt.plot(date_axis, country_gdp)
    plt.xlabel("Time")
    plt.ylabel("GDP")
    plt.title(countries[i] + "_gdp")
    plt.savefig(countries[i] + "_gdp")
    plt.show()