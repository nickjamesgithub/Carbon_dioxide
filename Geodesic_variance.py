import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyemd import emd, emd_with_flow
from math import radians, cos, sin, asin, sqrt
import datetime as dt

data = pd.read_csv("/Users/tassjames/Desktop/carbon_dioxide_emissions/owid-co2-data.csv")

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

# CO2 list
co2_list = []
names = []
for i in range(len(countries)):
    # Country C02 - country i
    country_c02 = data.loc[(data['country'] == countries[i]) & (data['year'] >= 1970), "co2"]
    co2_list.append(country_c02)
    names.append(countries[i])

# Date axis
date_index = pd.date_range("1970-01-01", "2019-01-01", len(co2_list[0]))
date_index_array = np.array(date_index)

# Compute PDF matrix
co2_array = np.array(co2_list)
row_sums = co2_array.sum(axis=0)
new_pdf_matrix_df = pd.DataFrame(co2_array / row_sums[np.newaxis, :])

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6378000 # 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# Read in latitude and longitude data
country_co2_location_data = # pd.read_csv("/Users/tassjames/Desktop/India_USA_Brazil_COVID/USA/USA_States_latitude_longitude.csv")
usa_geographic_distance = np.zeros((len(usa_location_data), len(usa_location_data)))
for i in range(len(usa_location_data)):
    for j in range(len(usa_location_data)):
        lats_i = usa_location_data["Latitude"][i]
        lats_j = usa_location_data["Latitude"][j]
        longs_i = usa_location_data["Longitude"][i]
        longs_j = usa_location_data["Longitude"][j]
        distance = haversine(longs_i, lats_i, longs_j, lats_j)
        usa_geographic_distance[i, j] = distance

# Convert USA Geographic distance to a numpy array
usa_geographic_distance_array = np.array(usa_geographic_distance)

if time_series_smoother:
    # Smoothed new cases and deaths
    new_cases_smoothed = []
    new_deaths_smoothed = []
    for i in range(len(new_cases_df)):
        new_cases_smoothed.append(new_cases_df.iloc[i,:].rolling(window=7).mean())
        new_deaths_smoothed.append(new_deaths_df.iloc[i, :].rolling(window=7).mean())
    # Set N/As to 0
    new_cases = np.nan_to_num(new_cases_smoothed)
    new_deaths = np.nan_to_num(new_deaths_smoothed)

if individual_pdfs:

    # Compute the probability density function for all points in time
    # Throw out first 'burn-in' elements
    burn_in = 50
    new_cases_burn = new_cases[:, burn_in:]
    date_index_array_burn = date_index[burn_in:]

    # Loop over time series
    counter = 0
    geodesic_variance = []
    for t in range(len(new_cases_burn[0])): # Looping over time
        cases_slice = new_cases_burn[:, t] # Slice of cases in time
        cases_slice_pdf = np.nan_to_num(cases_slice / np.sum(cases_slice))

        # Country variance matrix
        country_variance = np.zeros((len(cases_slice_pdf), len(cases_slice_pdf)))

        # Loop over all the countries in the pdf
        for x in range(len(cases_slice_pdf)):
            for y in range(len(cases_slice_pdf)):
                # # Print state names
                # print(names_dc[x])
                # print(names_dc[y])
                country_x_density = cases_slice_pdf[x]
                country_y_density = cases_slice_pdf[y]
                lats_x = usa_location_data["Latitude"][x]
                lats_y = usa_location_data["Latitude"][y]
                longs_x = usa_location_data["Longitude"][x]
                longs_y = usa_location_data["Longitude"][y]
                geodesic_distance = haversine(longs_x, lats_x, longs_y, lats_y)

                # Compute country variance
                country_variance[x,y] = geodesic_distance**2 * country_x_density * country_y_density

        # Sum above the diagonal
        upper_distance_sum = np.triu(country_variance).sum() - np.trace(country_variance)
        geodesic_variance.append(upper_distance_sum)
        print("Iteration " + str(t) + " / " + str(len(new_cases_burn[0])))

    # Time-varying geodesic variance
    plt.plot(date_index_array_burn, geodesic_variance)
    plt.xlabel("Time (days)")
    plt.ylabel("Geodesic Wasserstein variance (m^2)")
    plt.title("Spatial variance")
    plt.savefig("Geodesic_variance_individual_PDF")
    plt.show()

if grouped_pdfs:

    # Compute the probability density function for all points in time
    # Throw out first 'burn-in' elements
    burn_in = 50
    window_length = 30
    new_cases_burn = new_cases[:, burn_in:]
    date_index_array_burn = date_index[burn_in+window_length:]

    # Loop over time series
    counter = 0
    geodesic_variance = []
    for t in range(len(new_cases_burn[0])-window_length): # Looping over time
        cases_slice = new_cases_burn[:, t:(t+window_length)] # Slice of cases in time
        cases_g = np.sum((cases_slice), axis=1)
        cases_slice_pdf = np.nan_to_num(cases_g / np.sum(cases_g))

        # Country variance matrix
        country_variance = np.zeros((len(cases_slice_pdf), len(cases_slice_pdf)))

        # Loop over all the countries in the pdf
        for x in range(len(cases_slice_pdf)):
            for y in range(len(cases_slice_pdf)):
                # # Print state names
                # print(names_dc[x])
                # print(names_dc[y])
                country_x_density = cases_slice_pdf[x]
                country_y_density = cases_slice_pdf[y]
                lats_x = usa_location_data["Latitude"][x]
                lats_y = usa_location_data["Latitude"][y]
                longs_x = usa_location_data["Longitude"][x]
                longs_y = usa_location_data["Longitude"][y]
                geodesic_distance = haversine(longs_x, lats_x, longs_y, lats_y)

                # Compute country variance
                country_variance[x,y] = geodesic_distance**2 * country_x_density * country_y_density

        # Sum above the diagonal
        upper_distance_sum = np.triu(country_variance).sum() - np.trace(country_variance)
        geodesic_variance.append(upper_distance_sum)
        print("Iteration " + str(t) + " / " + str(len(new_cases_burn[0])-window_length))

    # Time-varying geodesic variance
    plt.plot(date_index_array_burn, geodesic_variance)
    plt.xlabel("Time (days)")
    plt.ylabel("Geodesic Wasserstein variance (m$^2$)")
    plt.title("Spatial variance")
    plt.savefig("Geodesic_variance_grouped_PDF")
    plt.show()

    print("Minimum Variance", date_index_array_burn[np.argmin(geodesic_variance)])
