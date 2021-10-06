import pandas as pd
# from sklearn_extra.cluster import KMedoids
from Utilities import dendrogram_plot_labels
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("/Users/tassjames/Desktop/carbon_dioxide_emissions/owid-co2-data.csv")

countries = ["Iran", "Germany", "Indonesia", "South Korea",
             "Saudi Arabia", "Canada",
             "South Africa", "Brazil", "Mexico", "Australia", "Turkey", "United Kingdom", "Italy", "France", "Poland",
             "Kazakhstan", "Thailand",
             "Taiwan", "Spain", "Malaysia", "Pakistan", "Vietnam", "Egypt", "Ukraine", "Iraq", "United Arab Emirates",
             "Argentina", "Algeria",
             "Netherlands", "Philippines", "Nigeria", "Venezuela", "Uzbekistan", "Qatar", "Kuwait", "Colombia",
             "Bangladesh", "Czechia", "Belgium",
             "Turkmenistan", "Chile", "Romania", "Morocco", "Oman", "Austria"]
# "China", "United States", "India", "Russia", "Japan",
countries.sort()

# Length
length_country_vector = len(countries)
print("# Countries", length_country_vector)

# Loop over countries
co2_matrix = np.zeros((len(countries),len(countries)))
population_matrix = np.zeros((len(countries),len(countries)))
gdp_matrix = np.zeros((len(countries),len(countries)))

# Loop over countries
for i in range(len(countries)):
    for j in range(len(countries)):
        # Country C02, population and GDP for Country i
        country_co2_i = np.nan_to_num(np.array(data.loc[(data['country'] == countries[i]) & (data['year'] >= 1970), "co2"]))
        country_population_i = np.nan_to_num(np.array(data.loc[(data['country'] == countries[i]) & (data['year'] >= 1970), "population"]))
        country_gdp_i = np.nan_to_num(np.array(data.loc[(data['country'] == countries[i]) & (data['year'] >= 1970), "gdp"]))

        # Country C02, population and GDP for Country j
        country_co2_j = np.nan_to_num(np.array(data.loc[(data['country'] == countries[j]) & (data['year'] >= 1970), "co2"]))
        country_population_j = np.nan_to_num(np.array(data.loc[(data['country'] == countries[j]) & (data['year'] >= 1970), "population"]))
        country_gdp_j = np.nan_to_num(np.array(data.loc[(data['country'] == countries[j]) & (data['year'] >= 1970), "gdp"]))

        # Populate matrix entries
        co2_matrix[i,j] = np.nan_to_num(np.sum(np.abs(country_co2_i - country_co2_j)))
        population_matrix[i, j] = np.nan_to_num(np.sum(np.abs(country_population_i - country_population_j)))
        gdp_matrix[i, j] = np.nan_to_num(np.sum(np.abs(country_gdp_i - country_gdp_j)))
    print("Iteration", i)

# CO2 Matrix
fig, ax = plt.subplots()
im = ax.imshow(co2_matrix)
fig.colorbar(im)
# X ticks and y ticks
ax.set_xticks(np.arange(len(countries)))
ax.set_yticks(np.arange(len(countries)))
# Tick labels
ax.set_xticklabels(countries, fontsize=6, rotation=90)
ax.set_yticklabels(countries, fontsize=6)
plt.title("CO2 Matrix")
plt.savefig("CO2_Matrix")
plt.show()

# GDP Matrix
fig, ax = plt.subplots()
im = ax.imshow(gdp_matrix)
fig.colorbar(im)
# X ticks and y ticks
ax.set_xticks(np.arange(len(countries)))
ax.set_yticks(np.arange(len(countries)))
# Tick labels
ax.set_xticklabels(countries, fontsize=6, rotation=90)
ax.set_yticklabels(countries, fontsize=6)
plt.title("GDP Matrix")
plt.savefig("GDP_Matrix")
plt.show()

# Population matrix
fig, ax = plt.subplots()
im = ax.imshow(population_matrix)
fig.colorbar(im)
# X ticks and y ticks
ax.set_xticks(np.arange(len(countries)))
ax.set_yticks(np.arange(len(countries)))
# Tick labels
ax.set_xticklabels(countries, fontsize=6, rotation=90)
ax.set_yticklabels(countries, fontsize=6)
plt.title("Population Matrix")
plt.savefig("Population_Matrix")
plt.show()

# Compute MAX of each matrix
max_co2 = np.max(co2_matrix)
max_population = np.max(population_matrix)
max_gdp = np.max(gdp_matrix)

# Print maximum distance matrix elements
print("Max co2", max_co2)
print("Max Population", max_population)
print("Max GDP", max_gdp)

# Loop over countries and compute distance matrix
total_distance_matrix = np.zeros((len(countries), len(countries)))
for i in range(len(countries)):
    for j in range(len(countries)):
        # Country C02, population and GDP for Country i
        country_co2_i = np.nan_to_num(np.array(data.loc[(data['country'] == countries[i]) & (data['year'] >= 1970), "co2"]))
        country_population_i = np.nan_to_num(np.array(data.loc[(data['country'] == countries[i]) & (data['year'] >= 1970), "population"]))
        country_gdp_i = np.nan_to_num(np.array(data.loc[(data['country'] == countries[i]) & (data['year'] >= 1970), "gdp"]))

        # Country C02, population and GDP for Country j
        country_co2_j = np.nan_to_num(np.array(data.loc[(data['country'] == countries[j]) & (data['year'] >= 1970), "co2"]))
        country_population_j = np.nan_to_num(np.array(data.loc[(data['country'] == countries[j]) & (data['year'] >= 1970), "population"]))
        country_gdp_j = np.nan_to_num(np.array(data.loc[(data['country'] == countries[j]) & (data['year'] >= 1970), "gdp"]))

        total_distance_matrix[i,j] = max_co2 * np.sum(np.abs(country_co2_i - country_co2_j)) + \
        max_population * np.sum(np.abs(country_population_i-country_population_j)) + \
        max_gdp * (np.sum(np.abs(country_gdp_i-country_gdp_j)))
    print(i)

# Loop over trajectory matrix list
matrix = total_distance_matrix
fig, ax = plt.subplots()
im = ax.imshow(matrix)
fig.colorbar(im)
# X ticks and y ticks
ax.set_xticks(np.arange(len(countries)))
ax.set_yticks(np.arange(len(countries)))
# Tick labels
ax.set_xticklabels(countries, fontsize=6, rotation=90)
ax.set_yticklabels(countries, fontsize=6)
plt.title("Real and Carbon economy")
plt.savefig("Real_Carbon_Economy")
plt.show()

# Plot dendrogram
dendrogram_plot_labels(matrix, "real_carbon_economy_", "_Trajectory_", labels=countries)

# Multidimensional scaling
model = MDS(n_components=3, dissimilarity='precomputed', random_state=1)
mds_matrix = model.fit_transform(total_distance_matrix)

# Use silhouette score
range_n_clusters = list(range(2,10))
silhouette_list = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict(mds_matrix)
    centers = clusterer.cluster_centers_

    score = silhouette_score(mds_matrix, preds)
    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
    silhouette_list.append(score) # Append silhouette score to list

# Find argmax of list
n_clusters_opt = np.argmax(silhouette_list) + 2
clusterer = KMeans(n_clusters=n_clusters_opt)
preds = clusterer.fit_predict(mds_matrix)
print(preds)