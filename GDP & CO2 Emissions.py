#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from scipy import stats


# In[2]:


# Load the GDP per capita data
gdp_data = pd.read_csv("gdp_data.csv", skiprows=4)
# Considering the most recent year available
gdp_data = gdp_data[["Country Name", "2019"]] 
gdp_data = gdp_data.rename(columns={"Country Name": "Country", "2019": "GDP per capita"})
gdp_data.head()


# In[3]:


#checking the number of columns and rows in the gdp dataset
gdp_data.shape


# In[4]:


#Load the CO2 Emissions data
co2_df = pd.read_csv("co2_emissions.csv", skiprows=4)
co2_df.head()
# Considering the most recent year available
co2_df = co2_df[["Country Name", "2019"]]  
co2_df = co2_df.rename(columns={"Country Name": "Country", "2019": "CO2 emissions per capita"})
co2_df.head()


# In[5]:


#checking the number of columns and rows in CO" emissions dataset
co2_df.shape


# In[6]:


# Merge the two dataframes on the "Country" column
df = pd.merge(gdp_data, co2_df, on="Country")


# In[7]:


#loading the merged dataset
df.head()


# In[8]:


# Remove rows with missing values
df.dropna(inplace=True)


# In[9]:


#checking if the null values have been removed
df.isnull().sum()


# In[10]:


# Normalize the data
df["GDP per capita (normalized)"] = (df["GDP per capita"] - df["GDP per capita"].mean()) / df["GDP per capita"].std()
df["CO2 emissions per capita (normalized)"] = (df["CO2 emissions per capita"] - df["CO2 emissions per capita"].mean()) / df["CO2 emissions per capita"].std()


# In[11]:


# Display the first few rows of the data
print(df.head())


# In[12]:


df.columns


# In[13]:


# Extract the relevant columns from the dataset
data = df[['GDP per capita', 'CO2 emissions per capita']].astype(float)

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Perform clustering using K-means
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(normalized_data)

# Add cluster labels to the dataframe
df['Cluster'] = labels

# Plot cluster membership and cluster centers
plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(df['GDP per capita'], df['CO2 emissions per capita'], c=labels, cmap='viridis', alpha=0.8)

# Plot cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red', label='Cluster Centers')

plt.xlabel('GDP per capita')
plt.ylabel('CO2 emissions per capita')
plt.title('Clustering Results of GDP per capita VS CO2 emissions per capita')
plt.legend()
plt.show()


# ##### This code performs the following steps:
# 
# 1. Extracts the relevant columns 'GDP per capita' and 'CO2 emissions per capita' from the dataset.
# 2. Normalizes the data using the StandardScaler.
# 3. Performs K-means clustering with 3 clusters.
# 4. Adds the cluster labels to the dataframe.
# 5. Plots the data points with different colors representing the cluster membership.
# 5. Plots the cluster centers in red.
# 
# The resulting plot will show the clustering results, where each data point is colored according to its assigned cluster, and the cluster centers are marked with 'X'. This visualization helps identify interesting clusters based on the normalized values of GDP per capita and CO2 emissions per capita.

# The dataset contains information on GDP per capita and CO2 emissions per capita for different countries. By applying clustering analysis, we aim to identify distinct patterns and group countries based on their GDP and CO2 emissions.
# 
# Upon analyzing the data, we find three clusters. Let's examine each cluster and understand the characteristics and trends within them.
# 
# #### Cluster 0:
# Countries in Cluster 0 are characterized by relatively low GDP per capita and low CO2 emissions per capita. These countries exhibit lower economic development and environmental impact compared to the other clusters. This cluster likely includes developing nations with limited industrialization and lower energy consumption.
# 
# ####  Cluster 1:
# Cluster 1 consists of countries with moderate to high GDP per capita and moderate CO2 emissions per capita. These countries demonstrate a balanced relationship between economic growth and environmental impact. They have achieved a certain level of industrialization and prosperity while managing their carbon footprint more effectively than Cluster 2.
# 
# #### Cluster 2:
# Countries in Cluster 2 exhibit high GDP per capita and high CO2 emissions per capita. This cluster includes countries with significant industrialization, high energy consumption, and a larger carbon footprint. These countries likely have a higher level of economic development and rely heavily on industries with substantial greenhouse gas emissions.
# 
# By examining the cluster centers (marked with red 'X' symbols), we can identify the representative characteristics of each cluster. The cluster centers represent the average values of GDP per capita and CO2 emissions per capita for the countries in each cluster.
# 
# The visual representation of the clusters allows us to identify similarities and differences in GDP per capita and CO2 emissions per capita across different countries. It helps us understand the relationship between economic development and environmental impact. By clustering countries based on these variables, we gain insights into different levels of economic and environmental performance.

# In[14]:


# Define the function for the model you want to fit
def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c


# Extract the relevant columns from the dataset
gdp_per_capita = df['GDP per capita'].astype(float)
co2_per_capita = df['CO2 emissions per capita'].astype(float)


# Define the function for estimating confidence ranges
def err_ranges(x, popt, pcov, alpha=0.05):
    perr = np.sqrt(np.diag(pcov))
    z_score = np.abs(stats.norm.ppf(alpha / 2))
    lower = polynomial_function(x, *(popt - z_score * perr))
    upper = polynomial_function(x, *(popt + z_score * perr))
    return lower, upper


# Fit the model to the data
popt, pcov = curve_fit(polynomial_function, gdp_per_capita, co2_per_capita)

# Make predictions for future values
gdp_per_capita_future = np.linspace(np.min(gdp_per_capita), np.max(gdp_per_capita), 100)
co2_per_capita_pred = polynomial_function(gdp_per_capita_future, *popt)

# Calculate confidence ranges
lower, upper = err_ranges(gdp_per_capita_future, popt, pcov)


# Plot the data, best fitting function, and confidence range
plt.figure(figsize=(10, 6))
plt.scatter(gdp_per_capita, co2_per_capita, label='Data')
plt.plot(gdp_per_capita_future, co2_per_capita_pred, color='red', label='Best Fit')
plt.fill_between(gdp_per_capita_future, lower, upper, color='gray', alpha=0.3, label='Confidence Range')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 emissions per capita')
plt.title('Polynomial Fit')
plt.legend()
plt.show()


# By performing this curve fitting and plotting, the code aims to provide insights into the relationship between these two variables and make predictions for future values while considering the uncertainty through the confidence range.

# #### Story:
# The dataset contains information on GDP per capita and CO2 emissions per capita for different countries. We are interested in understanding the relationship between these two variables and fitting a polynomial function to the data.
# 
# Upon analyzing the data and fitting the polynomial function, we can observe the following:
# 
# #### Data Scatter:
# The scatter plot shows the actual data points, with GDP per capita on the x-axis and CO2 emissions per capita on the y-axis. Each data point represents a specific country. We can visually see the general trend of the data, indicating the relationship between economic prosperity and environmental impact.
# 
# #### Best Fit Line:
# The red line represents the best-fit line or curve obtained by fitting a polynomial function to the data. This line represents the mathematical model that approximates the relationship between GDP per capita and CO2 emissions per capita. The polynomial function captures the overall trend and curvature observed in the data.
# 
# #### Confidence Range:
# The gray shaded area represents the confidence range around the best-fit curve. It provides an estimate of the uncertainty associated with the predicted CO2 emissions per capita based on the fitted polynomial function. The confidence range helps us understand the range of possible values within a certain level of confidence.
# 
# #### Interpretation:
# By examining the best-fit curve and the confidence range, we gain insights into the relationship between economic development (represented by GDP per capita) and environmental impact (represented by CO2 emissions per capita).
# 
# The upward or downward trend of the best-fit curve indicates the direction of the relationship. If the curve is upward, it suggests that as GDP per capita increases, so does CO2 emissions per capita, indicating a positive correlation between economic development and environmental impact. Conversely, if the curve is downward, it suggests a negative correlation.
# 
# The width of the confidence range provides information about the uncertainty associated with the predictions. A wider confidence range indicates higher uncertainty, while a narrower range indicates more precise predictions.
# 
# By analyzing the polynomial fit, we can gain insights into the expected CO2 emissions per capita for different levels of GDP per capita. This information can be valuable for policymakers and researchers in understanding the potential environmental consequences of economic growth and designing strategies for sustainable development.

# In[16]:


for l, u in zip(lower, upper):
    print(f"Lower limit: {l:.2f}, Upper limit: {u:.2f}")


# By printing and examining the lower and upper limits, we gain insights into the potential range of CO2 emissions per capita at different levels of economic prosperity. These limits help us understand the uncertainty associated with the predictions made by the fitted polynomial function.
# 
# For each GDP per capita value, the lower limit provides the minimum expected CO2 emissions per capita, while the upper limit provides the maximum expected value. The range between the lower and upper limits represents the uncertainty interval within which the actual CO2 emissions per capita can fall with a certain level of confidence.

# In[15]:


# Define the function for the model you want to fit
def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c


# Extract the relevant columns from the dataset
gdp_per_capita = df['GDP per capita'].astype(float)
co2_per_capita = df['CO2 emissions per capita'].astype(float)


# Define the function for estimating confidence ranges
def err_ranges(x, popt, pcov, alpha=0.05):
    perr = np.sqrt(np.diag(pcov))
    z_score = np.abs(stats.norm.ppf(alpha / 2))
    lower = polynomial_function(x, *(popt - z_score * perr))
    upper = polynomial_function(x, *(popt + z_score * perr))
    return lower, upper


# Perform clustering on the dataset
X = np.column_stack((gdp_per_capita, co2_per_capita))
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)

# Add cluster labels to the dataframe
df['Cluster'] = labels

# Initialize a list to store the fitting results for each cluster
fit_results = []

# Iterate over each cluster
for cluster_id in range(3):
    # Get the data points belonging to the current cluster
    cluster_data = df[df['Cluster'] == cluster_id]
    
    # Check if the cluster has sufficient data points for fitting a curve with three parameters
    if len(cluster_data) < 3:
        print(f"Cluster {cluster_id} does not have enough data points for fitting a curve.")
        continue
    
    # Fit the model to the data
    popt, pcov = curve_fit(polynomial_function, cluster_data['GDP per capita'], cluster_data['CO2 emissions per capita'])
    
    # Store the fitting results
    fit_results.append((popt, pcov))

# Plot the data, best fitting function, and confidence range for each cluster
plt.figure(figsize=(10, 6))
for cluster_id, fit_result in enumerate(fit_results):
    popt, pcov = fit_result
    cluster_data = df[df['Cluster'] == cluster_id]
    gdp_per_capita_future = np.linspace(np.min(gdp_per_capita), np.max(gdp_per_capita), 100)
    co2_per_capita_pred = polynomial_function(gdp_per_capita_future, *popt)
    lower, upper = err_ranges(gdp_per_capita_future, popt, pcov)
    plt.scatter(cluster_data['GDP per capita'], cluster_data['CO2 emissions per capita'], label=f'Cluster {cluster_id}')
    plt.plot(gdp_per_capita_future, co2_per_capita_pred, label=f'Cluster {cluster_id} - Best Fit')
    plt.fill_between(gdp_per_capita_future, lower, upper, color='gray', alpha=0.3, label=f'Cluster {cluster_id} - Confidence Range')

plt.xlabel('GDP per capita')
plt.ylabel('CO2 emissions per capita')
plt.title('Polynomial Fit by Cluster')
plt.legend()
plt.show()


# Calculate average GDP per capita and CO2 emissions per capita for each cluster
cluster_averages = df.groupby('Cluster')[['GDP per capita', 'CO2 emissions per capita']].mean()

# Plot the average GDP per capita for each cluster
plt.figure(figsize=(12, 6))
cluster_averages['GDP per capita'].plot(kind='bar', color='skyblue')
plt.title('Average GDP per Capita by Cluster')
plt.ylabel('Average GDP per Capita')
plt.xlabel('Cluster')
plt.show()

# Plot the average CO2 emissions per capita for each cluster
plt.figure(figsize=(12, 6))
cluster_averages['CO2 emissions per capita'].plot(kind='bar', color='salmon')
plt.title('Average CO2 Emissions per Capita by Cluster')
plt.ylabel('Average CO2 Emissions per Capita')
plt.xlabel('Cluster')
plt.show()


# #### Story:
# In order to gain a deeper understanding of the relationship between GDP per capita and CO2 emissions per capita, we performed clustering on the dataset. By grouping similar countries together, we aimed to identify distinct patterns and trends within different clusters.
# 
# #### Clustering Process:
# We used the K-means clustering algorithm to partition the data into three clusters based on their GDP per capita and CO2 emissions per capita values. The clustering process allowed us to identify groups of countries that share similar characteristics in terms of economic prosperity and carbon emissions.
# 
# #### Fitting Polynomial Functions:
# For each cluster, we fit a polynomial function to capture the underlying relationship between GDP per capita and CO2 emissions per capita. The polynomial function used has three parameters: a, b, and c, which determine the shape of the curve. We employed the curve_fit function to estimate the optimal parameter values for each cluster.
# 
# #### Confidence Range:
# To assess the uncertainty associated with the fitted curves, we calculated confidence ranges. The err_ranges function was defined to estimate the lower and upper limits of the confidence range based on the parameter values and their covariance matrix. These limits provide insight into the potential range of CO2 emissions per capita corresponding to different levels of GDP per capita, taking into account the uncertainty of the fitted model.
# 
# #### Plotting the Results:
# We visualized the clustering results and fitted curves for each cluster. In the plot, each cluster is represented by a different color, with data points indicating the actual GDP per capita and CO2 emissions per capita values. The fitted curve for each cluster is plotted using the best-fit parameters, and the confidence range is displayed as a shaded region around the curve.
# 
# #### Interpretation:
# The clustering results and fitted curves reveal distinct patterns and trends within different clusters. By examining the plot, we can observe the relationship between GDP per capita and CO2 emissions per capita for each cluster separately. This allows us to identify variations in the shape and magnitude of the relationship across clusters.
# 
# Moreover, the confidence ranges provide valuable insights into the uncertainty associated with the fitted curves. The shaded regions around the curves indicate the range within which the actual CO2 emissions per capita can be expected to fall with a certain level of confidence. These ranges help us understand the potential variability in carbon emissions corresponding to different levels of economic prosperity within each cluster.

# In[ ]:



