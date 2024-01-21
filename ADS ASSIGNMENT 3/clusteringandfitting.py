# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:12:46 2024

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import norm


df = pd.read_csv("API_19_DS2_en_csv_v2_6300757.csv", skiprows=3)


def read_filter_data(indicator1, indicator2, year):
    """
    Reads the file and filters data for two different indicators and a specified year.

    Parameters:
        indicator1 (str): Name of the first indicator.
        indicator2 (str): Name of the second indicator.
        year (str): Year for which the data is filtered.

    Returns:
        pd.DataFrame: Filtered and merged data for the specified indicators and year."""
    df = pd.read_csv("API_19_DS2_en_csv_v2_6300757.csv", skiprows=3)
    data1 = df[df['Indicator Name'] == indicator1][[
        'Country Name', year]].rename(columns={year: indicator1})
    data2 = df[df['Indicator Name'] == indicator2][[
        'Country Name', year]].rename(columns={year: indicator2})
    merged_data = pd.merge(data1, data2, on='Country Name',
                           how='outer').reset_index(drop=True)
    filtered_data = merged_data.dropna(how='any').reset_index(drop=True)
    return filtered_data


data1 = read_filter_data('Renewable energy consumption (% of total final energy consumption)',
                         'Electric power consumption (kWh per capita)', '1990')
data2 = read_filter_data('Renewable energy consumption (% of total final energy consumption)',
                         'Electric power consumption (kWh per capita)', '2010')


def transpose_and_clean_data(input_file):
    
    # Transpose the DataFrame
     df_transposed = df.transpose()

    # Reset index and set the first row as column headers
     df_transposed.reset_index(inplace=True)
     df_transposed.columns = df_transposed.iloc[0]

    # Drop the first row (old column headers)
     df_transposed = df_transposed.drop(['Country Code'])

    # Remove any leading or trailing whitespaces in column names
     df_transposed.columns = df_transposed.columns.str.strip()
     return df_transposed

def cluster_and_plot_subplot(df1, df2, cluster_columns, num_clusters=4, title1='', title2=''):
    """
    Applies KMeans clustering to the given DataFrames and plots the clusters.
    
    Parameters:
        df1 (pd.DataFrame): First DataFrame for clustering.
        df2 (pd.DataFrame): Second DataFrame for clustering.
        cluster_columns (list): List of columns to be used for clustering.
        num_clusters (int): Number of clusters for KMeans.
        title1 (str): Title for the first subplot.
        title2 (str): Title for the second subplot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for df, title, ax in zip([df1, df2], [title1, title2], axes):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(df[cluster_columns])
        df['cluster'] = df['cluster'].astype('category')

        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_

        silhouette_avg = silhouette_score(df[cluster_columns], df['cluster'])
        print(f"Silhouette Score for {title}: {silhouette_avg}")

        # Plot the clusters
        scatter_plot = sns.scatterplot(x=cluster_columns[0], y=cluster_columns[1], hue='cluster', data=df, ax=ax)
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', s=50, c='purple', label='Cluster Centers')
        ax.set_title(title)

        # Create a legend
        legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=f'C{i}', markersize=8) for i in range(num_clusters)]
        legend_handles.append(Line2D([0], [0], marker='*', color='purple', markersize=8))
        legend_labels = [f'Cluster {i + 1}' for i in range(num_clusters)] + ['Cluster Centers']

        ax.legend(handles=legend_handles, labels=legend_labels, loc='upper left', fontsize='small', handlelength=0.5, handletextpad=0.5)
        
    # Adjust layout to prevent overlapping
    plt.tight_layout()

# Set a custom color palette for the plot
sns.set_palette("husl")

# Example usage
cluster_columns = ['Renewable energy consumption (% of total final energy consumption)', 'Electric power consumption (kWh per capita)']
cluster_and_plot_subplot(data1, data2, cluster_columns, title1='Renewable energy consumption and Electric power consumption in 1990', title2='Renewable energy consumption and Electric power consumption in 2010')

def calculate_inertia(data, max_clusters=10):
    """
    Calculates the inertia for different numbers of clusters in KMeans.
    
    Parameters:
        data (pd.DataFrame): Input data for clustering.
        max_clusters (int): Maximum number of clusters to consider.
    
    Returns:
        list: List of inertia values for each number of clusters.
    """
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias

X = data1[['Renewable energy consumption (% of total final energy consumption)', 'Electric power consumption (kWh per capita)']]
Y = data2[['Renewable energy consumption (% of total final energy consumption)', 'Electric power consumption (kWh per capita)']]

# Calculate inertia for the X DataFrame
inertias_X = calculate_inertia(X)

# Calculate inertia for the Y DataFrame
inertias_Y = calculate_inertia(Y)

# Create an elbow plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertias_X, marker='o', label='1990')
plt.plot(range(1, 11), inertias_Y, marker='o', label='2010')
plt.title('Elbow Plot for KMeans Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.legend()
plt.savefig('Elbow_Plot_for_KMeans_Clustering.png')
plt.show()

def plot_with_error_range(X, y, degree, ax, title, color, actual_data_color):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict values for all years (1990 to 2025)
    X_pred = poly_features.transform(pd.DataFrame(all_years_extended, columns=['Year']))
    forecast_values = model.predict(X_pred)

    # Compute the error range using bootstrapping
    n_bootstraps = 1000
    bootstrapped_predictions = np.zeros((n_bootstraps, len(X_pred)))

    for i in range(n_bootstraps):
        indices = np.random.choice(len(X), len(X))
        X_bootstrapped = X.iloc[indices]
        y_bootstrapped = y.iloc[indices]

        X_poly_bootstrapped = poly_features.transform(X_bootstrapped)
        model.fit(X_poly_bootstrapped, y_bootstrapped)
        bootstrapped_predictions[i, :] = model.predict(X_pred)

    lower_bound = np.percentile(bootstrapped_predictions, 2.5, axis=0)
    upper_bound = np.percentile(bootstrapped_predictions, 97.5, axis=0)

    # Plot actual data with a different color
    ax.plot(X, y, marker='o', linestyle='-.', label='Actual Data', color=actual_data_color)

    # Plot the fitted curve
    ax.plot(all_years_extended, forecast_values, label='Fitted Curve', linestyle='-', color=color)

    # Plot forecast for 2025
    prediction_2025 = forecast_values[-1]
    ax.plot(2025, prediction_2025, marker='+', markersize=8, label=f'Prediction for 2025: {prediction_2025:.2f}', color='black')

    # Plot error range
    ax.fill_between(all_years_extended, lower_bound, upper_bound, color=color, alpha=0.3, label='95% Confidence Interval')

    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel('Percent of the total')
    ax.set_xlim(1990, 2030)
    ax.set_xticks(range(1990, 2031, 5))  # Adjust the step as needed
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=7)

selected_countries = ['Pakistan', 'India', 'United States']
indicator_name = 'Renewable energy consumption (% of total final energy consumption)'

# Filter the data
data_selected = df[(df['Country Name'].isin(selected_countries)) & (df['Indicator Name'] == indicator_name)].reset_index(drop=True)

# Melt the DataFrame
data_forecast = data_selected.melt(id_vars=['Country Name', 'Indicator Name'], var_name='Year', value_name='Value')

# Filter out non-numeric values in the 'Year' column
data_forecast = data_forecast[data_forecast['Year'].str.isnumeric()]

# Convert 'Year' to integers
data_forecast['Year'] = data_forecast['Year'].astype(int)

# Handle NaN values by filling with the mean value
data_forecast['Value'].fillna(data_forecast['Value'].mean(), inplace=True)

# Filter data for the years between 1990 and 2020
data_forecast = data_forecast[(data_forecast['Year'] >= 1990) & (data_forecast['Year'] <= 2020)]

# Create a dictionary to store predictions for each country
predictions = {}

# Extend the range of years to include 2025
all_years_extended = list(range(1990, 2026))

# Example usage
actual_data_colors = ['purple', 'green', 'violet']
colors = ['mediumspringgreen' ,'orange', 'beige']

for country, color, actual_data_color, colors in zip(selected_countries, sns.color_palette("pastel", len(selected_countries)), actual_data_colors, colors):
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Prepare data for the current country
    country_data = data_forecast[data_forecast['Country Name'] == country]
    X_country = country_data[['Year']]
    y_country = country_data['Value']
    
    # Plot with error range and different colors for actual data
    plot_with_error_range(X_country, y_country, degree=3, ax=ax, title=f'{indicator_name} Forecast for {country}', color=colors, actual_data_color=actual_data_color)
    
    # Save the figure with the title
    filename = f"{indicator_name}_Forecast_{country.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')
    
    # Show the plot
    plt.show()
