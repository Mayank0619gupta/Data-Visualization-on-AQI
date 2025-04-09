import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Specify the full file path with the file name
file_path = "C:/Users/ANUGYA GUPTA/OneDrive/Desktop/python ca2/air quality index.csv"

# Verify file existence
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}. Please check the path and file name.")
    print("Files in directory:", os.listdir("C:/Users/ANUGYA GUPTA/OneDrive/Desktop/python ca2"))
else:
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Print the dataframe to verify
    print("Original DataFrame:")
    print(df)

    # Inspect the data
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())

    # Clean the data
    numeric_columns = ['pollutant_min', 'pollutant_max', 'pollutant_avg']  # Update these based on your columns
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].replace('NA', np.nan)  # Handle 'NA' as string
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].mean(), inplace=True)

    if 'last_update' in df.columns:
        df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')

    df.drop_duplicates(inplace=True)
    df.to_csv('cleaned_air_quality_index.csv', index=False)
    print("\nCleaned dataset saved as 'cleaned_air_quality_index.csv'")

    # Pivot the data
    index_cols = ['country', 'state', 'city', 'station', 'last_update', 'latitude', 'longitude']
    value_cols = ['pollutant_avg']
    pivot_cols = [col for col in df.columns if col in ['PM2.5', 'NO2', 'OZONE', 'CO', 'SO2', 'NH3', 'PM10']]

    if all(col in df.columns for col in index_cols) and 'pollutant_id' in df.columns and value_cols[0] in df.columns:
        df_pivot = df.pivot_table(index=index_cols,
                                 columns='pollutant_id',
                                 values=value_cols[0],
                                 aggfunc='mean').reset_index()
        df_pivot.columns.name = None
        df_pivot.rename(columns={'OZONE': 'O3'}, inplace=True)  # Explicitly rename OZONE to O3
        df_pivot[['PM2.5', 'NO2', 'O3', 'CO', 'NH3', 'PM10', 'SO2']].fillna(0, inplace=True)
        print("\nPivoted DataFrame:")
        print(df_pivot.head())
    else:
        print("Error: Required columns for pivoting are missing. Available columns:", df.columns)

    # Inspect pivoted data
    print("\nPivoted DataFrame Info:")
    print(df_pivot.info())
    print("\nMissing Values in Pivoted DataFrame:\n", df_pivot.isnull().sum())

    # Visualizations
    if all(col in df_pivot.columns for col in ['PM2.5', 'NO2', 'O3', 'CO', 'NH3', 'PM10', 'SO2']) and df_pivot[['PM2.5', 'NO2', 'O3', 'CO', 'NH3', 'PM10', 'SO2']].notna().any().any():
        # 1. Boxplot: Pollutant Distribution by State
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='state', y='PM2.5', data=df_pivot)
        plt.xticks(rotation=90)
        plt.title('PM2.5 Distribution by State')
        plt.xlabel('State')
        plt.ylabel('PM2.5 Levels')
        plt.show()

        # 2. Pie Chart: Proportion of Average Pollutant Levels
        pollutant_avg = df_pivot[['PM2.5', 'NO2', 'O3', 'CO', 'NH3', 'PM10', 'SO2']].mean()
        plt.figure(figsize=(8, 8))
        plt.pie(pollutant_avg, labels=pollutant_avg.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
        plt.title('Proportion of Average Pollutant Levels')
        plt.axis('equal')
        plt.show()

        # 3. Bar Chart: Pollutant Averages by State
        state_avg = df_pivot.groupby('state')[['PM2.5', 'NO2', 'O3']].mean().reset_index()
        state_avg_melted = state_avg.melt(id_vars='state', value_vars=['PM2.5', 'NO2', 'O3'], var_name='Pollutant', value_name='Average Level')
        plt.figure(figsize=(12, 6))
        sns.barplot(x='state', y='Average Level', hue='Pollutant', data=state_avg_melted)
        plt.xticks(rotation=90)
        plt.title('Average Pollutant Levels by State')
        plt.xlabel('State')
        plt.ylabel('Average Level')
        plt.legend(title='Pollutant')
        plt.show()

        # 4. Line Chart: Pollutant Trends (Simulated for single date)
        # Since there's only one date, simulate a trend by duplicating rows or using a range
        if len(df_pivot) > 1:
            plt.figure(figsize=(10, 6))
            for pollutant in ['PM2.5', 'NO2', 'O3']:
                plt.plot(df_pivot.index, df_pivot[pollutant], marker='o', label=pollutant)
            plt.title('Pollutant Trends Over Index')
            plt.xlabel('Index')
            plt.ylabel('Pollutant Level')
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print("Warning: Insufficient rows for a meaningful line chart. Consider adding more data.")

        # 5. Scatter Plot: Latitude vs. Longitude with PM2.5 Levels
        plt.figure(figsize=(10, 6))
        plt.scatter(df_pivot['longitude'], df_pivot['latitude'], c=df_pivot['PM2.5'], cmap='viridis', s=100)
        plt.colorbar(label='PM2.5 Level')
        plt.title('Geographical Distribution of PM2.5 Levels')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
    else:
        print("Error: Insufficient data or missing required columns for visualizations. Available columns with data:")
        print(df_pivot[['PM2.5', 'NO2', 'O3', 'CO', 'NH3', 'PM10', 'SO2']].notna().any())
