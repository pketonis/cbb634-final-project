import pandas as pd

processed_data = pd.read_csv('processed_data.csv')
coordinates = pd.read_csv('countries_raw.csv')

filtered_coordinates = coordinates[coordinates['name'].isin(processed_data['Country or region'])]

filtered_coordinates.to_csv('countries.csv', index=False)

filtered_coordinates.head()