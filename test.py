#%%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

#%% Load Dataset
df_0 = pd.read_parquet('datasets/taxi/yellow_tripdata_2024-07.parquet')
df_0.head()
#%%
df_1 = df_0.dropna()
missing_values = df_1.isnull().sum()
print(missing_values)
df_1.loc[:, 'tpep_pickup_datetime'] = pd.to_datetime(df_1['tpep_pickup_datetime'], errors='coerce')
df_1.loc[:, 'tpep_dropoff_datetime'] = pd.to_datetime(df_1['tpep_dropoff_datetime'], errors='coerce')
df_1.loc[:, 'time_difference'] = (df_1['tpep_dropoff_datetime'] - df_1['tpep_pickup_datetime']).dt.total_seconds()
invalid_time = df_1[df_1['time_difference'] < 60]
df_2 = df_1.loc[df_1['time_difference'] >= 60].copy()
df_2 = df_2.drop(columns=['time_difference'])
df_3 = df_2[
    (df_2['VendorID'].isin([1, 2])) &                  # VendorID should be 1 or 2
    (df_2['passenger_count'] > 0) &                    # Panssenger Count should be positive number
    (df_2['trip_distance'] >= 0.1) &                   # Trip Distance should larger than 0.1 miles
    (df_2['PULocationID'].notna()) &                   # PULocationID should not be N/A
    (df_2['DOLocationID'].notna()) &                   # DOLocationID should not be N/A
    (df_2['RatecodeID'].isin([1, 2, 3, 4, 5, 6])) &    # RateCodeID should between 1 to 6
    (df_2['store_and_fwd_flag'].isin(['Y', 'N'])) &    # store_and_fwd_flag should be Y or N
    (df_2['payment_type'].isin([1, 2, 3, 4, 5, 6])) &  # Payment_type should be 1 to 6
    (df_2['fare_amount'] >= 0) &                       # Fare amount should be non negative number
    (df_2['extra'] >= 0) &                             # Extra charges should be non negative number
    (df_2['mta_tax'] == 0.5) &                         # MTA tax should be 50 cent
    (df_2['improvement_surcharge'] >= 0) &             # Improvement surcharge should be non negative number
    (df_2['tip_amount'] >= 0) &                        # Tip amount should be non negative number
    (df_2['tolls_amount'] >= 0) &                      # Tolls amount should be non negative number
    (df_2['total_amount'] >= 0) &                      # Total amount should be non negative number
    (df_2['congestion_surcharge'] >= 0) &              # Congestion surcharge should be non negative number
    (df_2['Airport_fee'] >= 0)                         # Airport fee should be non negative number
]

features_to_clean = [
    'trip_distance', 'fare_amount', 'extra', 
    'tip_amount', 'tolls_amount', 'total_amount']

def remove_outliers(df, columns, upper_percentile=0.99):
    for col in columns:
        upper_bound = df[col].quantile(upper_percentile)
        # only keep data that less or equal to 99 precentage
        df = df[df[col] <= upper_bound]
    return df

df = remove_outliers(df_3, features_to_clean)

#%%
with open('datasets/taxi/taxi_zones.geojson') as f:
    taxi_zones = json.load(f)

# %%
pickup_counts = df['PULocationID'].value_counts().reset_index()
pickup_counts.columns = ['location_id', 'trip_counts']

# %%
for feature in taxi_zones['features']:
    feature['properties']['location_id'] = str(feature['properties']['location_id'])

# %%
fig = px.choropleth_mapbox(
    pickup_counts,
    geojson=taxi_zones,
    locations='location_id',
    color='trip_counts',
    featureidkey='properties.location_id',
    color_continuous_scale="YlOrRd",
    range_color=(0, pickup_counts['trip_counts'].max()),
    mapbox_style="carto-positron",
    zoom=9,
    center={"lat": 40.7128, "lon": -74.0060},
    opacity=0.6,
    labels={'trip_counts': 'Pickup Counts'}
)

fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show(renderer="browser")


#%% 
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import plotly.graph_objects as go

# Load subway data directory
subway_dir = 'datasets/subway/google_transit/'

# Load GTFS files for subway
routes_path = os.path.join(subway_dir, 'routes.txt')
shapes_path = os.path.join(subway_dir, 'shapes.txt')
trips_path = os.path.join(subway_dir, 'trips.txt')

# Load GTFS files
routes = pd.read_csv(routes_path)
shapes = pd.read_csv(shapes_path)
trips = pd.read_csv(trips_path)

# Map shape_id to route_id using trips.txt
shape_to_route = trips[['route_id', 'shape_id']].drop_duplicates()

# Merge shape_id with route_id
shapes = shapes.merge(shape_to_route, on='shape_id', how='left')

# Group shapes by shape_id to create LineStrings
route_shapes = shapes.groupby('shape_id')

# Create GeoDataFrame for subway routes
subway_routes_gdf = []
for shape_id, group in route_shapes:
    # Sort by sequence and create a LineString
    group = group.sort_values(by='shape_pt_sequence')
    line = LineString(zip(group['shape_pt_lon'], group['shape_pt_lat']))
    route_id = group['route_id'].iloc[0]  # Get the route_id for this shape
    subway_routes_gdf.append({'route_id': route_id, 'geometry': line})

subway_routes_gdf = gpd.GeoDataFrame(subway_routes_gdf)

# Merge with routes.txt to add route_color
subway_routes_gdf = subway_routes_gdf.merge(routes[['route_id', 'route_color']], on='route_id', how='left')

# Ensure route_color is valid HEX format
subway_routes_gdf['route_color'] = subway_routes_gdf['route_color'].fillna('000000')  # Default to black
subway_routes_gdf['route_color'] = subway_routes_gdf['route_color'].apply(lambda x: f"#{x.strip()}" if not x.startswith('#') else x)

# Debug: Check data integrity
print(subway_routes_gdf[['route_id', 'route_color']].head())

# Plotly Map: Add Subway Routes
fig = go.Figure()

for _, row in subway_routes_gdf.iterrows():
    if row['geometry'] and isinstance(row['geometry'], LineString):
        coords = list(row['geometry'].coords)
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[coord[0] for coord in coords],
            lat=[coord[1] for coord in coords],
            line=dict(width=3, color=row['route_color']),
            name=f"Subway Route {row['route_id']}"
        ))

# Update layout for Plotly Map
fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        zoom=9,
        center={"lat": 40.7128, "lon": -74.0060}
    ),
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

# Show the map
fig.show(renderer="browser")


#%%
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import plotly.graph_objects as go

# Base directory for GTFS data
base_dir = 'datasets/bus/'

# List all subdirectories under base_dir
##### focus on Manhattan
subdirs = ['gtfs_m']

# Initialize a list to store all routes
all_routes_gdf = []

# Process each subdirectory
for subdir in subdirs:
    print(f"Processing {subdir}...")
    
    # Paths for required GTFS files
    shapes_path = os.path.join(base_dir, subdir, 'shapes.txt')
    trips_path = os.path.join(base_dir, subdir, 'trips.txt')
    routes_path = os.path.join(base_dir, subdir, 'routes.txt')

    # Load GTFS files
    if not (os.path.exists(shapes_path) and os.path.exists(trips_path) and os.path.exists(routes_path)):
        print(f"Missing files in {subdir}, skipping...")
        continue

    shapes = pd.read_csv(shapes_path)
    trips = pd.read_csv(trips_path)
    routes = pd.read_csv(routes_path)

    # Map shape_id to route_id using trips.txt
    shape_to_route = trips[['route_id', 'shape_id']].drop_duplicates()

    # Merge shape_id with route_id
    shapes = shapes.merge(shape_to_route, on='shape_id', how='left')

    # Group shapes by shape_id to create LineStrings
    route_shapes = shapes.groupby('shape_id')

    # Create GeoDataFrame for routes
    routes_gdf = []
    for shape_id, group in route_shapes:
        # Sort by sequence and create a LineString
        group = group.sort_values(by='shape_pt_sequence')
        line = LineString(zip(group['shape_pt_lon'], group['shape_pt_lat']))
        route_id = group['route_id'].iloc[0]  # Get the route_id for this shape
        routes_gdf.append({'route_id': route_id, 'geometry': line})

    routes_gdf = gpd.GeoDataFrame(routes_gdf)

    # Merge with route_info to add route_color
    routes_gdf = routes_gdf.merge(routes[['route_id', 'route_color']], on='route_id', how='left')

    # Ensure route_color is valid HEX format
    routes_gdf['route_color'] = routes_gdf['route_color'].fillna('000000')  # Fill missing colors with black
    routes_gdf['route_color'] = routes_gdf['route_color'].apply(lambda x: f"#{x.strip()}" if not x.startswith('#') else x)

    # Add to the global list
    all_routes_gdf.append(routes_gdf)

# Combine all routes into a single GeoDataFrame
all_routes_gdf = pd.concat(all_routes_gdf, ignore_index=True)

# Debug: Check data integrity
print(all_routes_gdf[['route_id', 'route_color']].head())

# Plotly Map: Add Bus Routes
fig = go.Figure()

for _, row in all_routes_gdf.iterrows():
    if row['geometry'] and isinstance(row['geometry'], LineString):
        coords = list(row['geometry'].coords)
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[coord[0] for coord in coords],
            lat=[coord[1] for coord in coords],
            line=dict(width=2, color=row['route_color']),
            name=f"Bus Route {row['route_id']}"
        ))

# Update layout for Plotly Map
fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        zoom=9,
        center={"lat": 40.7128, "lon": -74.0060}
    ),
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

# Show the map
fig.show(renderer="browser")


# %%
# heatmap
fig = px.choropleth_mapbox(
    pickup_counts,
    geojson=taxi_zones,
    locations='location_id',
    color='trip_counts',
    featureidkey='properties.location_id',
    color_continuous_scale="YlOrRd",
    range_color=(0, pickup_counts['trip_counts'].max()),
    mapbox_style="carto-positron",
    zoom=9,
    center={"lat": 40.7128, "lon": -74.0060},
    opacity=0.6,
    labels={'trip_counts': 'Pickup Counts'}
)

# Add Subway Routes to Map
for _, row in subway_routes_gdf.iterrows():
    if row['geometry'] and isinstance(row['geometry'], LineString):
        coords = list(row['geometry'].coords)
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[coord[0] for coord in coords],
            lat=[coord[1] for coord in coords],
            line=dict(width=3, color=row['route_color']),
            name=f"Subway Route {row['route_id']}"
        ))

# Add Bus Routes to Map
for _, row in routes_gdf.iterrows():
    if row['geometry'] and isinstance(row['geometry'], LineString):
        coords = list(row['geometry'].coords)
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[coord[0] for coord in coords],
            lat=[coord[1] for coord in coords],
            line=dict(width=2, color=row['route_color']),
            name=f"Bus Route {row['route_id']}"
        ))

# Update layout
fig.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    mapbox=dict(
        style="carto-positron",
        zoom=11,
        center={"lat": 40.7637, "lon": -73.9814}
    ),
    coloraxis_showscale=False
)

# Show Map
fig.show(renderer="browser")
# %%
