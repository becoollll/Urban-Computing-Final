import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import glob
from pathlib import Path
from tqdm import tqdm
import os
import geopandas as gpd
from shapely.geometry import LineString

def load_datasets(data_path, start_date='2023-10', end_date='2024-09'):
    all_files = glob.glob(f'{data_path}/yellow_tripdata_*.parquet')
    dfs = []
    
    for file in sorted(all_files):
        file_date = Path(file).stem.split('_')[-1][:7]
        
        if start_date <= file_date <= end_date:
            df = pd.read_parquet(
                file, 
                columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID']
            )
            df['month'] = file_date
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def clean_taxi_data(df):
    print(f"Initial shape: {df.shape}")
    
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    df = df.dropna()
    print(f"After removing nulls: {df.shape}")
    
    df = df[(df['trip_duration'] >= 1) & (df['trip_duration'] <= 60)]
    print(f"After filtering duration: {df.shape}")
    
    df = df[(df['PULocationID'] > 0) & (df['DOLocationID'] > 0)]
    print(f"Final shape: {df.shape}")
    
    return df

def create_pickup_heatmap(df, taxi_zones):
    pickup_counts = df['PULocationID'].value_counts().reset_index()
    pickup_counts.columns = ['location_id', 'trip_counts']
    
    # Convert location_id to string in geojson
    for feature in taxi_zones['features']:
        feature['properties']['location_id'] = str(feature['properties']['location_id'])
    
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
    return fig, pickup_counts

def process_subway_data(subway_dir):
    # Load GTFS files for subway
    routes_path = os.path.join(subway_dir, 'routes.txt')
    shapes_path = os.path.join(subway_dir, 'shapes.txt')
    trips_path = os.path.join(subway_dir, 'trips.txt')

    routes = pd.read_csv(routes_path)
    shapes = pd.read_csv(shapes_path)
    trips = pd.read_csv(trips_path)

    # Map shape_id to route_id using trips.txt
    shape_to_route = trips[['route_id', 'shape_id']].drop_duplicates()
    shapes = shapes.merge(shape_to_route, on='shape_id', how='left')

    # Create GeoDataFrame for subway routes
    subway_routes_gdf = []
    for shape_id, group in shapes.groupby('shape_id'):
        group = group.sort_values(by='shape_pt_sequence')
        line = LineString(zip(group['shape_pt_lon'], group['shape_pt_lat']))
        route_id = group['route_id'].iloc[0]
        subway_routes_gdf.append({'route_id': route_id, 'geometry': line})

    subway_routes_gdf = gpd.GeoDataFrame(subway_routes_gdf)
    subway_routes_gdf = subway_routes_gdf.merge(routes[['route_id', 'route_color']], on='route_id', how='left')

    # Ensure route_color is valid HEX format
    subway_routes_gdf['route_color'] = subway_routes_gdf['route_color'].fillna('000000')
    subway_routes_gdf['route_color'] = subway_routes_gdf['route_color'].apply(
        lambda x: f"#{x.strip()}" if not x.startswith('#') else x
    )

    return subway_routes_gdf

def process_bus_data(base_dir, subdirs=['gtfs_m']):
    all_routes_gdf = []

    for subdir in subdirs:
        print(f"Processing {subdir}...")
        
        shapes_path = os.path.join(base_dir, subdir, 'shapes.txt')
        trips_path = os.path.join(base_dir, subdir, 'trips.txt')
        routes_path = os.path.join(base_dir, subdir, 'routes.txt')

        if not (os.path.exists(shapes_path) and os.path.exists(trips_path) and os.path.exists(routes_path)):
            print(f"Missing files in {subdir}, skipping...")
            continue

        shapes = pd.read_csv(shapes_path)
        trips = pd.read_csv(trips_path)
        routes = pd.read_csv(routes_path)

        shape_to_route = trips[['route_id', 'shape_id']].drop_duplicates()
        shapes = shapes.merge(shape_to_route, on='shape_id', how='left')

        routes_gdf = []
        for shape_id, group in shapes.groupby('shape_id'):
            group = group.sort_values(by='shape_pt_sequence')
            line = LineString(zip(group['shape_pt_lon'], group['shape_pt_lat']))
            route_id = group['route_id'].iloc[0]
            routes_gdf.append({'route_id': route_id, 'geometry': line})

        routes_gdf = gpd.GeoDataFrame(routes_gdf)
        routes_gdf = routes_gdf.merge(routes[['route_id', 'route_color']], on='route_id', how='left')

        routes_gdf['route_color'] = routes_gdf['route_color'].fillna('000000')
        routes_gdf['route_color'] = routes_gdf['route_color'].apply(
            lambda x: f"#{x.strip()}" if not x.startswith('#') else x
        )

        all_routes_gdf.append(routes_gdf)

    return pd.concat(all_routes_gdf, ignore_index=True)

def create_combined_map(pickup_counts, taxi_zones, subway_routes_gdf, bus_routes_gdf):
    fig = px.choropleth_mapbox(
        pickup_counts,
        geojson=taxi_zones,
        locations='location_id',
        color='trip_counts',
        featureidkey='properties.location_id',
        color_continuous_scale="YlOrRd",
        range_color=(0, pickup_counts['trip_counts'].max()),
        mapbox_style="carto-positron",
        zoom=11,
        center={"lat": 40.7637, "lon": -73.9814},
        opacity=0.6,
        labels={'trip_counts': 'Pickup Counts'}
    )

    # Add Subway Routes
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

    # Add Bus Routes
    for _, row in bus_routes_gdf.iterrows():
        if row['geometry'] and isinstance(row['geometry'], LineString):
            coords = list(row['geometry'].coords)
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=[coord[0] for coord in coords],
                lat=[coord[1] for coord in coords],
                line=dict(width=2, color=row['route_color']),
                name=f"Bus Route {row['route_id']}"
            ))

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        mapbox=dict(
            style="carto-positron",
            zoom=11,
            center={"lat": 40.7637, "lon": -73.9814}
        ),
        coloraxis_showscale=False
    )
    
    return fig

def main():
    # Load and process taxi data
    data_path = 'datasets/taxi'
    df = load_datasets(data_path)
    clean_df = clean_taxi_data(df)
    
    # Load taxi zones
    with open('datasets/taxi/taxi_zones.geojson') as f:
        taxi_zones = json.load(f)
    
    # Create heatmap and get pickup counts
    heatmap_fig, pickup_counts = create_pickup_heatmap(clean_df, taxi_zones)
    
    # Process subway data
    subway_dir = 'datasets/subway/google_transit/'
    subway_routes_gdf = process_subway_data(subway_dir)
    
    # Process bus data
    bus_dir = 'datasets/bus/'
    bus_routes_gdf = process_bus_data(bus_dir)
    
    # Create and show combined visualization
    combined_fig = create_combined_map(pickup_counts, taxi_zones, subway_routes_gdf, bus_routes_gdf)
    combined_fig.show(renderer="browser")

if __name__ == "__main__":
    main()