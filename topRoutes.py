import pandas as pd
import plotly.graph_objects as go
import glob
from pathlib import Path
import json
from tqdm import tqdm
import plotly.express as px
import numpy as np

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

def get_top_routes(df, taxi_zones_data, top_n=10):
    location_names = {}
    zone_centroids = {}
    
    # Create location ID to name mapping and calculate centroids
    for feature in taxi_zones_data['features']:
        location_id = str(feature['properties']['location_id'])
        zone_name = feature['properties']['zone']
        borough = feature['properties']['borough']
        location_names[location_id] = f"{zone_name}, {borough}"
        
        # Calculate centroid for each zone
        coordinates = feature['geometry']['coordinates'][0]
        if isinstance(coordinates[0][0], list):  # Handle MultiPolygon
            coordinates = coordinates[0]
        lat = sum(coord[1] for coord in coordinates) / len(coordinates)
        lon = sum(coord[0] for coord in coordinates) / len(coordinates)
        zone_centroids[location_id] = {'lat': lat, 'lon': lon}
    
    flows = df.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='trip_count')
    flows_sorted = flows.sort_values('trip_count', ascending=False)
    
    # Get top N flows with location names and coordinates
    top_flows = []
    for _, flow in flows_sorted.head(top_n).iterrows():
        pu_id = str(int(flow['PULocationID']))
        do_id = str(int(flow['DOLocationID']))
        
        if pu_id in location_names and do_id in location_names:
            top_flows.append({
                'pickup_location': location_names[pu_id],
                'dropoff_location': location_names[do_id],
                'trip_count': flow['trip_count'],
                'pickup_id': pu_id,
                'dropoff_id': do_id,
                'pickup_lat': zone_centroids[pu_id]['lat'],
                'pickup_lon': zone_centroids[pu_id]['lon'],
                'dropoff_lat': zone_centroids[do_id]['lat'],
                'dropoff_lon': zone_centroids[do_id]['lon']
            })
    
    return pd.DataFrame(top_flows), zone_centroids

def create_bezier_path(start, end, height_factor=0.2):
    # Calculate midpoint
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    
    # Calculate perpendicular vector for control point
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    # Rotate 90 degrees and scale by height_factor
    control_x = mid_x - dy * height_factor
    control_y = mid_y + dx * height_factor
    
    # Generate points along the curve
    t = np.linspace(0, 1, 50)
    x = (1-t)**2 * start[0] + 2*(1-t)*t * control_x + t**2 * end[0]
    y = (1-t)**2 * start[1] + 2*(1-t)*t * control_y + t**2 * end[1]
    
    return x, y

def create_top_routes_visualization(top_flows_df, taxi_zones, zone_centroids):
    fig = go.Figure()
    
    # Add zone boundaries
    unique_zones = set()
    for _, flow in top_flows_df.iterrows():
        unique_zones.add(flow['pickup_id'])
        unique_zones.add(flow['dropoff_id'])
    
    fig.add_trace(go.Choroplethmapbox(
        geojson=taxi_zones,
        locations=pd.DataFrame(taxi_zones['features'])['properties'].apply(lambda x: str(x['location_id'])),
        z=[1 if str(x['properties']['location_id']) in unique_zones else 0 
           for x in taxi_zones['features']],
        colorscale=[[0, '#E0E0E0'], [1, '#A0A0A0']],
        showscale=False,
        marker_opacity=0.5,
        marker_line_width=1,
        featureidkey='properties.location_id'
    ))
    
    # Find bidirectional routes
    route_pairs = {}
    for i, flow1 in top_flows_df.iterrows():
        for j, flow2 in top_flows_df.iterrows():
            if (i < j and 
                flow1['pickup_id'] == flow2['dropoff_id'] and 
                flow1['dropoff_id'] == flow2['pickup_id']):
                key = tuple(sorted([flow1['pickup_id'], flow1['dropoff_id']]))
                route_pairs[key] = (i, j)
    
    # Add flow lines with arrows
    max_count = top_flows_df['trip_count'].max()
    colors = px.colors.qualitative.Set3
    
    for i, flow in enumerate(top_flows_df.iterrows()):
        flow = flow[1]
        color = colors[i % len(colors)]
        width = 1 + (flow['trip_count'] / max_count) * 4  # Reduced line width
        
        start = (flow['pickup_lon'], flow['pickup_lat'])
        end = (flow['dropoff_lon'], flow['dropoff_lat'])
        
        # Check if this is part of a bidirectional pair
        route_key = tuple(sorted([flow['pickup_id'], flow['dropoff_id']]))
        is_bidirectional = route_key in route_pairs
        
        # Adjust curve height based on whether it's bidirectional
        height_factor = 0.2 if is_bidirectional else 0.1
        
        # Create curved path with more points for smoother arrows
        t = np.linspace(0, 1, 100)
        path_x, path_y = create_bezier_path(start, end, height_factor)
        
        # Calculate points for arrow markers along the path
        arrow_positions = [0.4, 0.6, 0.8]  # Multiple arrows along the path
        for pos in arrow_positions:
            idx = int(pos * (len(path_x) - 1))
            
            # Calculate direction for arrow
            if idx < len(path_x) - 1:
                angle = np.degrees(np.arctan2(
                    path_y[idx+1] - path_y[idx-1],
                    path_x[idx+1] - path_x[idx-1]
                ))
                
                fig.add_trace(go.Scattermapbox(
                    mode='markers',
                    lon=[path_x[idx]],
                    lat=[path_y[idx]],
                    marker=dict(
                        size=6,
                        symbol='triangle-right',
                        angle=angle,
                        color=color
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        fig.add_trace(go.Scattermapbox(
            mode='lines',
            lon=path_x,
            lat=path_y,
            line=dict(
                width=width,
                color=color
            ),
            opacity=0.7,
            hovertext=f"Route {i+1}<br>"
                     f"From: {flow['pickup_location']}<br>"
                     f"To: {flow['dropoff_location']}<br>"
                     f"Trips: {flow['trip_count']:,}",
            name=f"Route {i+1}"
        ))

        fig.add_trace(go.Scattermapbox(
            mode='markers',
            lon=[flow['pickup_lon']],
            lat=[flow['pickup_lat']],
            marker=dict(size=8, color=color, symbol='circle'),
            hovertext=f"Pickup: {flow['pickup_location']}",
            showlegend=False
        ))
        
        fig.add_trace(go.Scattermapbox(
            mode='markers',
            lon=[flow['dropoff_lon']],
            lat=[flow['dropoff_lat']],
            marker=dict(size=8, color=color, symbol='square'),
            hovertext=f"Dropoff: {flow['dropoff_location']}",
            showlegend=False
        ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            zoom=12,
            center={"lat": 40.76, "lon": -74}
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig

def main():
    data_path = 'datasets/taxi'
    df = load_datasets(data_path)
    clean_df = clean_taxi_data(df)
    
    with open('datasets/taxi/taxi_zones.geojson') as f:
        taxi_zones = json.load(f)
    
    top_10_flows, zone_centroids = get_top_routes(clean_df, taxi_zones, top_n=10)
    
    total_trips = len(clean_df)
    
    print("\nTop 10 Most Frequent Taxi Routes:")
    print("=" * 100)
    for i, flow in top_10_flows.iterrows():
        percentage = (flow['trip_count'] / total_trips) * 100
        print(f"\n{i+1}. From: {flow['pickup_location']}")
        print(f"   To: {flow['dropoff_location']}")
        print(f"   Number of trips: {flow['trip_count']:,}")
        print(f"   Percentage of total trips: {percentage:.2f}%")
        print(f"   Location IDs: {flow['pickup_id']} -> {flow['dropoff_id']}")
    
    print("\nCreating visualization...")
    fig = create_top_routes_visualization(top_10_flows, taxi_zones, zone_centroids)
    fig.show(renderer="browser")

if __name__ == "__main__":
    main()