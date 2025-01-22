# project.py


import pandas as pd
import numpy as np
from pathlib import Path

###
from collections import deque
from shapely.geometry import Point
###

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pd.options.plotting.backend = 'plotly'

import geopandas as gpd

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------

#def helper_calculate_length(trip_id):
    #total = trip_data["shape_dist_traveled"].sum()
    #return total
def create_detailed_schedule(schedule, stops, trips, bus_lines):
    merged_trip_stops=pd.merge(stops, schedule, on='stop_id', how='left')
    merged_trip_stops=pd.merge(merged_trip_stops,trips, on="trip_id", how='left')
    merged_trip_stops=merged_trip_stops[merged_trip_stops["route_id"].isin(bus_lines)]
    merged_trip_stops['route_id'] = pd.Categorical(merged_trip_stops['route_id'], categories=bus_lines, ordered=True)
    merged_trip_stops = merged_trip_stops.sort_values('route_id')
    trip_lengths = merged_trip_stops.groupby('trip_id')['shape_dist_traveled'].sum().reset_index()
    trip_lengths.rename(columns={'shape_dist_traveled': 'trip_length'}, inplace=True)
    merged_trip_stops = pd.merge(merged_trip_stops, trip_lengths[['trip_id', 'trip_length']], on='trip_id', how='left')
    merged_trip_stops=merged_trip_stops.sort_values(by=['route_id', 'trip_length'], ascending=[True, True])
    merged_trip_stops=merged_trip_stops.set_index("trip_id")
    merged_trip_stops=merged_trip_stops.drop(columns="shape_dist_traveled")
    merged_trip_stops=merged_trip_stops.rename(columns={"trip_length":"shape_dist_traveled"})
    return merged_trip_stops


    

def visualize_bus_network(bus_df):
    # Load the shapefile for San Diego city boundary
    san_diego_boundary_path = 'data/data_city/data_city.shp'
    san_diego_city_bounds = gpd.read_file(san_diego_boundary_path)
    
    # Ensure the coordinate reference system is correct
    san_diego_city_bounds = san_diego_city_bounds.to_crs("EPSG:4326")
    
    san_diego_city_bounds['lon'] = san_diego_city_bounds.geometry.apply(lambda x: x.centroid.x)
    san_diego_city_bounds['lat'] = san_diego_city_bounds.geometry.apply(lambda x: x.centroid.y)
    
    fig = go.Figure()
    
    # Add city boundary
    fig.add_trace(go.Choroplethmapbox(
        geojson=san_diego_city_bounds.__geo_interface__,
        locations=san_diego_city_bounds.index,
        z=[1] * len(san_diego_city_bounds),
        colorscale="Greys",
        showscale=False,
        marker_opacity=0.5,
        marker_line_width=1,
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": san_diego_city_bounds['lat'].mean(), "lon": san_diego_city_bounds['lon'].mean()},
            zoom=10,
        ),
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    # Get's unique bus lines from df
    bus_lines = np.array(bus_df['route_id'].unique())

    # Assign colors to each bus line
    color_palette = px.colors.qualitative.Plotly
    color_dict = {line: color_palette[i] for i, line in enumerate(bus_lines)}


    # Create a Scattermapbox trace for each bus line
    for line in bus_lines:
        line_data = bus_df[bus_df['route_id'] == line]
        fig.add_trace(go.Scattermapbox(
            lat=line_data['stop_lat'],
            lon=line_data['stop_lon'],
            mode='markers',
            name="Bus Line " + str(line),
            marker=go.scattermapbox.Marker(
                size=8,
                color=color_dict[line],
                opacity=0.7
            ),
            text=line_data['stop_name'],
            hoverinfo='text'
        ))


    return fig
    
    


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def find_neighbors(station_name, detailed_schedule):
     current_stop = detailed_schedule[detailed_schedule["stop_name"] == station_name]
     current_stop["next_stop"]=current_stop["stop_sequence"]+1
     detailed_schedule=detailed_schedule.reset_index()
     current_stop=current_stop.reset_index()
     new_df=pd.merge(detailed_schedule, current_stop,left_on=['trip_id', 'stop_sequence'], right_on=['trip_id','next_stop'])
     return new_df["stop_name_x"].unique()

    


def bfs(start_station, end_station, detailed_schedule):
    if start_station not in detailed_schedule["stop_name"].unique():
        return {f'Start station {start_station} not found.'}
    elif end_station not in detailed_schedule["stop_name"].unique():
        return {f'Start station {end_station} not found.'}
    
    queue = [(start_station, [start_station])]
    visited = []
    fastest_route = None
    
    while queue:
        current_station, route = queue.pop(0)  
        
        if current_station in visited:
            continue
        visited.append(current_station)
        
        if current_station == end_station:
            fastest_route=route
            break
        
        neighbors = find_neighbors(current_station, detailed_schedule)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, route + [neighbor]))
    
    detailed_schedule=detailed_schedule[detailed_schedule["stop_name"].isin(fastest_route)]
    detailed_schedule = detailed_schedule.drop_duplicates(subset='stop_name', keep='first')
    detailed_schedule=detailed_schedule[['stop_name','stop_lat','stop_lon']]
    detailed_schedule['stop_name'] = pd.Categorical(detailed_schedule['stop_name'], categories=fastest_route, ordered=True)
    detailed_schedule = detailed_schedule.sort_values('stop_name')
    detailed_schedule = detailed_schedule.reset_index(drop=True)
    detailed_schedule["stop_num"]=np.arange(1,detailed_schedule.shape[0]+1)
    #detailed_schedule['stop_num'] = detailed_schedule['stop_name'].apply(lambda x: fastest_route.index(x)+1)
    return detailed_schedule
        




# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def simulate_bus_arrivals(tau, seed=12):
    
    np.random.seed(seed) # Random seed -- do not change
    
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def simulate_wait_times(arrival_times_df, n_passengers):

    ...

def visualize_wait_times(wait_times_df, timestamp):
    ...
