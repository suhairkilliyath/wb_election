import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from shapely.geometry import Point
from geopy.distance import geodesic
import pyproj
from pyproj import Geod
from pyproj import CRS, Transformer
import numpy as np


def load_data(file_path):
    lokniti_respondents = gpd.read_file(file_path)
    lokniti_localities = lokniti_respondents[['loca', 'psu_id', 'geometry']].drop_duplicates()
    return lokniti_localities

def load_polling_data(csv_path):
    df_polling = pd.read_csv(csv_path)
    df_polling = df_polling[~df_polling['latitude'].isna()]
    df_polling = df_polling[df_polling['district_name_21'] != 'DARJEELING']
    gdf_polling = gpd.GeoDataFrame(df_polling, geometry=gpd.points_from_xy(df_polling.longitude, df_polling.latitude))
    gdf_polling = gdf_polling.set_crs('EPSG:4326')
    return gdf_polling

def clip_polling_data(gdf_polling, wb_state_path):
    wb_state = gpd.read_file(wb_state_path)
    gdf_polling_clipped = gpd.clip(gdf_polling, wb_state)
    return gdf_polling_clipped

def separate_localities(lokniti_localities):
    urban_localities = lokniti_localities[lokniti_localities['loca'] == '2: Ward']
    rural_localities = lokniti_localities[lokniti_localities['loca'] == '1: Village']
    return urban_localities, rural_localities

def min_distance(point, gdf):
    nearest = gdf.geometry == point
    geod = Geod(ellps="WGS84")
    return gdf[~nearest].geometry.apply(lambda x: geod.inv(point.x, point.y, x.x, x.y)[-1]).min()

def calculate_distances(localities):
    distances = localities.geometry.apply(lambda point: min_distance(point, localities))
    return distances

def point_buffer(point, radius_m):
    lon, lat = point.x, point.y
    geod = Geod(ellps='WGS84')
    num_vtxs = 64
    lons, lats, _ = geod.fwd(np.repeat(lon, num_vtxs), np.repeat(lat, num_vtxs),
                             np.linspace(360, 0, num_vtxs), np.repeat(radius_m, num_vtxs), radians=False)
    return Polygon(zip(lons, lats))

def create_buffers(lokniti_localities, buffer_distance):
    lokniti_localities['buffered_geometry'] = lokniti_localities['geometry'].apply(
        lambda point: point_buffer(point, buffer_distance)
    )
    return lokniti_localities['buffered_geometry']

def visualize_data(wb_state, gdf_polling_clipped, lokniti_localities):
    ax = wb_state.plot(color='blue', alpha=0.5, figsize=(10, 10))
    lokniti_localities.plot(ax=ax, color='blue', alpha=0.5, figsize=(10, 10))
    lokniti_localities.set_geometry('buffered_geometry').plot(ax=ax, color='red', alpha=0.5)
    gdf_polling_clipped.plot(ax=ax, color='yellow', markersize=1, legend=True)
    plt.show()

def perform_spatial_join(gdf_polling_clipped, lokniti_localities):
    lokniti_buffered = lokniti_localities.set_geometry('buffered_geometry')
    return gpd.sjoin(gdf_polling_clipped, lokniti_buffered, how='inner', predicate='intersects')


def calculate_distances_between_points(gdf_polling_survey_joined):

    def calculate_distance(row):
        # Extract the coordinates from the two Point columns in the row
        point1 = (row['geometry_left'].y, row['geometry_left'].x)
        point2 = (row['geometry_right'].y, row['geometry_right'].x)
        
        # Calculate the geodesic distance in kilometers
        distance_km = geodesic(point1, point2).kilometers
        
        return distance_km
    # Apply the calculate_distance function to each row in the DataFrame
    distances = gdf_polling_survey_joined.apply(calculate_distance, axis=1)

    gdf_polling_survey_joined['distance_km'] = distances
    return gdf_polling_survey_joined

def main():
    lokniti_localities = load_data(survey_file_path)
    gdf_polling = load_polling_data(polling_data_file_path)

    gdf_polling_clipped = clip_polling_data(gdf_polling, wb_state_path)
    urban_localities, rural_localities = separate_localities(lokniti_localities)

    urban_distances = calculate_distances(urban_localities)
    rural_distances = calculate_distances(rural_localities)

    buffer_distance = min(rural_distances) / 2.0

    lokniti_localities['buffered_geometry'] = create_buffers(lokniti_localities, buffer_distance=buffer_distance)
    wb_state = gpd.read_file(wb_state_path)
    visualize_data(wb_state, gdf_polling_clipped, lokniti_localities)

    gdf_polling_survey_joined = perform_spatial_join(gdf_polling_clipped, lokniti_localities)
    gdf_polling_survey_joined = calculate_distances_between_points(gdf_polling_survey_joined)

    gdf_polling_survey_joined.to_csv('localities_polling_joined.csv')

    distance_stats = gdf_polling_survey_joined['distance_km'].describe()
    print("Descriptive stat for distance_km:")
    print(distance_stats)

    # Average frequency of polling booth per locality type
    frequency_per_loca = (
        gdf_polling_survey_joined
        .groupby(['loca', 'psu_id'])
        .size()
        .reset_index(name='frequency')
        .groupby('loca')['frequency']
        .mean()
    )
    print("Average frequency of polling booth per locality type")
    print(frequency_per_loca)

if __name__ == "__main__":
    survey_file_path = 'lokniti_respondents/lokniti_respondents.shp'
    polling_data_file_path = 'WB2019_PSresults.csv'
    wb_state_path = '/Users/suhairkilliyath/Downloads/map.geojson'
    main()
