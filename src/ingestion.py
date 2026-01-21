import pandas as pd
import ee
import json
import numpy as np
from tqdm import tqdm


# ----------------------------------------------
# Input data
# Please modify the path to your local EMDAT flood events CSV file
input_filepath = '/home/NAS/homes/ycchen-10014/data/flood_events/flood_events_2020-2025.csv'
emdat_data = pd.read_csv(input_filepath)
emdat_derived = emdat_data[['DisNo.', 'ISO', 'Country', 'Location', 'Latitude', 'Longitude', 'Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day', 'Admin Units']]


# ----------------------------------------------
# Fields: 'start_date', 'end_date', 'event_id'
emdat_derived['start_date'] = pd.to_datetime(emdat_derived[['Start Year', 'Start Month', 'Start Day']].rename(
    columns={'Start Year': 'year', 'Start Month': 'month', 'Start Day': 'day'}
))

emdat_derived['end_date'] = pd.to_datetime(emdat_derived[['End Year', 'End Month', 'End Day']].rename(
    columns={'End Year': 'year', 'End Month': 'month', 'End Day': 'day'}
))

emdat_derived['event_id'] = emdat_derived['DisNo.'].astype(str)


# ----------------------------------------------
# Define a function to safely parse 'Admin Units'
def parse_admin_units_safe(x):
    if pd.isna(x) or x == '':
        return []
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return []

# Convert 'Admin Units' from string to Python List
emdat_derived['admin_list'] = emdat_derived['Admin Units'].apply(parse_admin_units_safe)
# Explode the admin_list to have one row per admin unit
emdat_exploded = emdat_derived.explode('admin_list').reset_index(drop=True)
# Expand the dictionaries in admin_list into separate columns (for easier processing)
admin_details = pd.json_normalize(emdat_exploded['admin_list'])
emdat_final = pd.concat([emdat_exploded, admin_details], axis=1)


# ----------------------------------------------
# Access GAUL dataset in GEE
try:
    ee.Initialize()
except Exception as e:
    print("you need to authenticate GEE:")
    ee.Authenticate()  # Need to authenticate only once
    ee.Initialize()

gaul = ee.FeatureCollection('FAO/GAUL/2015/level2')


# ----------------------------------------------
# Field: 'bbox'
# Define a function to get bounding box from GEE
def get_bbox_from_gee(row):
    
    country = row['Country']

    adm2_code = row['adm2_code'] if pd.notna(row['adm2_code']) else None
    adm2_name = row['adm2_name'] if pd.notna(row['adm2_name']) else None
    adm1_name = row['adm1_name'] if pd.notna(row['adm1_name']) else None
    
    try:
        lon = float(row['Longitude'])
        lat = float(row['Latitude'])
        has_coords = not (np.isnan(lon) or np.isnan(lat))
    except (ValueError, TypeError):
        has_coords = False
        lon, lat = None, None
    
    # Set up initial variables
    target_feature = None
    match_method = None

    # --- 1. Comparing adm2_code ---
    if adm2_code is not None:
        try:
            code_int = int(adm2_code)
            filtered = gaul.filter(ee.Filter.eq('ADM2_CODE', code_int))
            
            if filtered.size().getInfo() > 0:
                target_feature = filtered.first()
                match_method = 'ADM2_CODE'
        except:
            pass

    # --- 2. If no adm2_code, use adm2_name + adm1_name + Country ---
    if target_feature is None and adm2_name is not None and adm1_name is not None:
        filtered = gaul.filter(ee.Filter.and_(
            ee.Filter.eq('ADM0_NAME', country),
            ee.Filter.eq('ADM2_NAME', adm2_name),
            ee.Filter.eq('ADM1_NAME', adm1_name)
        ))
        
        if filtered.size().getInfo() > 0:
            target_feature = filtered.first()
            match_method = 'ADM0&1&2_NAME'

    # --- 3. If no match above, use adm2_name + Country ---
    if target_feature is None and adm2_name is not None:
        filtered = gaul.filter(ee.Filter.and_(
            ee.Filter.eq('ADM0_NAME', country),
            ee.Filter.eq('ADM2_NAME', adm2_name)
        ))
        
        if filtered.size().getInfo() > 0:
            target_feature = filtered.first()
            match_method = 'ADM0&2_NAME'

    # --- 4. If no match by name, and coordinates are available, use coordinates lookup ---
    if target_feature is None and has_coords:
        point = ee.Geometry.Point([lon, lat])
        filtered = gaul.filterBounds(point)
        
        if filtered.size().getInfo() > 0:
            target_feature = filtered.first()
            match_method = 'Coordinates_Lookup'
    
    # --- 5. If no match found, return error message ---
    if target_feature is None:
        # print(f"event_id: {event_id} cannot be located")
        return {
            'bbox': [[0, 0], [0, 0], [0, 0], [0, 0]],
            'match_method': 'cannot_be_located'
        }

    # --- Getting the bounding box ---
    try:
        geom = target_feature.geometry()
        # bounds: [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
        bounds = geom.bounds().coordinates().get(0).getInfo()
        
        return {
            'bbox': bounds[0:4],
            'match_method': match_method
        }
    except Exception as e:
        # If an internal GEE geometry error occurs, also mark as failure
        return {
            'bbox': [[0, 0], [0, 0], [0, 0], [0, 0]],
            'match_method': f'error_{str(e)}'
        }
    

# ----------------------------------------------
# Process the DataFrame with progress bar
tqdm.pandas()

df_to_process = emdat_final.iloc[0:10].copy() # for testing, take first 10 rows
# df_to_process = emdat_final.copy()

print("Start querying GEE for the Bounding Box (this will take a little time)...")

# Use apply to perform the query
# The result will be stored as a dict, which will be unpacked later
df_to_process['bbox_result'] = df_to_process.progress_apply(get_bbox_from_gee, axis=1)

# Split the results into separate columns
bbox_df = pd.json_normalize(df_to_process['bbox_result'])
final_df = pd.concat([df_to_process.reset_index(drop=True), bbox_df], axis=1)

# Check which locations could not be found
missing_locations = final_df[final_df['match_method'] == 'cannot_be_located']
print(f"There are {len(missing_locations)} records that could not be located")


# ----------------------------------------------
# Output DataFrame with relevant fields
# Please modify the output path as needed
output = final_df[['event_id', 'start_date', 'end_date', 'bbox']]
output_filepath = '/home/chunen/HAZAMA/HAZAMA/outputs/data_ingestion.csv'
output.to_csv(output_filepath, index=False)
print(f"The Data for ingestion is saved to {output_filepath}")