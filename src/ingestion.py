import pandas as pd
import ee
import json
import numpy as np
from tqdm import tqdm
import sys


# ----------------------------------------------
# Filepath settings
# Google Earth Engine project name setting
input_filepath = '/home/NAS/homes/ycchen-10014/data/flood_events/flood_events_2020-2025.csv'
output_filepath = '/home/chunen/HAZAMA/HAZAMA/outputs/data_ingestion.csv'
MY_GEE_PROJECT = 'oceanic-hash-467505-r2'


# ----------------------------------------------
# Initialize GEE (in order to access GAUL dataset in GEE)
def initialize_gee():
    try:
        ee.Initialize(project=MY_GEE_PROJECT)
        print("Earth Engine initialized successfully.")
    except Exception as e:
        print("you need to authenticate GEE:")
        ee.Authenticate()  # Need to authenticate only once
        ee.Initialize(project=MY_GEE_PROJECT)


# ----------------------------------------------
# Define a function to safely parse 'Admin Units'
def parse_admin_units_safe(x):
    if pd.isna(x) or x == '':
        return []
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return []


# ----------------------------------------------
# Extract information for adm1 and adm2 from the 'Admin Units' list of a certain event
def reorganize_admin_data(admin_list):
    # Check admin_list is valid list, return [{}] if not
    if not isinstance(admin_list, list) or not admin_list:
        return [{}]
    
    # Collect all ADM1 information (using set to remove duplicates, filter out None/Empty)
    adm1_names = sorted(list(set([x.get('adm1_name') for x in admin_list if x.get('adm1_name')])))
    adm1_codes = sorted(list(set([x.get('adm1_code') for x in admin_list if x.get('adm1_code')])))
    
    # Collect all ADM2 entries
    adm2_entries = [x for x in admin_list if x.get('adm2_name') or x.get('adm2_code')]
    
    result_rows = []
    
    # Case 1: Data has ADM2 (regardless of ADM1)
    if adm2_entries:
        for entry in adm2_entries:
            new_row = {
                'adm2_name': entry.get('adm2_name'),
                'adm2_code': entry.get('adm2_code'),
                # Integrate ADM1 list into each ADM2 entry
                'adm1_name_list': adm1_names if adm1_names else [],
                'adm1_code_list': adm1_codes if adm1_codes else []
            }
            result_rows.append(new_row)
            
    # Case 2: Data has no ADM2 but has ADM1
    elif adm1_names or adm1_codes:
        new_row = {
            'adm2_name': None,
            'adm2_code': None,
            'adm1_name_list': adm1_names,
            'adm1_code_list': adm1_codes
        }
        result_rows.append(new_row)
        
    # Case 3: Data has neither ADM2 nor ADM1
    else:
        result_rows.append({})
        
    return result_rows


# ----------------------------------------------
# Fields: 'start_date', 'end_date', 'event_id'
# Convert 'Admin Units' from string to Python List
# Processing the data
def preprocess_data(filepath):
    print(f"Reading data from {filepath}...")
    emdat_data = pd.read_csv(filepath)
    
    # ----------------------------------------------
    emdat_derived = emdat_data[['DisNo.', 'ISO', 'Country', 'Location', 'Latitude', 'Longitude', 'Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day', 'Admin Units']].copy()

    # ----------------------------------------------
    emdat_derived['start_date'] = pd.to_datetime(emdat_derived[['Start Year', 'Start Month', 'Start Day']].rename(
        columns={'Start Year': 'year', 'Start Month': 'month', 'Start Day': 'day'}
    ))

    emdat_derived['end_date'] = pd.to_datetime(emdat_derived[['End Year', 'End Month', 'End Day']].rename(
        columns={'End Year': 'year', 'End Month': 'month', 'End Day': 'day'}
    ))

    emdat_derived['event_id'] = emdat_derived['DisNo.'].astype(str)

    # ----------------------------------------------
    print("Parsing and Restructuring Admin Units...")
    emdat_derived['admin_list_raw'] = emdat_derived['Admin Units'].apply(parse_admin_units_safe)
    emdat_derived['admin_list_structured'] = emdat_derived['admin_list_raw'].apply(reorganize_admin_data)
    # Explode the admin_list to have one row per admin unit
    emdat_exploded = emdat_derived.explode('admin_list_structured').reset_index(drop=True)
    # Expand the dictionaries in admin_list into separate columns (for easier processing)
    admin_details = pd.json_normalize(emdat_exploded['admin_list_structured'])
    emdat_final = pd.concat([emdat_exploded.drop(columns=['admin_list_raw', 'admin_list_structured']), admin_details], axis=1)
    
    return emdat_final


# ----------------------------------------------
# Field: 'bbox'
# Define a function to get bounding box from GEE
def get_bbox_from_gee(row, gaul_dataset):
    
    country = row['Country']

    adm2_code = row['adm2_code'] if pd.notna(row['adm2_code']) else None
    adm2_name = row['adm2_name'] if pd.notna(row['adm2_name']) else None
    adm1_list = row['adm1_name_list'] if isinstance(row.get('adm1_name_list'), list) and len(row['adm1_name_list']) > 0 else None
    
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
            filtered = gaul_dataset.filter(ee.Filter.eq('ADM2_CODE', code_int))
            
            if filtered.size().getInfo() > 0:
                target_feature = filtered.first()
                match_method = 'ADM2_CODE'
        except:
            pass

    # --- 2. If no adm2_code, use adm2_name + adm1_name + Country ---
    if target_feature is None and adm2_name is not None and adm1_list is not None:
        filtered = gaul_dataset.filter(ee.Filter.and_(
            ee.Filter.eq('ADM0_NAME', country),
            ee.Filter.eq('ADM2_NAME', adm2_name),
            ee.Filter.inList('ADM1_NAME', adm1_list)
        ))
        
        if filtered.size().getInfo() > 0:
            target_feature = filtered.first()
            match_method = 'ADM0&1&2_NAME'

    # --- 3. If no match above, use adm2_name + Country ---
    if target_feature is None and adm2_name is not None:
        filtered = gaul_dataset.filter(ee.Filter.and_(
            ee.Filter.eq('ADM0_NAME', country),
            ee.Filter.eq('ADM2_NAME', adm2_name)
        ))
        
        if filtered.size().getInfo() > 0:
            target_feature = filtered.first()
            match_method = 'ADM0&2_NAME'

    # --- 4. If no match by name, and coordinates are available, use coordinates lookup ---
    if target_feature is None and has_coords:
        point = ee.Geometry.Point([lon, lat])
        filtered = gaul_dataset.filterBounds(point)
        
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
# Main function
def main():
    # A. Initialize GEE
    initialize_gee()
    # Read GAUL dataset
    gaul = ee.FeatureCollection('FAO/GAUL/2015/level2')

    # B. Read and process data
    try:
        df_processed = preprocess_data(input_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {input_filepath}")
        sys.exit(1)   # Exit the program with an error code

    # C. Set the range to execute (test mode or full mode)
    df_to_process = df_processed.iloc[0:20].copy()
    # df_to_process = df_processed.copy()

    print(f"Start querying GEE for {len(df_to_process)} records...")
    tqdm.pandas()

    # D. Execute query (use lambda to pass gaul into the function)
    df_to_process['bbox_result'] = df_to_process.progress_apply(
        lambda row: get_bbox_from_gee(row, gaul), axis=1
    )

    # E. Organize results
    bbox_df = pd.json_normalize(df_to_process['bbox_result'])
    final_df = pd.concat([df_to_process.reset_index(drop=True), bbox_df], axis=1)

    # Check which locations could not be found
    missing_count = len(final_df[final_df['match_method'] == 'cannot_be_located'])
    print(f"There are **{missing_count}** records that could not be located")

    # F. Output file
    output = final_df[['event_id', 'start_date', 'end_date', 'bbox']]
    output.to_csv(output_filepath, index=False)
    print(f"The Data for ingestion is saved to {output_filepath}")


# ----------------------------------------------
# Run the main function
if __name__ == "__main__":
    main()