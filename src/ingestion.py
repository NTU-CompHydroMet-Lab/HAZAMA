import json
import sys

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------------------------
# Initialize GEE (in order to access GAUL dataset in GEE)
def initialize_gee(gee_project):
    try:
        ee.Initialize(project=gee_project)
        print("Earth Engine initialized successfully.")
    except Exception:
        print("you need to authenticate GEE:")
        ee.Authenticate()  # Need to authenticate only once
        ee.Initialize(project=gee_project)


# ----------------------------------------------
# Define a function to safely parse 'Admin Units' or 'GADM Admin Units'
def parse_admin_units_safe(x):
    if pd.isna(x) or x == "":
        return []
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return []


# ----------------------------------------------
# Extract information for adm1 and adm2 from the 'Admin Units', 'GADM Admin Units' list of a certain event
def reorganize_admin_data(admin_list, admin_list_gadm=None):
    # Check admin_list (GAUL) is valid list
    if isinstance(admin_list, list) and admin_list:
        # Collect all ADM1 information (using set to remove duplicates, filter out None/Empty)
        adm1_names = sorted(
            list(set([x.get("adm1_name") for x in admin_list if x.get("adm1_name")]))
        )
        adm1_codes = sorted(
            list(set([x.get("adm1_code") for x in admin_list if x.get("adm1_code")]))
        )
        # Collect all ADM2 entries
        adm2_entries = [
            x for x in admin_list if x.get("adm2_name") or x.get("adm2_code")
        ]
    else:
        adm1_names, adm1_codes, adm2_entries = [], [], []

    # ----------------------------------------------
    # Check admin_list (GADM) is valid list
    if isinstance(admin_list_gadm, list) and admin_list_gadm:
        # Collect all ADM1 information (using set to remove duplicates, filter out None/Empty)
        adm1_names_gadm = sorted(
            list(set([x.get("name_1") for x in admin_list_gadm if x.get("name_1")]))
        )
        adm1_codes_gadm = sorted(
            list(set([x.get("gid_1") for x in admin_list_gadm if x.get("gid_1")]))
        )
        # Collect all ADM2 entries
        adm2_entries_gadm = [
            x for x in admin_list_gadm if x.get("name_2") or x.get("gid_2")
        ]
    else:
        adm1_names_gadm, adm1_codes_gadm, adm2_entries_gadm = [], [], []

    # ----------------------------------------------
    result_rows = []

    # Ensure all cases have consistent keys
    def get_base_row():
        return {
            "adm2_name": None,
            "adm2_code": None,
            "adm1_name_list": None,
            "adm1_code_list": None,
            "adm2_name_gadm": None,
            "adm2_code_gadm": None,
            "adm1_name_list_gadm": None,
            "adm1_code_list_gadm": None,
        }

    # Case 1: Data has ADM2_GAUL (regardless of ADM1_GAUL)
    if adm2_entries:
        for entry in adm2_entries:
            row = get_base_row()
            row.update(
                {
                    "adm2_name": entry.get("adm2_name"),
                    "adm2_code": entry.get("adm2_code"),
                    "adm1_name_list": adm1_names if adm1_names else [],
                    "adm1_code_list": adm1_codes if adm1_codes else [],
                }
            )
            result_rows.append(row)

    # Case 2: Data has no ADM2_GAUL but has ADM2_GADM (regardless of ADM1)
    elif adm2_entries_gadm:
        for entry in adm2_entries_gadm:
            row = get_base_row()
            row.update(
                {
                    "adm2_name_gadm": entry.get("name_2"),
                    "adm2_code_gadm": entry.get("gid_2"),
                    # Keep both GAUL and GADM ADM1 lists context
                    "adm1_name_list": adm1_names if adm1_names else [],
                    "adm1_code_list": adm1_codes if adm1_codes else [],
                    "adm1_name_list_gadm": adm1_names_gadm if adm1_names_gadm else [],
                    "adm1_code_list_gadm": adm1_codes_gadm if adm1_codes_gadm else [],
                }
            )
            result_rows.append(row)

    # Case 3: Data has no ADM2_GAUL but has ADM1_GAUL (regardless of ADM1_GADM)
    elif adm1_names or adm1_codes:
        row = get_base_row()
        row.update(
            {
                "adm1_name_list": adm1_names,
                "adm1_code_list": adm1_codes,
                "adm1_name_list_gadm": adm1_names_gadm if adm1_names_gadm else [],
                "adm1_code_list_gadm": adm1_codes_gadm if adm1_codes_gadm else [],
            }
        )
        result_rows.append(row)

    # Case 4: Data has neither ADM2_GAUL nor ADM1_GAUL but has ADM1_GADM
    elif adm1_names_gadm or adm1_codes_gadm:
        row = get_base_row()
        row.update(
            {
                "adm1_name_list_gadm": adm1_names_gadm,
                "adm1_code_list_gadm": adm1_codes_gadm,
            }
        )
        result_rows.append(row)

    # Case 5: Data has neither ADM2 nor ADM1
    else:
        result_rows.append(get_base_row())

    return result_rows


# ----------------------------------------------
# Fields: 'start_date', 'end_date', 'event_id'
# Convert 'Admin Units' from string to Python List
# Processing the data
def preprocess_data(filepath):
    print(f"Reading data from {filepath}...")
    emdat_data = pd.read_csv(filepath)

    # ----------------------------------------------
    emdat_derived = emdat_data[
        [
            "DisNo.",
            "Country",
            "Location",
            "Latitude",
            "Longitude",
            "Start Year",
            "Start Month",
            "Start Day",
            "End Year",
            "End Month",
            "End Day",
            "Admin Units",
            "GADM Admin Units",
        ]
    ].copy()

    # ----------------------------------------------
    emdat_derived["start_date"] = pd.to_datetime(
        emdat_derived[["Start Year", "Start Month", "Start Day"]].rename(
            columns={"Start Year": "year", "Start Month": "month", "Start Day": "day"}
        )
    )

    emdat_derived["end_date"] = pd.to_datetime(
        emdat_derived[["End Year", "End Month", "End Day"]].rename(
            columns={"End Year": "year", "End Month": "month", "End Day": "day"}
        )
    )

    emdat_derived["event_id"] = emdat_derived["DisNo."].astype(str)

    # ----------------------------------------------
    print("Parsing and Restructuring Admin Units...")
    emdat_derived["admin_list_raw"] = emdat_derived["Admin Units"].apply(
        parse_admin_units_safe
    )
    emdat_derived["admin_list_raw_gadm"] = emdat_derived["GADM Admin Units"].apply(
        parse_admin_units_safe
    )
    emdat_derived["admin_list_structured"] = emdat_derived.apply(
        lambda row: reorganize_admin_data(
            row["admin_list_raw"], row["admin_list_raw_gadm"]
        ),
        axis=1,
    )
    # Explode the admin_list to have one row per admin unit
    emdat_exploded = emdat_derived.explode("admin_list_structured").reset_index(
        drop=True
    )
    # Expand the dictionaries in admin_list into separate columns (for easier processing)
    admin_details = pd.json_normalize(emdat_exploded["admin_list_structured"])
    emdat_final = pd.concat(
        [
            emdat_exploded.drop(
                columns=[
                    "admin_list_raw",
                    "admin_list_raw_gadm",
                    "admin_list_structured",
                ]
            ),
            admin_details,
        ],
        axis=1,
    )

    return emdat_final


# ----------------------------------------------
# Field: 'bbox'
# Define a function to get bounding box from GEE
def get_bbox_from_gee(row, gaul_dataset, gadm_dataset):
    country = row["Country"]

    adm2_code = row["adm2_code"] if pd.notna(row["adm2_code"]) else None
    adm2_name = row["adm2_name"] if pd.notna(row["adm2_name"]) else None
    adm1_list = (
        row["adm1_name_list"]
        if isinstance(row.get("adm1_name_list"), list)
        and len(row["adm1_name_list"]) > 0
        else None
    )

    adm2_code_gadm = (
        row["adm2_code_gadm"] if pd.notna(row.get("adm2_code_gadm")) else None
    )
    adm2_name_gadm = (
        row["adm2_name_gadm"] if pd.notna(row.get("adm2_name_gadm")) else None
    )
    adm1_list_gadm = (
        row["adm1_name_list_gadm"]
        if isinstance(row.get("adm1_name_list_gadm"), list)
        and len(row.get("adm1_name_list_gadm")) > 0
        else None
    )

    try:
        lon = float(row["Longitude"])
        lat = float(row["Latitude"])
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
            filtered = gaul_dataset.filter(ee.Filter.eq("ADM2_CODE", code_int))

            if filtered.size().getInfo() > 0:
                target_feature = filtered.first()
                match_method = "ADM2_CODE"
        except Exception:
            pass

    # --- 2. If no adm2_code, use adm2_name + adm1_name + Country ---
    if target_feature is None and adm2_name is not None and adm1_list is not None:
        filtered = gaul_dataset.filter(
            ee.Filter.and_(
                ee.Filter.eq("ADM0_NAME", country),
                ee.Filter.eq("ADM2_NAME", adm2_name),
                ee.Filter.inList("ADM1_NAME", adm1_list),
            )
        )

        if filtered.size().getInfo() > 0:
            target_feature = filtered.first()
            match_method = "ADM0&1&2_NAME"

    # --- 3. If no match above, use adm2_name + Country ---
    if target_feature is None and adm2_name is not None:
        filtered = gaul_dataset.filter(
            ee.Filter.and_(
                ee.Filter.eq("ADM0_NAME", country), ee.Filter.eq("ADM2_NAME", adm2_name)
            )
        )

        if filtered.size().getInfo() > 0:
            target_feature = filtered.first()
            match_method = "ADM0&2_NAME"

    # --- 4. If no match above, use adm2_code_gadm ---
    if target_feature is None and adm2_code_gadm is not None:
        try:
            code_int = int(adm2_code_gadm)
            matches = gadm_dataset[gadm_dataset["GID_2"] == code_int]

            if not matches.empty:
                target_feature = matches.iloc[0]
                match_method = "ADM2_CODE_GADM"
        except Exception:
            pass

    # --- 5. If no match above, use adm2_name_gadm + adm1_name_gadm + Country ---
    if (
        target_feature is None
        and adm2_name_gadm is not None
        and adm1_list_gadm is not None
    ):
        matches = gadm_dataset[
            (gadm_dataset["COUNTRY"] == country)
            & (gadm_dataset["NAME_2"] == adm2_name_gadm)
            & (gadm_dataset["NAME_1"].isin(adm1_list_gadm))
        ]

        if not matches.empty:
            target_feature = matches.iloc[0]
            match_method = "ADM0&1&2_NAME_GADM"

    # --- 6. If no match above, use adm2_name_gadm + Country ---
    if target_feature is None and adm2_name_gadm is not None:
        matches = gadm_dataset[
            (gadm_dataset["COUNTRY"] == country)
            & (gadm_dataset["NAME_2"] == adm2_name_gadm)
        ]

        if not matches.empty:
            target_feature = matches.iloc[0]
            match_method = "ADM0&2_NAME_GADM"

    # --- 7. If no match by name, and coordinates are available, use coordinates lookup ---
    if target_feature is None and has_coords:
        point = ee.Geometry.Point([lon, lat])
        filtered = gaul_dataset.filterBounds(point)

        if filtered.size().getInfo() > 0:
            target_feature = filtered.first()
            match_method = "Coordinates_Lookup"

    # --- 8. If no match found, return error message ---
    if target_feature is None:
        # print(f"event_id: {event_id} cannot be located")
        return {
            "bbox": [[0, 0], [0, 0], [0, 0], [0, 0]],
            "match_method": "cannot_be_located",
        }

    # --- Getting the bounding box ---
    try:
        if isinstance(target_feature, (pd.Series, gpd.GeoSeries)):
            # if target_feature is a GeoPandas GeoSeries (from GADM)
            geom = target_feature.geometry
            minx, miny, maxx, maxy = geom.bounds
            bbox = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]]
            return {"bbox": bbox, "match_method": match_method}
        else:
            # if target_feature is an ee.Feature (from GAUL)
            geom = target_feature.geometry()
            # bounds: [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
            bounds = geom.bounds().coordinates().get(0).getInfo()

            return {"bbox": bounds[0:4], "match_method": match_method}

    except Exception as e:
        # If an internal GEE geometry error occurs, also mark as failure
        return {
            "bbox": [[0, 0], [0, 0], [0, 0], [0, 0]],
            "match_method": f"error_{str(e)}",
        }


# ----------------------------------------------
# Main function
def main():
    # Filepath settings & Google Earth Engine project name setting
    input_filepath = "/home/chunen/nas/HAZAMA_data/public_emdat_custom_request_2026-01-28.csv"
    # output_filepath = "/home/chunen/HAZAMA/HAZAMA/outputs/data_ingestion.csv"
    my_gee_project = "oceanic-hash-467505-r2"
    # GADM GeoPackage filepath setting
    gadm_filepath = "/home/chunen/nas/HAZAMA_data/gadm_410-levels-ADM2.gpkg"

    # A-1. Initialize GEE and read GAUL dataset
    initialize_gee(my_gee_project)
    gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")

    # A-2. Load GADM dataset
    print("Loading GADM GeoPackage...")
    gadm = gpd.read_file(gadm_filepath)
    # Reproject to EPSG:4326 if the CRS is different
    if gadm.crs != "EPSG:4326":
        print("Reprojecting GeoDataFrame to EPSG:4326...")
        gadm = gadm.to_crs("EPSG:4326")

    # B. Read and process data
    try:
        df_processed = preprocess_data(input_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {input_filepath}")
        sys.exit(1)  # Exit the program with an error code

    # C. Set the range to execute (test mode or full mode)
    df_to_process = df_processed.iloc[0:100].copy()
    # df_to_process = df_processed.copy()

    print(f"Start querying GEE for {len(df_to_process)} records...")
    tqdm.pandas()

    # D. Execute query (use lambda to pass gaul into the function)
    df_to_process["bbox_result"] = df_to_process.progress_apply(
        lambda row: get_bbox_from_gee(row, gaul, gadm), axis=1
    )

    # E. Organize results
    bbox_df = pd.json_normalize(df_to_process["bbox_result"])
    final_df = pd.concat([df_to_process.reset_index(drop=True), bbox_df], axis=1)

    # Check how many events could not be located
    missing_count = len(final_df[final_df["match_method"] == "cannot_be_located"])
    print(f"There are **{missing_count}** records that could not be located")
    # Check how many events had errors
    error_count = len(final_df[final_df["match_method"].str.startswith("error_")])
    print(f"There are **{error_count}** records that had errors during processing")

    # F. Output file
    output = final_df[["event_id", "start_date", "end_date", "bbox"]]
    # Drop [[0, 0], [0, 0], [0, 0], [0, 0]] entries
    output_clear = output[
        output["bbox"].apply(lambda x: x != [[0, 0], [0, 0], [0, 0], [0, 0]])
    ]
    # output_clear.to_csv(output_filepath, index=False)
    print("The Data for ingestion was modified to output_clear variable.")


# ----------------------------------------------
# Run the main function
if __name__ == "__main__":
    main()
