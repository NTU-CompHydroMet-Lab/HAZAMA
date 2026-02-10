
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

# Add src to implementation path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ingestion import get_bbox_from_gee, parse_admin_units_safe, reorganize_admin_data


class TestIngestion(unittest.TestCase):

    def test_parse_admin_units_safe(self):
        # Case 1: Valid JSON string
        valid_json = '[{"adm1_name": "Region A", "adm2_name": "City B"}]'
        expected = [{"adm1_name": "Region A", "adm2_name": "City B"}]
        self.assertEqual(parse_admin_units_safe(valid_json), expected)

        # Case 2: Empty string
        self.assertEqual(parse_admin_units_safe(""), [])

        # Case 3: None/NaN
        self.assertEqual(parse_admin_units_safe(None), [])
        self.assertEqual(parse_admin_units_safe(pd.NA), [])
        self.assertEqual(parse_admin_units_safe(np.nan), [])
        self.assertEqual(parse_admin_units_safe(float('nan')), [])

        # Case 4: Invalid JSON string
        invalid_json = '{"adm1_name": "Region A"' # Missing closing brace
        self.assertEqual(parse_admin_units_safe(invalid_json), [])

    def test_reorganize_admin_data(self):
        # Case 1: Valid GAUL data
        gaul_list = [
            {
                "adm1_name": "Region A",
                "adm1_code": 101,
                "adm2_name": "City B",
                "adm2_code": 202,
            },
            {
                "adm1_name": "Region A",
                "adm1_code": 101,
                "adm2_name": "City C",
                "adm2_code": 203,
            },
        ]
        result = reorganize_admin_data(gaul_list, None)
        # Should create a separate row for each ADM2 entry
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['adm2_name'], "City B")
        self.assertEqual(result[1]['adm2_name'], "City C")
        self.assertEqual(result[0]['adm1_name_list'], ["Region A"])

        # Case 2: Valid GADM data (no GAUL)
        gadm_list = [
            {"name_1": "Province X", "gid_1": "ID_X", "name_2": "District Y", "gid_2": "ID_Y"}  # noqa: E501
        ]
        result = reorganize_admin_data(None, gadm_list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['adm2_name_gadm'], "District Y")
        self.assertEqual(result[0]['adm1_name_list_gadm'], ["Province X"])

        # Case 3: Both GAUL and GADM
        # Current logic: If GAUL ADM2 exists, it takes precedence and ignores GADM ADM2 entries for row creation?  # noqa: E501
        # Code:
        # if adm2_entries: ... for entry in adm2_entries: ...
        # elif adm2_entries_gadm: ...
        # So yes, GAUL takes precedence.
        result = reorganize_admin_data(gaul_list, gadm_list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['adm2_name'], "City B")
        
        # Case 4: Metadata only (ADM1) for GAUL
        gaul_adm1_only = [{"adm1_name": "Region A", "adm1_code": 101}]
        result = reorganize_admin_data(gaul_adm1_only, None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['adm1_name_list'], ["Region A"])
        self.assertIsNone(result[0]['adm2_name'])

        # Case 5: Empty input
        result = reorganize_admin_data([], [])
        # Should return one empty base row
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]['adm1_name_list'])

    @patch('ingestion.ee')
    def test_get_bbox_from_gee(self, mock_ee):
        # Mock row data
        row = {
            "Country": "TestCountry",
            "adm2_name": "TestCity",
            "adm2_code": 123,
            "adm1_name_list": ["TestRegion"],
            "adm2_name_gadm": "TestCityGADM",
            "adm2_code_gadm": "ID_123",
            "adm1_name_list_gadm": ["TestRegionGADM"],
            "Longitude": 100.0,
            "Latitude": 10.0
        }

        # Setup Mock for GEE Feature
        mock_feature = MagicMock()
        # Ensure geometry().bounds().coordinates().get(0).getInfo() returns a list of coordinates
        # [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
        mock_bounds_list = [[100.0, 10.0], [101.0, 10.0], [101.0, 11.0], [100.0, 11.0], [100.0, 10.0]]
        
        mock_geom = MagicMock()
        mock_geom.bounds.return_value.coordinates.return_value.get.return_value.getInfo.return_value = mock_bounds_list
        mock_feature.geometry.return_value = mock_geom

        # Setup Mock for GAUL Collection
        mock_gaul = MagicMock()
        
        # Mock GADM DataFrame
        # For simple testing, we can use an empty GADM or one that doesn't match to force GEE lookup
        mock_gadm = gpd.GeoDataFrame(
            {"GID_2": [], "NAME_2": [], "COUNTRY": [], "NAME_1": [], "geometry": []},
            crs="EPSG:4326"
        )

        
        # --- Test 1: Match by ADM2_CODE (GAUL) ---
        # Configure mock to find a match when filtering by ADM2_CODE
        # Code logic:
        # filtered = gaul_dataset.filter(ee.Filter.eq("ADM2_CODE", code_int))
        # if filtered.size().getInfo() > 0: ...
        
        # We need to simulate the filter chain.
        # Since we use `size().getInfo()`, we mock that.
        
        # Reset mocks
        mock_gaul.reset_mock()
        mock_feature.reset_mock()
        mock_geom.reset_mock()
        
        # Setup specific return for size() > 0
        mock_filtered = MagicMock()
        mock_filtered.size.return_value.getInfo.return_value = 1
        mock_filtered.first.return_value = mock_feature
        
        mock_gaul.filter.return_value = mock_filtered

        result = get_bbox_from_gee(row, mock_gaul, mock_gadm)
        
        self.assertEqual(result['match_method'], "ADM2_CODE")
        # Check coordinates (first 4 points)
        self.assertEqual(result['bbox'], mock_bounds_list[0:4])


        # --- Test 2: Match by Coordinate Fallback ---
        # Force name/code lookups to fail
        # This requires `filter(...).size().getInfo()` to return 0 for the first few calls
        # and `filterBounds(...).size().getInfo()` to return 1.
        
        # We can use side_effect for the different filter calls if they happen on the same object.
        # However, `gaul_dataset.filter(...)` returns a NEW object (mock_filtered). 
        # So we mock the `gaul_dataset.filter` to return a "empty" collection mock first.
        
        mock_empty_collection = MagicMock()
        mock_empty_collection.size.return_value.getInfo.return_value = 0
        
        mock_found_collection = MagicMock()
        mock_found_collection.size.return_value.getInfo.return_value = 1
        mock_found_collection.first.return_value = mock_feature
        
        # Logic flow:
        # 1. ADM2_CODE -> filter -> empty
        # 2. ADM0&1&2 -> filter -> empty
        # 3. ADM0&2 -> filter -> empty
        # ... GADM checks (local DF usage) ... we ensure they fail by passing empty GADM
        # 4. Coordinate -> filterBounds -> found
        
        mock_gaul.filter.return_value = mock_empty_collection
        mock_gaul.filterBounds.return_value = mock_found_collection
        
        result_coords = get_bbox_from_gee(row, mock_gaul, mock_gadm)
        self.assertEqual(result_coords['match_method'], "Coordinates_Lookup")


        # --- Test 3: Match GADM (local GeoDataFrame) ---
        # We need to populate the GADM dataframe with a match
        # Let's match by ADM2_CODE_GADM (which is cast to int in code, so must be numeric string)
        # row["adm2_code_gadm"] is "ID_123". int("ID_123") raises ValueError. 
        # So it skips ADM2_CODE_GADM check in catch block.
        
        # Let's try matching by NAME (ADM0&1&2_NAME_GADM)
        # matches = gadm_dataset[(COUNTRY==...) & (NAME_2==...) & (NAME_1.isin(...))]
        
        gadm_match = gpd.GeoDataFrame({
            "GID_2": ["ShouldNotMatch"],
            "NAME_2": ["TestCityGADM"],
            "COUNTRY": ["TestCountry"],
            "NAME_1": ["TestRegionGADM"],
            "geometry": [Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])]
        }, crs="EPSG:4326")
        
        # Force GAUL to fail everything
        mock_gaul.filter.return_value = mock_empty_collection
        mock_gaul.filterBounds.return_value = mock_empty_collection
        
        result_gadm = get_bbox_from_gee(row, mock_gaul, gadm_match)
        self.assertEqual(result_gadm['match_method'], "ADM0&1&2_NAME_GADM")
        self.assertEqual(result_gadm['bbox'], [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])


if __name__ == '__main__':
    unittest.main()
