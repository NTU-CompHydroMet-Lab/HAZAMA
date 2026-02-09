import json
import logging
import math
import os
from datetime import datetime, timedelta

import boto3
import pandas as pd
import rasterio
from dotenv import load_dotenv
from pystac_client import Client
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.session import AWSSession
from rasterio.shutil import copy as rio_copy
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, transform_bounds
from rasterio.windows import from_bounds

load_dotenv()

access_key = os.getenv("CDSE_S3_ACCESS_KEY")
secret_key = os.getenv("CDSE_S3_SECRET_KEY")
session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
logger = logging.getLogger("CDSE_Fetcher")
logging.basicConfig(level=logging.INFO)

DEFAULT_BBOX = [121.56, 25.03, 121.57, 25.04]


def get_stac_client():
    return Client.open("https://catalogue.dataspace.copernicus.eu/stac")


def get_utm_crs(lon, lat, item=None):
    if item and "proj:epsg" in item.properties:
        return CRS.from_epsg(item.properties["proj:epsg"])

    zone = int(math.floor((lon + 180) / 6) + 1)
    epsg_code = (32600 + zone) if lat >= 0 else (32700 + zone)
    return CRS.from_epsg(epsg_code)


def download_raw_vsis3(item, event_id, raw_dir, band, bbox_wgs84, logger):
    asset = item.assets.get(band)
    if not asset:
        return None

    vsis3_url = asset.href.replace("s3://eodata/", "/vsis3/eodata/")
    raw_name = f"{event_id}_{item.datetime.strftime('%Y%m%d')}_{band}_{item.id}_RAW.tif"
    raw_path = os.path.join(raw_dir, raw_name)

    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
        return os.path.abspath(raw_path)

    try:
        tmp_path = raw_path + ".part"
        logger.info(f"Downloading {band} from CDSE S3...")

        with rasterio.Env(
            AWSSession(session),
            AWS_S3_ENDPOINT="eodata.dataspace.copernicus.eu",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_USE_HEAD="NO",
        ):
            rio_copy(vsis3_url, tmp_path, driver="GTiff")

            with rasterio.open(tmp_path, "r+") as f:
                if f.crs is None:
                    c_lon = (bbox_wgs84[0] + bbox_wgs84[2]) / 2
                    c_lat = (bbox_wgs84[1] + bbox_wgs84[3]) / 2
                    inferred_crs = get_utm_crs(c_lon, c_lat, item)
                    f.crs = inferred_crs
                    logger.info(f"Fixed missing CRS: set to {inferred_crs}")

        os.replace(tmp_path, raw_path)
        return os.path.abspath(raw_path)
    except Exception as e:
        logger.error(f"Download error: {e}")
        return None


def cut_bbox_from_raw(raw_path, bbox_wgs84, out_path, logger):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        t_left, t_bottom, t_right, t_top = bbox_wgs84

        with rasterio.open(raw_path) as src:
            source_crs = src.crs

            img_left, img_bottom, img_right, img_top = transform_bounds(
                source_crs, "EPSG:4326", *src.bounds
            )

            inter_left, inter_bottom = max(img_left, t_left), max(img_bottom, t_bottom)
            inter_right, inter_top = min(img_right, t_right), min(img_top, t_top)

            if inter_left >= inter_right or inter_bottom >= inter_top:
                logger.warning(f"No overlap for {os.path.basename(raw_path)}")
                return None

            dst_crs = "EPSG:4326"
            transform, width, height = calculate_default_transform(
                source_crs, dst_crs, src.width, src.height, *src.bounds
            )

            vrt_params = {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "resampling": Resampling.nearest,
            }

            with WarpedVRT(src, **vrt_params) as vrt:
                window = from_bounds(
                    inter_left,
                    inter_bottom,
                    inter_right,
                    inter_top,
                    transform=vrt.transform,
                ).round()

                if window.width < 1 or window.height < 1:
                    return None

                data = vrt.read(window=window)
                profile = vrt.profile.copy()
                profile.update(
                    {
                        "driver": "GTiff",
                        "height": window.height,
                        "width": window.width,
                        "transform": vrt.window_transform(window),
                        "crs": dst_crs,
                        "tiled": True,
                        "compress": "deflate",
                    }
                )

                with rasterio.open(out_path + ".tmp", "w", **profile) as dst:
                    dst.write(data)

                os.replace(out_path + ".tmp", out_path)
                return os.path.abspath(out_path)
    except Exception as e:
        logger.error(f"Cut error: {e}")
        return None


def save_as_cog(item, bbox_wgs84, event_id, output_dir, band, raw_dir, logger):
    raw_path = download_raw_vsis3(item, event_id, raw_dir, band, bbox_wgs84, logger)
    if not raw_path:
        return None
    cropped_dir = os.path.join(output_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)
    out_name = f"{event_id}_{item.datetime.strftime('%Y%m%d')}_{band}_{item.id}.tif"
    out_path = os.path.join(cropped_dir, out_name)
    return cut_bbox_from_raw(raw_path, bbox_wgs84, out_path, logger)


def process_event_for_cdse(
    event_id, bbox, date_range, collection, bands, base_output_dir, logger
):
    event_folder = os.path.join(base_output_dir, event_id)
    actual_bbox = bbox if bbox else DEFAULT_BBOX
    raw_dir = os.path.join(event_folder, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    rows = []

    try:
        catalog = get_stac_client()
        search = catalog.search(
            collections=[collection], bbox=actual_bbox, datetime=date_range
        )
        items = list(search.items())

        if not items:
            logger.warning(f"[NO_DATA_FOUND] {event_id}")
            return []

        for item in items:
            metadata_dir = os.path.join(event_folder, "metadata")
            os.makedirs(metadata_dir, exist_ok=True)
            metadata_path = os.path.abspath(
                os.path.join(metadata_dir, f"{item.id}.json")
            )
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(item.to_dict(), f, indent=4, ensure_ascii=False)

            for band in bands:
                raw_path = download_raw_vsis3(
                    item, event_id, raw_dir, band, actual_bbox, logger
                )

                if raw_path:
                    time_str = item.datetime.strftime("%Y%m%d")
                    out_name = f"{event_id}_{time_str}_{band}_{item.id}_cropped.tif"
                    cropped_dir = os.path.join(event_folder, "cropped")
                    os.makedirs(cropped_dir, exist_ok=True)
                    out_path = os.path.join(cropped_dir, out_name)
                    final_path = cut_bbox_from_raw(
                        raw_path, actual_bbox, out_path, logger
                    )

                    if final_path:
                        rows.append(
                            {
                                "event_id": event_id,
                                "item_id": item.id,
                                "date": item.datetime.strftime("%Y-%m-%d"),
                                "band": band,
                                "cloud_cover": item.properties.get("eo:cloud_cover"),
                                "metadata_path": metadata_path,
                                "raw_path": os.path.abspath(raw_path),
                                "cropped_path": os.path.abspath(final_path),
                                "status": "SUCCESS",
                            }
                        )
                    else:
                        logger.error(f"Fail to crop: {item.id} {band}")
                else:
                    logger.error(f"Fail to download raw: {item.id} {band}")

        return rows

    except Exception as e:
        logger.error(f"Error processing event {event_id}: {e}")
        return []


def main(
    event_list,
    collection=None,
    bands=None,
    base_dir=None,
):
    all_results = []
    with rasterio.Env(
        AWSSession(session),
        AWS_S3_ENDPOINT="eodata.dataspace.copernicus.eu",
        GDAL_S3_ENDPOINT_DIRECT="eodata.dataspace.copernicus.eu",
        AWS_VIRTUAL_HOSTING="FALSE",
        AWS_HTTPS="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    ):
        for event in event_list:
            # 時間計算邏輯 預計在ingestion.py先處理好
            start_dt = datetime.strptime(event["start_date"], "%Y-%m-%d")
            end_dt = datetime.strptime(event["end_date"], "%Y-%m-%d")
            full_start = start_dt - timedelta(days=int(event["pre_event_days"]))
            full_end = end_dt + timedelta(days=int(event["post_event_days"]))
            date_range = (
                f"{full_start.strftime('%Y-%m-%d')}/{full_end.strftime('%Y-%m-%d')}"
            )

            event_results = process_event_for_cdse(
                event["id"],
                event.get("bbox"),
                date_range,
                collection,
                bands,
                base_dir,
                logger,
            )

            if event_results:
                for row in event_results:
                    row.update(
                        {
                            "pre_event_days": event["pre_event_days"],
                            "post_event_days": event["post_event_days"],
                        }
                    )
                all_results.extend(event_results)

    output_path = "data/results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)
    logger.info("CSV had been updated！")


if __name__ == "__main__":
    # Example usage
    test_events = [
        {
            "id": "ISTANBUL_TEST2",
            "start_date": "2024-12-05",  # 這是你之前測試過有圖的日期
            "end_date": "2024-12-10",
            "pre_event_days": 1,
            "post_event_days": 1,
            "bbox": [28.97, 41.0, 28.99, 41.02],  # 伊斯坦堡座標
        }
    ]

    config = {
        "collection": "sentinel-2-l2a",
        "bands": ["TCI_10m"],
        "base_dir": "data/istanbul_test",
    }

    main(test_events, **config)
