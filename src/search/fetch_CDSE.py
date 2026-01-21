import logging
import os
from datetime import datetime, timedelta

import boto3

# import pandas as pd
import rasterio
from dotenv import load_dotenv
from pystac_client import Client
from rasterio.session import AWSSession
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CDSE_Fetcher")

load_dotenv()

DEFAULT_BBOX = [121.56, 25.03, 121.57, 25.04]  # 台北 101 座標


def get_stac_client():
    return Client.open("https://catalogue.dataspace.copernicus.eu/stac")


def save_as_cog(item, bbox_wgs84, event_id, output_dir, band):
    access_key = os.getenv("CDSE_S3_ACCESS_KEY")
    secret_key = os.getenv("CDSE_S3_SECRET_KEY")
    asset = item.assets.get(band)
    if not asset:
        available_assets = list(item.assets.keys())
        logger.warning(f"Image {item.id} can't find band {band}")
        logger.info(f"The available bands of this product: {available_assets}")
        return None

    s3_url = asset.href.replace(
        "https://eodata.dataspace.copernicus.eu/", "/vsis3/eodata/"
    )
    date_str = item.datetime.strftime("%Y%m%d")
    unique_filename = f"{event_id}_{date_str}_{band}.tif"
    local_path = os.path.join(output_dir, unique_filename)
    try:
        session = boto3.Session(
            aws_access_key_id=access_key, aws_secret_access_key=secret_key
        )

        with rasterio.Env(
            AWSSession(session),
            AWS_S3_ENDPOINT="eodata.dataspace.copernicus.eu",
            GDAL_S3_ENDPOINT_DIRECT="eodata.dataspace.copernicus.eu",
            AWS_VIRTUAL_HOSTING="FALSE",
            AWS_HTTPS="YES",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        ):
            with rasterio.open(s3_url) as src:
                with WarpedVRT(src, dst_crs="EPSG:4326") as vrt:
                    target_crs = vrt.crs
                    try:
                        # 取得影像邊界 (WGS84)
                        t_left, t_bottom, t_right, t_top = transform_bounds(
                            "EPSG:4326", target_crs, *bbox_wgs84
                        )
                        img_left, img_bottom, img_right, img_top = vrt.bounds

                        # 計算交集範圍
                        inter_left = max(img_left, t_left)
                        inter_bottom = max(img_bottom, t_bottom)
                        inter_right = min(img_right, t_right)
                        inter_top = min(img_top, t_top)
                        logger.info(f"Bound of image: {vrt.bounds}")
                        logger.info(f"Bound of target: {bbox_wgs84}")
                        # 檢查是否有實質交集
                        if inter_left >= inter_right or inter_bottom >= inter_top:
                            logger.warning(
                                f"Image {item.id} does not overlap with target bbox."
                            )
                            return None

                        # 使用交集範圍計算window
                        window = from_bounds(
                            inter_left,
                            inter_bottom,
                            inter_right,
                            inter_top,
                            transform=vrt.transform,
                        )

                        # 再次安全檢查
                        if window.width < 1 or window.height < 1:
                            logger.error(
                                f"Invalid window (w={window.width},h={window.height})"
                            )
                            return None

                        logger.info(
                            f"Download {int(window.width)}x{int(window.height)}"
                        )
                        data = vrt.read(window=window)

                        # 更新 Profile
                        profile = vrt.profile.copy()
                        profile.update(
                            {
                                "driver": "GTiff",
                                "height": data.shape[1],
                                "width": data.shape[2],
                                "transform": rasterio.windows.transform(
                                    window, vrt.transform
                                ),
                                "tiled": True,
                                "compress": "deflate",
                                "crs": "EPSG:4326",
                            }
                        )

                        with rasterio.open(local_path, "w", **profile) as dst:
                            dst.write(data)

                    except Exception as e:
                        logger.error(f"{band} Fail: {e}")
                        return None

        return os.path.abspath(local_path)
    except Exception as e:
        logger.error(f"Fail to download image: {e}")
        return None


def process_event_for_cdse(
    event_id, bbox, date_range, collection, bands, base_output_dir
):
    event_folder = os.path.join(base_output_dir, event_id)
    if not os.path.exists(event_folder):
        os.makedirs(event_folder)

    actual_bbox = bbox if bbox else DEFAULT_BBOX

    results = {
        "event_id": event_id,
        "metadata": [],
        "cloud_coverage": [],
        "path": os.path.abspath(event_folder),
        "status": "PENDING",
    }

    try:
        catalog = get_stac_client()
        search = catalog.search(
            collections=[collection], bbox=actual_bbox, datetime=date_range
        )

        items = list(search.items())

        if not items:
            logger.warning(
                f"[NO_DATA_FOUND] {event_id} in {date_range} does not have data."
            )
            results["status"] = "NO_IMAGE"
            return results

        success_count = 0
        for item in items:
            download_success = False
            for band in bands:
                path = save_as_cog(item, actual_bbox, event_id, event_folder, band)
                if path:
                    download_success = True

            if download_success:
                success_count += 1
                results["metadata"].append(item.id)
                cc = item.properties.get("eo:cloud_cover")
                if cc is not None:
                    results["cloud_coverage"].append(cc)

        if success_count > 0:
            results["status"] = "SUCCESS"
            results["metadata"] = ", ".join(results["metadata"])
            results["cloud_coverage"] = (
                round(
                    sum(results["cloud_coverage"]) / len(results["cloud_coverage"]), 2
                )
                if results["cloud_coverage"]
                else 0
            )
        else:
            results["status"] = "NO_IMAGE"

    except Exception as e:
        logger.error(f"Processing {event_id} error: {e}")
        results["status"] = "API_ERROR"

    return results


def main(
    event_list,
    collection=None,
    bands=None,
    base_dir=None,
):
    all_results = []
    for event in event_list:
        # 時間計算邏輯 預計在ingestion.py先處理好
        start_dt = datetime.strptime(event["start_date"], "%Y-%m-%d")
        end_dt = datetime.strptime(event["end_date"], "%Y-%m-%d")
        full_start = start_dt - timedelta(days=int(event["pre_event_days"]))
        full_end = end_dt + timedelta(days=int(event["post_event_days"]))
        date_range = (
            f"{full_start.strftime('%Y-%m-%d')}/{full_end.strftime('%Y-%m-%d')}"
        )

        res = process_event_for_cdse(
            event["id"], event.get("bbox"), date_range, collection, bands, base_dir
        )

        # 合併輸出
        all_results.append(
            {
                **res,
                "pre-event days": event["pre_event_days"],
                "post-event days": event["post_event_days"],
            }
        )
    output_path = "data/results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # df = pd.DataFrame(all_results)
    # df.to_csv(output_path, index=False)
    logger.info("CSV had been updated！")


if __name__ == "__main__":
    # Example usage
    test_events = [
        {
            "id": "S2_TEST",
            "start_date": "2024-12-05",
            "end_date": "2024-12-10",
            "pre_event_days": 5,
            "post_event_days": 5,
            "bbox": [121.56, 25.03, 121.57, 25.04],
        }
    ]

    config = {
        "collection": "sentinel-2-l2a",
        "bands": ["B04_10m", "TCI_10m"],
        "base_dir": "data/sentinel_test",
    }

    main(test_events, **config)
