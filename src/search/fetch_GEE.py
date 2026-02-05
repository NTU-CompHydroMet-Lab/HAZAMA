import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import ee
import geemap
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("GEE_Window_Search")

# -------------------------
# Init
# -------------------------
def init_gee_service_account(service_account_email: str, key_json_path: str) -> None:
    credentials = ee.ServiceAccountCredentials(service_account_email, key_json_path)
    ee.Initialize(credentials)
    log.info("✅ GEE initialized (service account).")

# -------------------------
# Helpers
# -------------------------
def bbox_to_geom(bbox: List[float]) -> ee.Geometry:
    return ee.Geometry.Rectangle(bbox)

def to_ymd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def date_tag_from_img(img: ee.Image) -> str:
    return ee.Date(img.get("system:time_start")).format("YYYYMMdd").getInfo()

# -------------------------
# Windows
# -------------------------
@dataclass
class Windows:
    pre_start: str
    pre_end: str
    post_start: str
    post_end: str

def build_windows(event_date_ymd: str) -> Windows:
    """
    pre-event:  [D-15, D-1]  -> filterDate(D-15, D)  (end exclusive)
    post-event: [D, D+3]     -> filterDate(D, D+4)   (end exclusive)
    """
    d = datetime.strptime(event_date_ymd, "%Y-%m-%d")
    pre_start = d - timedelta(days=15)
    pre_end_excl = d
    post_start = d
    post_end_excl = d + timedelta(days=4)

    return Windows(
        pre_start=to_ymd(pre_start),
        pre_end=to_ymd(pre_end_excl),
        post_start=to_ymd(post_start),
        post_end=to_ymd(post_end_excl),
    )

# -------------------------
# Coverage computation (common)
# -------------------------
def add_coverage_pct(img: ee.Image, bbox_geom: ee.Geometry) -> ee.Image:
    bbox_area = bbox_geom.area(maxError=1)
    inter = img.geometry().intersection(bbox_geom, maxError=1)
    inter_area = inter.area(maxError=1)
    pct = ee.Number(inter_area).divide(bbox_area).multiply(100)
    return img.set({"coverage_pct": pct})

def filter_by_coverage(ic: ee.ImageCollection, threshold_pct: float) -> ee.ImageCollection:
    return ic.filter(ee.Filter.gte("coverage_pct", threshold_pct))

# -------------------------
# S2 functions
# -------------------------
def s2_collection(bbox_geom: ee.Geometry, start: str, end: str,
                  cloud_mask: bool = False) -> ee.ImageCollection:
    """
    這裡先不強制雲遮罩（你主要是要 coverage + metadata + export）
    若你要雲遮罩/反射率縮放，可以再加 map(mask)。
    """
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(bbox_geom)
            .filterDate(start, end))

def s2_to_feature(img: ee.Image) -> ee.Feature:
    date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
    props = ee.Dictionary({
        "id": img.get("system:index"),
        "datetime": date_str,
        "time_start": img.get("system:time_start"),
        "coverage_pct": img.get("coverage_pct"),
        "cloud_pct": img.get("CLOUDY_PIXEL_PERCENTAGE"),
        "band_names": img.bandNames(),
    })
    return ee.Feature(None, props)

def s2_pick_best(ic: ee.ImageCollection) -> Optional[ee.Image]:
    """
    coverage 高優先、cloud 低優先、最新優先
    """
    n = ic.size()
    return ee.Image(ee.Algorithms.If(
        n.gt(0),
        ic.sort("coverage_pct", False)
          .sort("CLOUDY_PIXEL_PERCENTAGE", True)
          .sort("system:time_start", False)
          .first(),
        None
    ))

# -------------------------
# S1 functions
# -------------------------
def s1_collection(bbox_geom: ee.Geometry, start: str, end: str,
                  orbit: Optional[str] = None,
                  instrument_mode: str = "IW") -> ee.ImageCollection:
    ic = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(bbox_geom)
          .filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode", instrument_mode)))
    if orbit in ("ASCENDING", "DESCENDING"):
        ic = ic.filter(ee.Filter.eq("orbitProperties_pass", orbit))
    return ic

def s1_to_feature(img: ee.Image) -> ee.Feature:
    date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
    props = ee.Dictionary({
        "id": img.get("system:index"),
        "datetime": date_str,
        "time_start": img.get("system:time_start"),
        "coverage_pct": img.get("coverage_pct"),
        "orbit_pass": img.get("orbitProperties_pass"),
        "instrument_mode": img.get("instrumentMode"),
        "polarizations": img.get("transmitterReceiverPolarisation"),
        "band_names": img.bandNames(),
    })
    return ee.Feature(None, props)

def s1_pick_best(ic: ee.ImageCollection) -> Optional[ee.Image]:
    """
    coverage 高優先、最新優先
    """
    n = ic.size()
    return ee.Image(ee.Algorithms.If(
        n.gt(0),
        ic.sort("coverage_pct", False)
          .sort("system:time_start", False)
          .first(),
        None
    ))

# -------------------------
# Export (common + dataset-specific wrappers)
# -------------------------
def export_one_band(image: ee.Image, band: str, region: ee.Geometry,
                    out_path: str, scale: int, crs: str = "EPSG:4326") -> None:
    ensure_dir(os.path.dirname(out_path))
    geemap.ee_export_image(
        image=image.select([band]).clip(region),
        filename=out_path,
        scale=scale,
        region=region,
        crs=crs,
        file_per_band=False,
    )

def band_exists(image: ee.Image, band: str) -> bool:
    bnames = image.bandNames().getInfo()
    return band in bnames

def export_s2(image: ee.Image, region: ee.Geometry, outdir: str, event_id: str,
              mode: str = "RGB",
              bands: Optional[List[str]] = None,
              scale: int = 10) -> List[str]:
    """
    mode:
      - "RGB": export B4,B3,B2
      - "BANDS": export specified bands
    return list of file paths exported
    """
    ts = date_tag_from_img(image)
    exported = []

    if mode.upper() == "RGB":
        bands_to_export = ["B4", "B3", "B2"]
    else:
        if not bands:
            raise ValueError("S2 mode=BANDS requires bands=[...]")
        bands_to_export = bands

    for b in bands_to_export:
        if not band_exists(image, b):
            log.warning(f"[S2] missing band {b}, skip.")
            continue
        fp = os.path.join(outdir, f"{event_id}_S2_{ts}_{b}.tif")
        export_one_band(image, b, region, fp, scale=scale)
        exported.append(os.path.abspath(fp))

    return exported

def export_s1(image: ee.Image, region: ee.Geometry, outdir: str, event_id: str,
              bands: Optional[List[str]] = None,
              scale: int = 10) -> List[str]:
    """
    Default export VV/VH; if VH missing -> auto skip
    """
    ts = date_tag_from_img(image)
    exported = []
    bands_to_export = bands or ["VV", "VH"]

    for b in bands_to_export:
        if not band_exists(image, b):
            log.warning(f"[S1] missing band {b}, skip.")
            continue
        fp = os.path.join(outdir, f"{event_id}_S1_{ts}_{b}.tif")
        export_one_band(image, b, region, fp, scale=scale)
        exported.append(os.path.abspath(fp))

    return exported

# -------------------------
# Window fetch (metadata + best + coverage filter)
# -------------------------
def fetch_window(
    dataset: str,
    bbox: List[float],
    start: str,
    end: str,
    coverage_threshold: float = 95.0,
    max_items: int = 200,
    s1_orbit: Optional[str] = None,
) -> Dict:
    region = bbox_to_geom(bbox)

    if dataset == "S2":
        ic = s2_collection(region, start, end)
        ic = ic.map(lambda img: add_coverage_pct(img, region))
        ic = filter_by_coverage(ic, coverage_threshold)
        ic = ic.limit(max_items, "system:time_start", False)

        fc = ee.FeatureCollection(ic.map(s2_to_feature))
        info = fc.getInfo()

        best = s2_pick_best(ic)
        best_id = None
        try:
            best_id = ee.Image(best).get("system:index").getInfo()
        except Exception:
            best_id = None

    elif dataset == "S1":
        ic = s1_collection(region, start, end, orbit=s1_orbit)
        ic = ic.map(lambda img: add_coverage_pct(img, region))
        ic = filter_by_coverage(ic, coverage_threshold)
        ic = ic.limit(max_items, "system:time_start", False)

        fc = ee.FeatureCollection(ic.map(s1_to_feature))
        info = fc.getInfo()

        best = s1_pick_best(ic)
        best_id = None
        try:
            best_id = ee.Image(best).get("system:index").getInfo()
        except Exception:
            best_id = None
    else:
        raise ValueError("dataset must be 'S1' or 'S2'")

    feats = info.get("features", [])
    rows = []
    band_union = set()

    for f in feats:
        props = f.get("properties", {})
        bnames = props.get("band_names", [])
        if isinstance(bnames, list):
            band_union.update(bnames)
        rows.append(props)

    return {
        "dataset": dataset,
        "window_start": start,
        "window_end": end,
        "coverage_threshold": coverage_threshold,
        "count_after_coverage_filter": len(rows),
        "available_bands_union": sorted(list(band_union)),
        "metadata_rows": rows,        # ✅ every imagery metadata queried
        "best_image_id": best_id,
    }

# -------------------------
# Event runner: pre/post for S1/S2 + export best (optional)
# -------------------------
def run_event(
    event_id: str,
    event_date_ymd: str,
    bbox: List[float],
    outdir: str,
    coverage_threshold: float = 95.0,
    max_items: int = 200,
    s1_orbit: Optional[str] = None,
    export_best: bool = True,
    s2_export_mode: str = "RGB",         # "RGB" or "BANDS"
    s2_export_bands: Optional[List[str]] = None,
    s1_export_bands: Optional[List[str]] = None,  # default ["VV","VH"]
    s2_scale: int = 10,
    s1_scale: int = 10,
) -> Dict:
    ensure_dir(outdir)
    w = build_windows(event_date_ymd)
    region = bbox_to_geom(bbox)

    results = {
        "event_id": event_id,
        "event_date": event_date_ymd,
        "bbox": bbox,
        "windows": w.__dict__,
        "coverage_threshold": coverage_threshold,
        "buckets": {},
        "exports": {},
    }

    # ---- fetch metadata for all 4 buckets
    buckets = {
        "S2_pre": ("S2", w.pre_start, w.pre_end),
        "S2_post": ("S2", w.post_start, w.post_end),
        "S1_pre": ("S1", w.pre_start, w.pre_end),
        "S1_post": ("S1", w.post_start, w.post_end),
    }

    # keep collections for export step (we recreate them deterministically)
    for name, (ds, start, end) in buckets.items():
        results["buckets"][name] = fetch_window(
            dataset=ds,
            bbox=bbox,
            start=start,
            end=end,
            coverage_threshold=coverage_threshold,
            max_items=max_items,
            s1_orbit=s1_orbit,
        )

    # ---- export best images if requested
    if export_best:
        # export each bucket's best image (if exists)
        for name, (ds, start, end) in buckets.items():
            best_id = results["buckets"][name]["best_image_id"]
            if not best_id:
                results["exports"][name] = {"status": "NO_IMAGE", "files": []}
                continue

            # rebuild the same filtered collection then pick first by sorting rule
            if ds == "S2":
                ic = s2_collection(region, start, end)
                ic = ic.map(lambda img: add_coverage_pct(img, region))
                ic = filter_by_coverage(ic, coverage_threshold)
                ic = ic.limit(max_items, "system:time_start", False)
                best_img = s2_pick_best(ic)
                best_img = ee.Image(best_img)

                bucket_dir = os.path.join(outdir, event_id, name)
                ensure_dir(bucket_dir)
                files = export_s2(
                    best_img, region, bucket_dir, event_id,
                    mode=s2_export_mode,
                    bands=s2_export_bands,
                    scale=s2_scale,
                )
                results["exports"][name] = {"status": "EXPORTED" if files else "NO_BAND_EXPORTED", "files": files}

            else:
                ic = s1_collection(region, start, end, orbit=s1_orbit)
                ic = ic.map(lambda img: add_coverage_pct(img, region))
                ic = filter_by_coverage(ic, coverage_threshold)
                ic = ic.limit(max_items, "system:time_start", False)
                best_img = s1_pick_best(ic)
                best_img = ee.Image(best_img)

                bucket_dir = os.path.join(outdir, event_id, name)
                ensure_dir(bucket_dir)
                files = export_s1(
                    best_img, region, bucket_dir, event_id,
                    bands=s1_export_bands,  # None -> default VV/VH
                    scale=s1_scale,
                )
                results["exports"][name] = {"status": "EXPORTED" if files else "NO_BAND_EXPORTED", "files": files}

    return results

# -------------------------
# Batch + summary CSV
# -------------------------
def summarize_to_df(event_outputs: List[Dict]) -> pd.DataFrame:
    rows = []
    for ev in event_outputs:
        for bucket_name, bucket in ev["buckets"].items():
            exp = ev["exports"].get(bucket_name, {})
            rows.append({
                "event_id": ev["event_id"],
                "event_date": ev["event_date"],
                "bucket": bucket_name,
                "coverage_threshold": ev["coverage_threshold"],
                "count_after_coverage_filter": bucket["count_after_coverage_filter"],
                "best_image_id": bucket["best_image_id"],
                "available_bands_union": ", ".join(bucket["available_bands_union"]),
                "export_status": exp.get("status", ""),
                "export_files": " | ".join(exp.get("files", [])),
                "window_start": bucket["window_start"],
                "window_end": bucket["window_end"],
            })
    return pd.DataFrame(rows)

# -------------------------
# Example
# -------------------------
if __name__ == "__main__":
    init_gee_service_account(
        "blue051407@gmail.com",
        "global-flood-mapping-a7463d9cfde6.json",
    )

    event_list = [
        {
            "id": "event_001",
            "date": "2018-07-15",  # ✅ 你定義的 Date（例如事件開始日/峰值日）
            "bbox": [2.66843104, 5.73895788, 12.48964119, 13.3760643],
        }
    ]

    outputs = []
    for ev in event_list:
        outputs.append(
            run_event(
                event_id=ev["id"],
                event_date_ymd=ev["date"],
                bbox=ev["bbox"],
                outdir="data/output_images",
                coverage_threshold=95.0,
                max_items=200,
                s1_orbit=None,            # or "ASCENDING"/"DESCENDING"
                export_best=True,

                # ✅ S2 export
                s2_export_mode="RGB",      # "RGB" or "BANDS"
                s2_export_bands=None,      # used if mode="BANDS" e.g. ["B8","B4","B3"]

                # ✅ S1 export
                s1_export_bands=["VV", "VH"],  # VH missing auto skip

                s2_scale=10,
                s1_scale=10,
            )
        )

    df = summarize_to_df(outputs)
    print(df)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/results.csv", index=False, encoding="utf-8-sig")
    log.info("✅ Saved summary CSV: data/results.csv")
