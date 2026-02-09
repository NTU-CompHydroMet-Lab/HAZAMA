from unittest.mock import MagicMock, patch

import pytest

from src.search.fetch_CDSE import DEFAULT_BBOX
from src.search.fetch_CDSE import main as cdse

# --- 測試資料準備 ---


@pytest.fixture
def mock_events():
    return [
        {
            "id": "EVENT_S2",
            "start_date": "2024-12-10",
            "end_date": "2024-12-11",
            "pre_event_days": 5,
            "post_event_days": 5,
            "bbox": None,  # 測試補全邏輯
        }
    ]


# --- 核心測試 1：參數處理與日期計算 ---
@patch("src.search.fetch_CDSE.get_stac_client")
@patch("src.search.fetch_CDSE.save_as_cog")
@patch("os.makedirs")
def test_cdse_logic_and_defaults(mock_mkdir, mock_save, mock_stac, mock_events):
    # 模擬 STAC 搜尋回傳空結果，我們只測進入 process_event_for_cdse 之前的參數
    mock_stac.return_value.search.return_value.item_collection.return_value = []
    mock_save.return_value = "/mock/path/file.tif"
    # 執行任務
    cdse(mock_events, collection="sentinel-2-l2a", bands=["B04"], base_dir="/mock/data")

    # 驗證 1：日期計算是否正確 (2024-01-10 減 5 天 = 2024-01-05)
    # 取得搜尋時傳入的參數
    args, kwargs = mock_stac.return_value.search.call_args
    assert kwargs["datetime"] == "2024-12-05/2024-12-16"

    # 驗證 2：BBox 是否補全為預設值
    assert kwargs["bbox"] == DEFAULT_BBOX


# --- 核心測試 2：座標轉換 (CRS) 驗證 ---
def test_coordinate_transformation_logic():
    from rasterio.warp import transform_bounds

    wgs84_bbox = [121.5, 25.0, 121.6, 25.1]
    # 模擬轉換到台灣常用的 UTM 51N 座標系
    target_crs = "EPSG:32651"

    left, bottom, right, top = transform_bounds("EPSG:4326", target_crs, *wgs84_bbox)

    # 檢查轉換後的數值是否符合投影座標特徵 (通常是大於 100,000 的數值)
    assert left > 100000
    assert bottom > 1000000




# --- 核心測試 4：CSV 產出格式驗證 ---
@pytest.mark.skip(reason="data資料夾尚未建立，暫時跳過")
@patch("src.search.fetch_CDSE.process_event_for_cdse")
def test_csv_output_generation(mock_process, tmp_path):
    # 模擬處理後的結果
    mock_process.return_value = {
        "event_id": "EVENT_S2",
        "metadata": "S2A_xxx",
        "cloud_coverage": [10.5],
        "path": str(tmp_path),
        "status": "SUCCESS",
    }

    # 執行 cdse (這裡我們可以用真實的 pandas 儲存，但存在暫時目錄)
    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        cdse(
            [
                {
                    "id": "E1",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                    "pre_event_days": 1,
                    "post_event_days": 1,
                }
            ]
        )

        # 驗證是否有呼叫儲存
        assert mock_to_csv.called
