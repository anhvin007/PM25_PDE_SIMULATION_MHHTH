# File: src/data_processor.py
import pandas as pd
import numpy as np
import os
from src import config

def load_and_clean_data(weather_path, pm25_path):
    """
    Đọc file CSV từ Open-Meteo, loại bỏ metadata dư thừa và đồng bộ tên cột.
    """
    # 1. Đọc dữ liệu Thời tiết (Bỏ 3 dòng metadata đầu tiên)
    df_weather = pd.read_csv(weather_path, skiprows=3)
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather.set_index('time', inplace=True)
    
    # Đổi tên cột cho chuẩn mực, dễ gọi trong các hàm vật lý
    df_weather.columns = [
        'temperature', 'wind_speed', 'wind_direction',
        'pressure', 'cloud_cover', 'blh', 'precipitation'
    ]

    # 2. Đọc dữ liệu PM2.5 (Bỏ 3 dòng metadata)
    df_pm25 = pd.read_csv(pm25_path, skiprows=3)
    df_pm25['time'] = pd.to_datetime(df_pm25['time'])
    df_pm25.set_index('time', inplace=True)
    
    # Đổi tên cột để khắc phục lỗi font chữ tiếng Nhật/Trung do Open-Meteo xuất ra
    df_pm25.columns = ['pm25']

    return df_weather, df_pm25

def process_and_split():
    """
    Hợp nhất dữ liệu, xử lý khuyết thiếu và cắt tập Train/Test theo thời gian.
    """
    print("🚀 Đang khởi động tiến trình ETL Dữ liệu...")
    
    # Bước 1: Load data
    df_weather, df_pm25 = load_and_clean_data(config.RAW_METEO_PATH, config.RAW_PM25_PATH)
    
    # Bước 2: Hợp nhất (Inner Join) dựa trên cột thời gian (Index)
    print("🔗 Đang hợp nhất dữ liệu Khí tượng và PM2.5...")
    df_merged = df_weather.join(df_pm25, how='inner')
    
    # Bước 3: Nội suy dữ liệu khuyết thiếu (Interpolation)
    # Giả sử trạm đo bị mất điện vài giờ, ta dùng nội suy tuyến tính nối các điểm lại.
    print("🩹 Đang vá các khoảng trống dữ liệu (Missing Values)...")
    df_merged.interpolate(method='linear', limit=3, inplace=True)
    df_merged.dropna(inplace=True) # Xóa các dòng NaN ở ngoài rìa không nội suy được
    
    # Bước 4: Cắt dữ liệu Chronological (Giữ nguyên trật tự thời gian)
    print("✂️ Đang chia tập Train (80%) và Test (20%)...")
    split_idx = int(len(df_merged) * config.TRAIN_RATIO)
    
    train_df = df_merged.iloc[:split_idx]
    test_df  = df_merged.iloc[split_idx:]
    
    # In báo cáo thông kê ra màn hình
    print("-" * 50)
    print(f"📊 BÁO CÁO PHÂN MẢNH DỮ LIỆU:")
    print(f"Tổng số giờ mô phỏng : {len(df_merged):,} giờ")
    print(f"Tập TRAIN (Hiệu chỉnh): {len(train_df):,} giờ | Từ {train_df.index[0].date()} đến {train_df.index[-1].date()}")
    print(f"Tập TEST  (Xác thực)  : {len(test_df):,} giờ | Từ {test_df.index[0].date()} đến {test_df.index[-1].date()}")
    print("-" * 50)
    
    # Bước 5: Lưu kết quả ra file CSV chuẩn để các module sau dùng
    train_df.to_csv(config.TRAIN_DATA_PATH)
    test_df.to_csv(config.TEST_DATA_PATH)
    print(f"✅ Đã lưu file thành công tại: {os.path.dirname(config.TRAIN_DATA_PATH)}")
    
    return train_df, test_df

if __name__ == "__main__":
    # Chạy thử file này độc lập để kiểm tra ETL Pipeline
    process_and_split()