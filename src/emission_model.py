# File: src/emission_model.py
import numpy as np
import osmnx as ox
import pandas as pd
import os
from src import config

class EmissionModel:
    def __init__(self, center_lat=10.800003, center_lon=106.600006):
        """
        Khởi tạo Mô hình Phát thải. Tọa độ mặc định lấy từ trạm quan trắc Tân Phú/Bình Tân.
        """
        self.lat = center_lat
        self.lon = center_lon
        self.W_spatial = None
        
        # Gọi hàm tạo ma trận không gian ngay khi khởi tạo class
        self._build_spatial_weight_matrix()

    def _build_spatial_weight_matrix(self):
        """
        [Ma trận W_spatial] Tải bản đồ OpenStreetMap và Rasterize thành lưới 100x100.
        """
        print(f"🌍 Đang tải bản đồ giao thông bán kính {config.DOMAIN_LENGTH/2}m từ OpenStreetMap...")
        
        # Kiểm tra xem đã có file ma trận lưu sẵn chưa (để không phải tải lại từ mạng)
        cache_path = os.path.join(config.BASE_DIR, 'data', 'geo', 'w_spatial_100x100.npy')
        if os.path.exists(cache_path):
            self.W_spatial = np.load(cache_path)
            print("✅ Đã load Ma trận W_spatial từ bộ nhớ đệm (Cache).")
            return

        # 1. Tải đồ thị giao thông (Mạng lưới đường lái xe ô tô/xe máy)
        dist_m = config.DOMAIN_LENGTH / 2.0  # Bán kính 1000m
        G = ox.graph_from_point((self.lat, self.lon), dist=dist_m, network_type='drive')
        
        # 2. Khởi tạo ma trận nền (Background noise = 0.1)
        self.W_spatial = np.full((config.N_Y, config.N_X), 0.1, dtype=np.float32)
        
        # 3. Tính toán ranh giới bounding box để ánh xạ tọa độ (Lat/Lon sang Index)
        nodes = ox.graph_to_gdfs(G, edges=False)
        min_lon, min_lat, max_lon, max_lat = nodes.total_bounds
        
        d_lon = (max_lon - min_lon) / config.N_X
        d_lat = (max_lat - min_lat) / config.N_Y

        # 4. Rasterize (Gán trọng số cho các ô chứa đường giao thông)
        edges = ox.graph_to_gdfs(G, nodes=False)
        for _, edge in edges.iterrows():
            # Phân loại đường để gán trọng số
            hw_type = edge.get('highway', '')
            if 'primary' in hw_type or 'trunk' in hw_type:
                weight = 1.0  # Đường quốc lộ/đại lộ
            elif 'secondary' in hw_type or 'tertiary' in hw_type:
                weight = 0.6  # Đường nhánh
            else:
                weight = 0.3  # Hẻm, khu dân cư
                
            # Ánh xạ hình học (LineString) xuống ma trận
            if hasattr(edge['geometry'], 'coords'):
                coords = list(edge['geometry'].coords)
                for lon, lat in coords:
                    j = int((lon - min_lon) / d_lon)
                    i = int((lat - min_lat) / d_lat)
                    
                    # Ràng buộc không cho văng ra khỏi ma trận
                    i = max(0, min(config.N_Y - 1, i))
                    j = max(0, min(config.N_X - 1, j))
                    
                    # Cập nhật ô lưới (Lấy Max để giữ lại đường lớn nếu đè lên nhau)
                    self.W_spatial[i, j] = max(self.W_spatial[i, j], weight)

        # Lưu lại để lần sau chạy không cần tải lại mạng
        np.save(cache_path, self.W_spatial)
        print("✅ Đã Rasterize thành công ma trận không gian giao thông!")

    def _get_temporal_weight(self, hour):
        """
        [Hàm W_temporal] Nhịp sinh học giao thông 24h.
        Sử dụng phân phối Bi-modal Gaussian (Đỉnh đôi) để mô phỏng 2 khung giờ cao điểm.
        """
        # Đỉnh sáng: 7h30 (Độ lệch chuẩn 1.5h), Đỉnh chiều: 17h30 (Độ lệch chuẩn 2.0h)
        morning_peak = np.exp(-0.5 * ((hour - 7.5) / 1.5) ** 2)
        evening_peak = np.exp(-0.5 * ((hour - 17.5) / 2.0) ** 2)
        
        # Nền ban đêm = 0.2 (20% lượng xe), Đỉnh = 1.0 (100% lượng xe)
        w_t = 0.2 + 0.8 * morning_peak + 0.8 * evening_peak
        return min(w_t, 1.0) # Đảm bảo không vượt ngưỡng 1.0

    def _get_volume_factor(self, blh):
        """
        [Hàm Phi_vol] Hệ số ép thể tích lớp biên khí quyển.
        """
        # Tránh lỗi chia cho 0 hoặc BLH quá thấp gây nổ ma trận
        safe_blh = max(blh, config.EPSILON_BLH) 
        return config.H_STD / safe_blh

    def get_emission_matrix(self, dt_time, blh):
        """
        TRÁI TIM MODULE: Sinh ra ma trận phát thải tổng S(x,y,t) cho một giờ cụ thể.
        Inputs:
            - dt_time: Đối tượng Datetime (để trích xuất giờ)
            - blh: Chiều cao lớp biên (Boundary Layer Height) từ Open-Meteo
        Output:
            - Ma trận 2D nồng độ PM2.5 sinh ra trong giờ đó (shape: 100x100)
        """
        hour = dt_time.hour
        
        W_t = self._get_temporal_weight(hour)
        Phi_vol = self._get_volume_factor(blh)
        
        # Phương trình S(x,y,t) kinh điển
        S_matrix = config.S_BASE * self.W_spatial * W_t * Phi_vol
        
        return S_matrix

# Khối Test Nhanh
if __name__ == "__main__":
    import datetime
    
    # Khởi tạo mô hình (Lần đầu chạy sẽ mất khoảng 10-15 giây để kéo bản đồ từ vệ tinh)
    em = EmissionModel()
    
    # Test Giờ cao điểm sáng (8h) với Lớp biên thấp (300m)
    test_time = datetime.datetime(2026, 6, 1, 8, 0, 0)
    S_morning = em.get_emission_matrix(test_time, blh=300.0)
    
    print(f"\nGiờ cao điểm (8:00), BLH=300m:")
    print(f"- W_temporal: {em._get_temporal_weight(8):.2f}")
    print(f"- Phi_vol: {em._get_volume_factor(300.0):.2f}")
    print(f"- Đỉnh phát thải tối đa trên ma trận: {np.max(S_morning):.2f} µg/m³/h")
    print(f"- Vùng nền (hẻm nhỏ): {np.min(S_morning):.2f} µg/m³/h")