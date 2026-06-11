# File: src/emission_model.py
import numpy as np
import os
from src import config

class EmissionModel:
    def __init__(self):
        """
        Khởi tạo Mô hình Phát thải Nguồn đường (Line Source Emission Model).
        Tuyệt đối KHÔNG CÓ PHÁT THẢI NỀN. Bụi chỉ sinh ra trên đường giao thông.
        """
        self.S_base = getattr(config, 'S_BASE', 1.0)
        
        # Thư mục chứa file outputs
        os.makedirs(os.path.join(config.BASE_DIR, 'outputs'), exist_ok=True)
        self.w_spatial_path = os.path.join(config.BASE_DIR, 'outputs', 'W_spatial.npy')
        
        self.road_matrix = self._load_spatial_matrix()

    def _draw_line_dda(self, matrix, x0, y0, x1, y1, weight):
        """ Thuật toán đồ họa DDA: Vẽ đường thẳng nối 2 node tọa độ lên ma trận Numpy """
        dx, dy = x1 - x0, y1 - y0
        steps = max(abs(dx), abs(dy))
        
        if steps == 0:
            if 0 <= y0 < matrix.shape[0] and 0 <= x0 < matrix.shape[1]:
                matrix[int(y0), int(x0)] = max(matrix[int(y0), int(x0)], weight)
            return
            
        x_inc, y_inc = dx / steps, dy / steps
        x, y = float(x0), float(y0)
        
        for _ in range(int(steps) + 1):
            idx_x, idx_y = int(round(x)), int(round(y))
            if 0 <= idx_y < matrix.shape[0] and 0 <= idx_x < matrix.shape[1]:
                # Dùng max để tránh các ngã tư giao nhau bị cộng dồn phát thải quá cao
                matrix[idx_y, idx_x] = max(matrix[idx_y, idx_x], weight) 
            x += x_inc
            y += y_inc

    def _generate_osm_matrix(self):
        """ Tự động tải bản đồ thực tế từ OpenStreetMap và rasterize thành lưới PDE """
        print("🌍 Bắt đầu kết nối OpenStreetMap (Bán kính 1000m)...")
        try:
            import osmnx as ox
        except ImportError:
            print("❌ Lỗi: Thư viện 'osmnx' chưa cài đặt. Vui lòng chạy: pip install osmnx")
            return None

        # Tọa độ mặc định (Trạm Tân Phú, TP.HCM)
        lat = getattr(config, 'STATION_LAT', 10.800003) 
        lon = getattr(config, 'STATION_LON', 106.600006)
         
        try:
            # Tải Graph mạng lưới đường xe chạy
            G = ox.graph_from_point((lat, lon), dist=1000, network_type='drive')
            
            # Khởi tạo ma trận nồng độ trống
            W = np.zeros((config.N_Y, config.N_X))
            
            # Trích xuất ranh giới tọa độ (Bounding Box)
            nodes = ox.graph_to_gdfs(G, edges=False)
            min_lon, max_lon = nodes['x'].min(), nodes['x'].max()
            min_lat, max_lat = nodes['y'].min(), nodes['y'].max()
            
            def latlon_to_xy(n_lat, n_lon):
                x = int((n_lon - min_lon) / (max_lon - min_lon) * (config.N_X - 1))
                y = int((n_lat - min_lat) / (max_lat - min_lat) * (config.N_Y - 1))
                return x, y
            
            # Quét toàn bộ các đoạn đường và vẽ lên ma trận
            for u, v, data in G.edges(data=True):
                x0, y0 = latlon_to_xy(G.nodes[u]['y'], G.nodes[u]['x'])
                x1, y1 = latlon_to_xy(G.nodes[v]['y'], G.nodes[v]['x'])
                
                # Phân loại trọng số đường (Đại lộ kẹt xe nhiều hơn hẻm)
                highway = str(data.get('highway', ''))
                if 'primary' in highway or 'trunk' in highway:
                    weight = 3.0
                elif 'secondary' in highway or 'tertiary' in highway:
                    weight = 2.0
                else:
                    weight = 1.0
                    
                self._draw_line_dda(W, x0, y0, x1, y1, weight)
            
            return W
            
        except Exception as e:
            print(f"⚠️ Không thể tải dữ liệu OSM (Có thể do lỗi mạng): {e}")
            return None

    def _load_spatial_matrix(self):
        """ Quản lý tiến trình nạp ma trận (Ưu tiên Cache -> OSM -> Giả lập) """
        if os.path.exists(self.w_spatial_path):
            W = np.load(self.w_spatial_path)
            print("✅ Đã load Ma trận W_spatial từ bộ nhớ đệm (Cache).")
        else:
            print("⚠️ Không tìm thấy W_spatial.npy. Đang khởi tạo bộ máy trích xuất...")
            W = self._generate_osm_matrix()
            
            if W is None:
                print("⚠️ Kích hoạt mạng lưới giao thông giả lập khẩn cấp...")
                W = np.zeros((config.N_Y, config.N_X))
                W[48:52, :] = 3.0  # Đại lộ ngang
                W[:, 20:22] = 1.0  # Hẻm dọc 1
                W[:, 80:82] = 1.0  # Hẻm dọc 2
                
            # Chuẩn hóa để đường lớn nhất (Đại lộ) có cường độ = 1.0
            if np.max(W) > 0:
                W = W / np.max(W) 
                
            # Lưu lại Cache để lần sau chạy thuật toán PDE không phải tải lại OSM
            np.save(self.w_spatial_path, W)
            print(f"💾 Đã lưu cache bản đồ thành công tại: {self.w_spatial_path}")
            
        return W

    def _get_diurnal_factor(self, hour):
        if 7 <= hour <= 9: return 1.8   
        elif 17 <= hour <= 19: return 2.2 
        elif 22 <= hour or hour <= 4: return 0.1 
        elif 11 <= hour <= 13: return 1.0   
        else: return 0.8   

    def _get_blh_factor(self, blh):
        h_std = getattr(config, 'H_STD', 500.0)
        blh_safe = max(blh, 50.0)
        return h_std / blh_safe

    def get_emission_matrix(self, current_time, blh):
        hour = current_time.hour
        temporal_factor = self._get_diurnal_factor(hour)
        stability_factor = self._get_blh_factor(blh)
        
        S_matrix = self.S_base * temporal_factor * stability_factor * self.road_matrix
        return S_matrix