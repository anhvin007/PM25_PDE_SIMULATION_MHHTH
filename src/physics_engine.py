# File: src/physics_engine.py
import numpy as np
from src import config

def wind_to_uv(ws_kmh, theta_deg):
    """
    Chuyển đổi Tốc độ gió (km/h) và Hướng gió (độ) sang Vector Động lượng (u, v) (m/h).
    Tuân thủ sự đảo chiều hệ quy chiếu Khí tượng học -> Toán học.
    """
    ws_mh = ws_kmh * 1000.0  # Đổi km/h sang m/h
    theta_rad = np.radians(theta_deg)
    
    # Dấu (-) để chỉ hướng vật chất đi TỚI đâu
    u = -ws_mh * np.sin(theta_rad)
    v = -ws_mh * np.cos(theta_rad)
    return u, v

def get_pg_diffusivity(ws_kmh, cc_percent, hour):
    """
    Định lượng hệ số khuếch tán D_PDE (m^2/h) dựa trên ma trận Pasquill-Gifford.
    """
    v_10 = ws_kmh / 3.6  # m/s
    is_daytime = (6 <= hour < 18)
    
    pg_class = 'D' # Mặc định Trung tính
    
    if is_daytime:
        if cc_percent < 50.0:
            if v_10 <= 2.0: pg_class = 'A'
            elif v_10 <= 5.0: pg_class = 'B'
            else: pg_class = 'C'
        else:
            if v_10 <= 2.0: pg_class = 'B'
            elif v_10 <= 5.0: pg_class = 'C'
            else: pg_class = 'D'
    else: # Ban đêm
        if cc_percent >= 50.0:
            if v_10 <= 5.0: pg_class = 'E' if v_10 <= 2.0 else 'D'
            else: pg_class = 'D'
        else:
            if v_10 <= 2.0: pg_class = 'F'
            elif v_10 <= 5.0: pg_class = 'E'
            else: pg_class = 'D'
            
    # Ánh xạ Cấp độ sang D (m^2/s) rồi nhân 3600 ra m^2/h
    d_mapping = {'A': 50.0, 'B': 30.0, 'C': 20.0, 'D': 10.0, 'E': 5.0, 'F': 2.0}
    return d_mapping[pg_class] * 3600.0

def get_washout_coeff(precip_mmh):
    """
    Tính hệ số chìm do giáng thủy (Washout / Wet Scavenging).
    """
    if precip_mmh <= 0:
        return 0.0
    return config.ALPHA_RAIN * (precip_mmh ** config.BETA_RAIN)

def generate_nudging_kernel():
    """
    Khởi tạo Ma trận Nudging Gaussian Kernel tĩnh (chỉ chạy 1 lần để tối ưu RAM).
    """
    y, x = np.ogrid[0:config.N_Y, 0:config.N_X]
    # Tính khoảng cách bình phương từ mọi điểm tới Trạm đo (OBS_I, OBS_J)
    d_sq = ((x - config.OBS_J) * config.DX)**2 + ((y - config.OBS_I) * config.DY)**2
    # Hàm chuông Gaussian phân rã
    kernel = np.exp(-d_sq / (2 * config.R_NUDGE**2))
    return kernel