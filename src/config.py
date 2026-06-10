# File: src/config.py
import os

# =====================================================================
# 1. THÔNG SỐ KHÔNG GIAN LƯỚI (Spatial Grid Configuration)
# =====================================================================
DX = 20.0             # Bước không gian trục X (m)
DY = 20.0             # Bước không gian trục Y (m)
N_X = 100             # Số ô lưới trục X
N_Y = 100             # Số ô lưới trục Y
DOMAIN_LENGTH = 2000.0 # Chiều dài toàn miền (m)

# Tọa độ trạm quan trắc (Measurement Anchor)
OBS_I = 50            # Chỉ số hàng (Trục Y)
OBS_J = 50            # Chỉ số cột (Trục X)

# =====================================================================
# 2. THÔNG SỐ THỜI GIAN VÀ CHIA TẬP DỮ LIỆU
# =====================================================================
DT_MACRO = 1.0        # Bước thời gian vĩ mô của dữ liệu Open-Meteo (giờ)
TRAIN_RATIO = 0.8     # Tỷ lệ chia tập Train (80%) để hiệu chỉnh
TEST_RATIO = 0.2      # Tỷ lệ chia tập Test (20%) để xác thực ngoài mẫu

# =====================================================================
# 3. THÔNG SỐ VẬT LÝ KHÍ QUYỂN & PHÁT THẢI (Physics & Emissions)
# =====================================================================
S_BASE = 0.85 #21.44        # Cường độ phát thải cơ sở (micro-gram / m^3 / h)
H_STD = 500        # Chiều cao lớp biên tiêu chuẩn (m)
EPSILON_BLH = 50.0    # Tránh chia cho 0 khi tính Phi_vol

# Thông số Rửa trôi do giáng thủy (Wet Scavenging Washout)
ALPHA_RAIN = 3.0e-4   # Hệ số bắt giữ hạt
BETA_RAIN = 0.79      # Hằng số dạng hạt PM2.5

# Hệ số lắng đọng khô
DRY_DEPOSITION = 0.01  # Bụi tự động rơi xuống mặt đường % mỗi giờ

# =====================================================================
# 4. THÔNG SỐ ĐỒNG HÓA DỮ LIỆU (Nudging / Parametric Kalman)
# =====================================================================
G_NUDGE = 0.8         # Hệ số khuếch đại lực nắn (Kalman Gain cực đại) [0 -> 1.0]
R_NUDGE = 100.0       # Bán kính ảnh hưởng thực tế (m). VD: 100m = 5 ô lưới

# =====================================================================
# 5. CẤU TRÚC ĐƯỜNG DẪN DỰ ÁN (Path Management)
# =====================================================================
# Lấy đường dẫn tuyệt đối của thư mục chứa file config.py (thư mục src)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Lùi lại 1 cấp để ra thư mục gốc dự án
BASE_DIR = os.path.dirname(SRC_DIR)

# Đường dẫn Data
RAW_METEO_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'open-meteo-10.79N106.63E6m.csv')
RAW_PM25_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'open-meteo-10.80N106.60E6m.csv')
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'train_data.csv')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'test_data.csv')
ROAD_NETWORK_PATH = os.path.join(BASE_DIR, 'data', 'geo', 'hochiminh_roads.osm')

# Đảm bảo các thư mục đầu ra tồn tại sẵn
os.makedirs(os.path.join(BASE_DIR, 'data', 'processed'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data', 'geo'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'outputs', 'animations'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'outputs', 'plots'), exist_ok=True)  