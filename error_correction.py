# File: error_correction.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

from src import config
from src.emission_model import EmissionModel
from src.pde_solver import PDESolver
from src.physics_engine import wind_to_uv, get_pg_diffusivity, get_washout_coeff

def train_ecm_model():
    print("🧠 BƯỚC 1/2: Đang chạy PDE để trích xuất Chuỗi Sai số (Residuals)...")
    train_df = pd.read_csv(config.TRAIN_DATA_PATH, index_col='time', parse_dates=True)
    spinup_hours = getattr(config, 'SPINUP_HOURS', 117)
    total_hours = len(train_df)
    
    em = EmissionModel()
    solver = PDESolver()
    
    # --- ĐƯỜNG DẪN LƯU CHECKPOINT ---
    chkpt_csv_path = os.path.join(config.BASE_DIR, 'outputs', 'ecm_sim_checkpoint.csv')
    chkpt_mat_path = os.path.join(config.BASE_DIR, 'outputs', 'ecm_matrix_checkpoint.npy')
    
    start_idx = 0
    simulated = []

    # --- CƠ CHẾ KHÔI PHỤC (RESUME) ---
    if os.path.exists(chkpt_csv_path) and os.path.exists(chkpt_mat_path):
        print("\n🔄 PHÁT HIỆN DỮ LIỆU ĐANG CHẠY DỞ! Kích hoạt tiến trình Khôi phục...")
        try:
            old_sim_df = pd.read_csv(chkpt_csv_path)
            simulated = old_sim_df['sim_pde'].tolist()
            start_idx = len(simulated)
            
            if start_idx < total_hours:
                print(f"✅ Đã khôi phục {start_idx} giờ. Tiếp tục chạy từ giờ thứ {start_idx + 1}...")
                # Khôi phục trạng thái vật lý của ma trận PDE tại thời điểm bị ngắt
                solver.C = np.load(chkpt_mat_path)
            else:
                print(f"✅ PDE đã hoàn tất 100% từ trước ({total_hours}/{total_hours} giờ). Chuyển thẳng sang Huấn luyện!")
        except Exception as e:
            print(f"⚠️ Dữ liệu Checkpoint bị lỗi, hệ thống sẽ chạy lại từ đầu. Lỗi: {e}")
            start_idx = 0
            simulated = []

    if start_idx < total_hours:
        remaining_df = train_df.iloc[start_idx:]
        
        try:
            with tqdm(total=total_hours, initial=start_idx, desc="Đang trích xuất") as pbar:
                for current_time, row in remaining_df.iterrows():
                    ws, theta = row['wind_speed'], row['wind_direction']
                    u, v = wind_to_uv(ws, theta)
                    D = get_pg_diffusivity(ws, row['cloud_cover'], current_time.hour)
                    Lambda_rain = get_washout_coeff(row['precipitation'])
                    S_matrix = em.get_emission_matrix(current_time, row['blh'])
                    
                    # Chạy dự báo mù (f_PDE)
                    C_new = solver.step(1.0, u, v, D, Lambda_rain, S_matrix, np.nan)
                    simulated.append(C_new[config.OBS_I, config.OBS_J])
                    pbar.update(1)
                    
        except KeyboardInterrupt:
            print("\n\n⚠️ NHẬN LỆNH DỪNG KHẨN CẤP! Đang đóng gói dữ liệu Checkpoint...")
            # 1. Lưu chuỗi 1D đã chạy được
            pd.DataFrame({'sim_pde': simulated}).to_csv(chkpt_csv_path, index=False)
            # 2. Lưu trạng thái ma trận 2D cuối cùng để lần sau nối tiếp
            np.save(chkpt_mat_path, solver.C)
            print("📦 Đã lưu tiến độ an toàn. Lần sau chạy lại lệnh sẽ tự động nối tiếp.")
            return # Thoát hàm sớm, không huấn luyện mô hình khi dữ liệu chưa đủ

    # Nếu mã chạy đến đây tức là PDE đã hoàn tất 100% dữ liệu Train
    print("\n🌲 BƯỚC 2/2: Huấn luyện Ma trận Trạng thái (Random Forest ECM)...")
    
    train_df['sim_pde'] = simulated
    
    # Cắt bỏ vùng nhiễu Spin-up
    train_df = train_df.iloc[spinup_hours:].copy()
    
    # 1. TÍNH TOÁN PHẦN DƯ (RESIDUALS: Epsilon = Y_true - Y_pred)
    train_df['residual'] = train_df['pm25'] - train_df['sim_pde']
    
    # Lọc bỏ các dòng bị khuyết dữ liệu trạm đo thực tế
    train_clean = train_df.dropna(subset=['pm25', 'residual']).copy()
    
    # 2. XÂY DỰNG KHÔNG GIAN ĐẶC TRƯNG (Feature Space)
    X = train_clean[['wind_speed', 'wind_direction', 'blh', 'precipitation', 'cloud_cover', 'sim_pde']]
    X = X.assign(hour=train_clean.index.hour)
    
    Y = train_clean['residual']
    
    # Khởi tạo mô hình học máy phi tuyến
    ecm_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    ecm_model.fit(X, Y)
    
    # Lưu mô hình Random Forest vào ổ cứng
    model_path = os.path.join(config.BASE_DIR, 'outputs', 'ecm_model.pkl')
    joblib.dump(ecm_model, model_path)
    
    print(f"✅ Đã đóng gói Mô hình Không gian Trạng thái tại: {model_path}")
    print(f"👉 Mức độ giải thích (R^2 Score) của mô hình: {ecm_model.score(X, Y):.2f}")
    
    # Dọn dẹp file Checkpoint thừa sau khi huấn luyện thành công (Tùy chọn)
    if os.path.exists(chkpt_csv_path): os.remove(chkpt_csv_path)
    if os.path.exists(chkpt_mat_path): os.remove(chkpt_mat_path)
    
    print("🚀 Sẵn sàng chạy Tập Test ở chế độ Hybrid!")

if __name__ == "__main__":
    train_ecm_model()