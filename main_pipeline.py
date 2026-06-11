# File: main_pipeline.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm  
import joblib

from src import config
from src.emission_model import EmissionModel
from src.pde_solver import PDESolver
from src.physics_engine import wind_to_uv, get_pg_diffusivity, get_washout_coeff

def run_hybrid_simulation():
    print("🚀 BƯỚC 1: Đang tải Tập dữ liệu Xác thực (Test Data) & Mô hình Thống kê...")
    test_df = pd.read_csv(config.TEST_DATA_PATH, index_col='time', parse_dates=True)
    
    # Nếu chỉ muốn chạy một phần của tập Test để tiết kiệm thời gian
    if getattr(config, 'TEST_RUN_RATIO', 1.0) < 1.0:
        run_len = max(1, int(len(test_df) * config.TEST_RUN_RATIO))
        print(f"⚡ Chỉ chạy {config.TEST_RUN_RATIO*100:.0f}% tập Test: {run_len}/{len(test_df)} giờ")
        test_df = test_df.iloc[:run_len].copy()
    
    ecm_model_path = os.path.join(config.BASE_DIR, 'outputs', 'ecm_model.pkl')
    if not os.path.exists(ecm_model_path):
        print("❌ Lỗi: Chưa có Mô hình Sửa lỗi! Hãy chạy 'python error_correction.py' trước.")
        return
        
    ecm_model = joblib.load(ecm_model_path)
    print("✅ Đã load thành công Mô hình Sửa lỗi Thống kê (ECM).")
    
    em = EmissionModel()
    solver = PDESolver()

    cube_path = os.path.join(config.BASE_DIR, 'outputs', 'history_C_cube.npy')
    csv_path = os.path.join(config.BASE_DIR, 'outputs', 'simulation_results.csv')
    
    simulated_pde_only = []
    simulated_hybrid = []
    observed = []
    history_C = []

    print("\n⏳ BƯỚC 2: Kích hoạt Hệ thống Dự báo Lai (Hybrid: PDE + ECM)...")
    
    try:
        with tqdm(total=len(test_df), desc="Đang mô phỏng") as pbar:
            for current_time, row in test_df.iterrows():
                
                # --- PHẦN 1: MÔ HÌNH VẬT LÝ TẤT ĐỊNH (PDE DRIFT) ---
                ws, theta = row['wind_speed'], row['wind_direction']
                u, v = wind_to_uv(ws, theta)
                D = get_pg_diffusivity(ws, row['cloud_cover'], current_time.hour)
                Lambda_rain = get_washout_coeff(row['precipitation'])
                S_matrix = em.get_emission_matrix(current_time, row['blh'])

                # Chạy PDE mù tuyệt đối (KHÔNG CÓ NUDGING)
                C_new = solver.step(
                    dt_hour=1.0, u=u, v=v, D=D, 
                    Lambda_rain=Lambda_rain, S_matrix=S_matrix, obs_pm25=np.nan
                )

                f_pde = C_new[config.OBS_I, config.OBS_J]
                
                # --- PHẦN 2: MÔ HÌNH THỐNG KÊ NGẪU NHIÊN (ECM DIFFUSION) ---
                features = pd.DataFrame([{
                    'wind_speed': ws, 'wind_direction': theta, 
                    'blh': row['blh'], 'precipitation': row['precipitation'], 
                    'cloud_cover': row['cloud_cover'], 'sim_pde': f_pde,
                    'hour': current_time.hour
                }])
                
                # Dự phóng sai số kỳ vọng: E[Epsilon | M]
                predicted_residual = ecm_model.predict(features)[0]
                
                # Nồng độ Hybrid Cuối cùng = PDE + Sai số
                c_final = f_pde + predicted_residual
                
                # --- LƯU TRỮ TÁCH BIỆT (DECOUPLING) ---
                simulated_pde_only.append(f_pde)
                simulated_hybrid.append(c_final) # Báo cáo CSV lấy điểm số Hybrid
                observed.append(row['pm25'])
                
                # LƯU MA TRẬN VẬT LÝ THUẦN TÚY: Giúp video 3D hiện rõ các dải khói trên đường bộ
                history_C.append(C_new.copy())
                pbar.update(1)

    except KeyboardInterrupt:
        print("\n⚠️ NHẬN LỆNH DỪNG! Đang lưu dữ liệu...")

    ran_hours = len(simulated_hybrid)
    test_df_sliced = test_df.iloc[:ran_hours].copy()

    print("\n💾 BƯỚC 3: Đóng gói Kết quả...")
    # Lưu khối Cube 3D nguyên bản
    np.save(cube_path, np.array(history_C))
    
    test_df_sliced['sim_pde'] = simulated_pde_only
    test_df_sliced['simulated_pm25'] = simulated_hybrid # Kết quả Hybrid lấy làm chính
    test_df_sliced.to_csv(csv_path)
    
    print(f"📦 Đã xuất Time-series CSV có chứa cột Hybrid tại: {csv_path}")
    print("\n✅ HOÀN TẤT! Bạn hãy chạy 'evaluate_results.py' để xem chỉ số RMSE Hybrid!")

if __name__ == '__main__':
    run_hybrid_simulation()