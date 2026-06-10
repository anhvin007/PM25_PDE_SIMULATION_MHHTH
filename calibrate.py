# File: src/calibrate.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from src import config
from src.emission_model import EmissionModel
from src.pde_solver import PDESolver
from src.physics_engine import wind_to_uv, get_pg_diffusivity, get_washout_coeff

def run_calibration_grid_search():
    print("🔍 KHỞI ĐỘNG HỆ THỐNG HIỆU CHỈNH THAM SỐ (CALIBRATION)")
    
    train_df = pd.read_csv(config.TRAIN_DATA_PATH, index_col='time', parse_dates=True)
    train_df = train_df.iloc[:720] 
    
    # Đọc động từ file config thay vì hardcode số 117
    spinup_hours = config.SPINUP_HOURS
    print(f"⏱️ Sẽ loại bỏ {spinup_hours} giờ đầu tiên (Spin-up time) từ config.py khi tính toán RMSE.")
    
    # 2. Không gian tìm kiếm (Grid Search Space)
    # Tinh chỉnh 2 tham số nhạy cảm nhất: Nguồn xả (S_base) và Lực nắn (R_nudge)
    search_space = [
        {'S_base': 0.8567,  'R_nudge': 200.0, 'G_nudge': 0.8}, 
        {'S_base': 0.8567,  'R_nudge': 400.0, 'G_nudge': 0.8}, 
        {'S_base': 0.8567,  'R_nudge': 600.0, 'G_nudge': 0.9}, 
        {'S_base': 0.8567,  'R_nudge': 800.0, 'G_nudge': 0.95},
        {'S_base': 0.8567,  'R_nudge': 900.0, 'G_nudge': 0.9},
        {'S_base': 0.8567,  'R_nudge': 1000.0, 'G_nudge': 0.9}
    ]
    
    best_rmse = float('inf')
    best_params = None
    
    for params in search_space:
        print(f"\n⚙️ Đang thử nghiệm bộ tham số: {params}")
        
        # Ghi đè cấu hình tạm thời
        config.S_BASE = params['S_base']
        config.R_NUDGE = params['R_NUDGE'] = params['R_nudge']
        config.G_NUDGE = params['G_nudge']
        
        em = EmissionModel()
        solver = PDESolver() # Khởi tạo lại solver để reset ma trận về 0 và update Kernel mới
        
        simulated = []
        observed = []
        
        # Chạy PDE
        for current_time, row in tqdm(train_df.iterrows(), total=len(train_df), leave=False):
            ws, theta = row['wind_speed'], row['wind_direction']
            u, v = wind_to_uv(ws, theta)
            D = get_pg_diffusivity(ws, row['cloud_cover'], current_time.hour)
            Lambda_rain = get_washout_coeff(row['precipitation'])
            
            S_matrix = em.get_emission_matrix(current_time, row['blh'])
            
            C_new = solver.step(1.0, u, v, D, Lambda_rain, S_matrix, row['pm25'])
            
            simulated.append(C_new[config.OBS_I, config.OBS_J])
            observed.append(row['pm25'])
            
        # Tính RMSE (Bỏ qua 117 giờ đầu tiên để loại trừ hiện tượng Spin-up)
        sim = np.array(simulated)[spinup_hours:]
        obs = np.array(observed)[spinup_hours:]
        
        valid = ~np.isnan(obs)
        rmse = np.sqrt(np.mean((sim[valid] - obs[valid])**2))
        
        print(f"👉 RMSE Đạt được: {rmse:.2f} µg/m³")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    print("\n" + "="*50)
    print("🏆 BỘ THAM SỐ TỐI ƯU NHẤT SAU KHI TRAIN:")
    print(f"- S_BASE  : {best_params['S_base']}")
    print(f"- R_NUDGE : {best_params['R_nudge']}")
    print(f"- G_NUDGE : {best_params['G_nudge']}")
    print(f"- RMSE    : {best_rmse:.2f} µg/m³")
    print("="*50)
    print("💡 Hãy mở file src/config.py và cập nhật các con số này trước khi chạy lại Test!")

if __name__ == "__main__":
    run_calibration_grid_search()