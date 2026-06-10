# File: main_pipeline.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm  # Thư viện tạo thanh tiến trình cực đẹp

from src import config
from src.emission_model import EmissionModel
from src.pde_solver import PDESolver
from src.physics_engine import wind_to_uv, get_pg_diffusivity, get_washout_coeff
from src import validator

def run_simulation():
    print("🚀 BƯỚC 1: Đang tải tập dữ liệu Xác thực (Test Data)...")
    if not os.path.exists(config.TEST_DATA_PATH):
        print("❌ Lỗi: Không tìm thấy file test_data.csv. Bạn cần chạy data_processor.py trước!")
        return

    test_df = pd.read_csv(config.TEST_DATA_PATH, index_col='time', parse_dates=True)
    
    # -------------------------------------------------------------
    # [TÙY CHỈNH] Rút ngắn số lượng vòng lặp để test code nhanh
    # Chạy mô phỏng 72 giờ đầu tiên (3 ngày). 
    # Khi chạy thật, hãy comment dòng này lại.
    test_df = test_df.iloc[:72]
    # -------------------------------------------------------------

    print("\n⚙️ BƯỚC 2: Khởi tạo Ma trận Không gian và Động cơ PDE...")
    em = EmissionModel()
    solver = PDESolver()

    # Khởi tạo các mảng lưu trữ để làm Báo cáo và Vẽ Đồ họa
    simulated_station_pm25 = []
    observed_station_pm25 = []
    history_C = [] # Lưu lại toàn bộ ma trận không gian qua từng giờ để làm Video

    print("\n⏳ BƯỚC 3: Kích hoạt Vòng lặp Động lực học Không gian - Thời gian...")
    # tqdm sẽ tự động tạo một thanh tiến trình chạy % dưới terminal
    for current_time, row in tqdm(test_df.iterrows(), total=len(test_df)):
        
        # 1. Khai thác dữ liệu Khí tượng
        ws = row['wind_speed']
        theta = row['wind_direction']
        cc = row['cloud_cover']
        blh = row['blh']
        precip = row['precipitation']
        obs_pm25 = row['pm25']

        # 2. Tham số hóa Động lực học (Vật lý)
        u, v = wind_to_uv(ws, theta)
        D = get_pg_diffusivity(ws, cc, current_time.hour)
        Lambda_rain = get_washout_coeff(precip)
        
        # Sinh ma trận nguồn phát thải S(x,y) cho giờ hiện tại
        S_matrix = em.get_emission_matrix(current_time, blh)

        # 3. Kích hoạt bước nhảy PDE 1 giờ
        C_new = solver.step(
            dt_hour=1.0, 
            u=u, v=v, D=D, 
            Lambda_rain=Lambda_rain, 
            S_matrix=S_matrix, 
            obs_pm25=obs_pm25
        )

        # 4. Lưu lại kết quả
        # Lấy giá trị tại đúng ô trung tâm trạm đo (50, 50)
        sim_val = C_new[config.OBS_I, config.OBS_J]
        simulated_station_pm25.append(sim_val)
        observed_station_pm25.append(obs_pm25)
        
        # Lưu bản copy của ma trận không gian để Phase 6 làm Xưởng Trực quan
        history_C.append(C_new.copy())

    print("\n✅ BƯỚC 4: Xác thực Hiệu suất Toán học...")
    validator.calculate_metrics(
        np.array(simulated_station_pm25), 
        np.array(observed_station_pm25)
    )

    print("\n💾 BƯỚC 5: Đóng gói Dữ liệu Đồ họa...")
    # Lưu toàn bộ khối ma trận 3D [Thời gian, Y, X] ra file numpy nhị phân siêu nhẹ
    history_cube = np.array(history_C)
    out_path = os.path.join(config.BASE_DIR, 'outputs', 'history_C_cube.npy')
    np.save(out_path, history_cube)
    
    # Lưu lại file thời tiết kèm nồng độ dự báo để đối chiếu
    test_df['simulated_pm25'] = simulated_station_pm25
    test_df.to_csv(os.path.join(config.BASE_DIR, 'outputs', 'simulation_results.csv'))
    
    print(f"📦 Đã xuất khối dữ liệu 3D thành công! Chuẩn bị chuyển sang Xưởng Trực quan hóa.")

if __name__ == '__main__':
    run_simulation()