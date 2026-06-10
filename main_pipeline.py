# File: main_pipeline.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm  

from src import config
from src.emission_model import EmissionModel
from src.pde_solver import PDESolver
from src.physics_engine import wind_to_uv, get_pg_diffusivity, get_washout_coeff

def run_simulation():
    print("🚀 BƯỚC 1: Đang tải Toàn bộ tập dữ liệu Xác thực (Test Data)...")
    if not os.path.exists(config.TEST_DATA_PATH):
        print("❌ Lỗi: Không tìm thấy file test_data.csv. Bạn cần chạy data_processor.py trước!")
        return

    test_df = pd.read_csv(config.TEST_DATA_PATH, index_col='time', parse_dates=True)
    total_hours = len(test_df)
    
    print("\n⚙️ BƯỚC 2: Khởi tạo Ma trận Không gian và Động cơ PDE...")
    em = EmissionModel()
    solver = PDESolver()

    # --- HỆ THỐNG CHECKPOINTING & RESTART ---
    cube_path = os.path.join(config.BASE_DIR, 'outputs', 'history_C_cube.npy')
    csv_path = os.path.join(config.BASE_DIR, 'outputs', 'simulation_results.csv')
    
    start_idx = 0
    simulated_station_pm25 = []
    observed_station_pm25 = []
    history_C = []

    if os.path.exists(cube_path) and os.path.exists(csv_path):
        print("\n🔄 PHÁT HIỆN DỮ LIỆU CŨ! Đang kích hoạt tiến trình Khôi phục (Resume)...")
        try:
            old_df = pd.read_csv(csv_path, index_col='time', parse_dates=True)
            old_cube = np.load(cube_path)
            
            start_idx = len(old_df)
            
            if start_idx < total_hours:
                print(f"✅ Đã khôi phục thành công {start_idx} giờ. Tiếp tục chạy từ giờ thứ {start_idx + 1}...")
                
                # 1. Khôi phục Trạng thái Vật lý của Ma trận (Cực kỳ quan trọng)
                solver.C = old_cube[-1].copy()
                
                # 2. Khôi phục danh sách kết quả để nối tiếp
                simulated_station_pm25 = old_df['simulated_pm25'].tolist()
                observed_station_pm25 = old_df['pm25'].tolist()
                history_C = list(old_cube)
            else:
                print(f"✅ Dữ liệu đã hoàn tất 100% ({total_hours}/{total_hours} giờ). Không cần chạy thêm!")
                return
        except Exception as e:
            print(f"⚠️ Dữ liệu cũ bị lỗi, hệ thống sẽ chạy lại từ đầu. Lỗi: {e}")
            start_idx = 0

    if start_idx == 0:
        print(f"📊 Tổng số giờ cần mô phỏng: {total_hours} giờ. (Chạy mới hoàn toàn)")

    # Cắt bộ dữ liệu để chỉ chạy phần chưa chạy
    remaining_df = test_df.iloc[start_idx:]

    print("\n⏳ BƯỚC 3: Kích hoạt Vòng lặp Động lực học (Nhấn Ctrl+C để dừng và Lưu sớm)...")
    
    # Kiểm tra trạng thái công tắc Nudging (Mặc định là False nếu quên chưa cấu hình)
    use_nudging = getattr(config, 'USE_NUDGING_IN_TEST', False)
    mode_name = "TÁI PHÂN TÍCH ĐỒNG HÓA (CÓ NUDGING)" if use_nudging else "DỰ BÁO MÙ (KHÔNG NUDGING)"
    print(f"👉 Chế độ hiện tại: {mode_name}")
    
    try:
        # Cấu hình thanh tqdm để hiển thị đúng tiến độ tổng
        with tqdm(total=total_hours, initial=start_idx, desc="Đang mô phỏng") as pbar:
            for current_time, row in remaining_df.iterrows():
                
                ws = row['wind_speed']
                theta = row['wind_direction']
                cc = row['cloud_cover']
                blh = row['blh']
                precip = row['precipitation']
                
                # --- LOGIC NUDGING CHUẨN HỌC THUẬT ---
                obs_real = row['pm25']
                # Nếu đang ở chế độ Test Mù (Blind Test), che giấu số liệu thực tế bằng NaN
                obs_to_feed = obs_real if use_nudging else np.nan

                u, v = wind_to_uv(ws, theta)
                D = get_pg_diffusivity(ws, cc, current_time.hour)
                Lambda_rain = get_washout_coeff(precip)
                
                S_matrix = em.get_emission_matrix(current_time, blh)

                C_new = solver.step(
                    dt_hour=1.0, 
                    u=u, v=v, D=D, 
                    Lambda_rain=Lambda_rain, 
                    S_matrix=S_matrix, 
                    obs_pm25=obs_to_feed  # Truyền biến đã qua kiểm duyệt vào đây
                )

                sim_val = C_new[config.OBS_I, config.OBS_J]
                simulated_station_pm25.append(sim_val)
                observed_station_pm25.append(obs_real) # Vẫn lưu đáp án thật để chấm điểm
                
                history_C.append(C_new.copy())
                pbar.update(1)

    except KeyboardInterrupt:
        print("\n\n⚠️ NHẬN ĐƯỢC LỆNH DỪNG KHẨN CẤP TỪ NGƯỜI DÙNG!")
        print(f"🔄 Hệ thống đang kích hoạt quy trình đóng gói sớm...")

    # ==========================================
    # LƯU TRỮ DỮ LIỆU ĐÃ CHẠY HOẶC NỐI TIẾP
    # ==========================================
    ran_hours = len(simulated_station_pm25)
    if ran_hours == start_idx:
        print("❌ Chưa chạy thêm được giờ nào, hủy bỏ quá trình lưu.")
        return
        
    # Cắt df tổng hợp để lưu thành file liên tục
    test_df_sliced = test_df.iloc[:ran_hours].copy()

    print("\n💾 BƯỚC 4: Đóng gói Dữ liệu & Lưu trữ (Saving Mode)...")
    
    history_cube = np.array(history_C)
    np.save(cube_path, history_cube)
    print(f"📦 Đã xuất/cập nhật khối dữ liệu 3D: {cube_path}")
    
    test_df_sliced['simulated_pm25'] = simulated_station_pm25
    test_df_sliced.to_csv(csv_path)
    print(f"📦 Đã xuất/cập nhật Time-series CSV: {csv_path}")
    
    print("\n✅ HOÀN TẤT LƯU TRỮ AN TOÀN! Giờ bạn có thể chạy lại file này để tiếp tục, hoặc chạy 'evaluate_results.py'.")

if __name__ == '__main__':
    run_simulation()