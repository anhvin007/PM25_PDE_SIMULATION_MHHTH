# File: src/spinup_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from src import config
from src.emission_model import EmissionModel
from src.pde_solver import PDESolver
from src.physics_engine import wind_to_uv, get_pg_diffusivity, get_washout_coeff

def analyze_spin_up_time(test_hours=720):
    """
    Chạy mô phỏng để tìm ra điểm bão hòa của ma trận (Spin-up time).
    Mặc định chạy thử 15 ngày (360 giờ) trên tập Train.
    """
    print(f"🔍 Đang phân tích Thời gian Khởi động lạnh (Spin-up) trong {test_hours} giờ...")
    
    # Load 1 đoạn dữ liệu từ tập Train
    df = pd.read_csv(config.TRAIN_DATA_PATH, index_col='time', parse_dates=True).iloc[:test_hours]
    
    em = EmissionModel()
    # Tắt Nudging để quan sát bản chất vật lý thuần túy của PDE
    original_g = config.G_NUDGE
    config.G_NUDGE = 0.0 
    solver = PDESolver()
    
    mean_concentrations = []
    
    for current_time, row in tqdm(df.iterrows(), total=len(df)):
        ws, theta = row['wind_speed'], row['wind_direction']
        u, v = wind_to_uv(ws, theta)
        D = get_pg_diffusivity(ws, row['cloud_cover'], current_time.hour)
        Lambda_rain = get_washout_coeff(row['precipitation'])
        
        S_matrix = em.get_emission_matrix(current_time, row['blh'])
        
        # Chạy PDE 1 bước
        C_new = solver.step(1.0, u, v, D, Lambda_rain, S_matrix, obs_pm25=np.nan)
        
        # Ghi nhận NỒNG ĐỘ TRUNG BÌNH CỦA TOÀN BỘ MA TRẬN
        mean_concentrations.append(np.mean(C_new))
        
    config.G_NUDGE = original_g # Trả lại thông số cũ
    
    # --- TOÁN HỌC TÌM ĐIỂM SPIN-UP ---
    mean_series = pd.Series(mean_concentrations)
    
    # 1. Làm mượt chuỗi bằng Moving Average (Chu kỳ 24h) để bỏ nhiễu ngày đêm
    smoothed_trend = mean_series.rolling(window=24, min_periods=1).mean()
    
    # 2. Tính đạo hàm bậc 1 (Gradient / Tốc độ thay đổi)
    gradient = np.gradient(smoothed_trend)
    
    # 3. Tìm thời điểm mà Đạo hàm tiệm cận 0 (tức là đường cong đi ngang)
    # Ngưỡng hội tụ: Tốc độ tăng trưởng trung bình rơi xuống dưới 0.1 µg/m3/giờ
    convergence_threshold = 0.1
    spinup_hours = 0
    for t in range(24, len(gradient)): # Bỏ qua ngày đầu tiên vì dao động cực mạnh
        if abs(gradient[t]) < convergence_threshold and abs(gradient[t+1]) < convergence_threshold:
            spinup_hours = t
            break
            
    if spinup_hours == 0:
        spinup_hours = len(gradient) # Nếu không tìm thấy, lấy toàn bộ
        print("⚠️ Cảnh báo: Lượng phát thải quá lớn, ma trận chưa đạt bão hòa trong khung thời gian test.")
        
    # --- VẼ ĐỒ THỊ TRỰC QUAN ---
    plt.figure(figsize=(12, 6), facecolor='white')
    
    # Vẽ đường nồng độ thực tế biến thiên
    plt.plot(mean_series, label='Nồng độ Trung bình Lưới (Thực tế)', color='lightgray', alpha=0.7)
    
    # Vẽ đường xu hướng (Trend)
    plt.plot(smoothed_trend, label='Xu hướng Tích lũy (MA-24h)', color='red', linewidth=2)
    
    # Cắm cờ đánh dấu điểm Spin-up
    plt.axvline(x=spinup_hours, color='blue', linestyle='--', linewidth=2, 
                label=f'Điểm bão hòa (Spin-up) = {spinup_hours} giờ')
                
    # Trang trí
    plt.title('Phân Tích Spin-up Time Của Phương Trình Đối Lưu - Khuếch Tán', fontsize=14, fontweight='bold')
    plt.xlabel('Thời gian mô phỏng (Giờ)', fontsize=12)
    plt.ylabel('Nồng độ $PM_{2.5}$ Trung bình ($\mu g/m^3$)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    out_path = os.path.join(config.BASE_DIR, 'outputs', 'plots', 'spinup_analysis.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*50)
    print(f"✅ Đã tìm ra điểm cân bằng vật lý!")
    print(f"👉 Bạn nên loại bỏ: {spinup_hours} giờ đầu tiên (Khoảng {spinup_hours/24:.1f} ngày) khi tính RMSE.")
    print(f"📊 Biểu đồ trực quan đã được lưu tại: {out_path}")
    print("="*50)
    
    plt.show()
    return spinup_hours

if __name__ == "__main__":
    analyze_spin_up_time()