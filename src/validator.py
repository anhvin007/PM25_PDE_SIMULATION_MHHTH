# File: src/validator.py
import numpy as np
import os
import matplotlib.pyplot as plt
from src import config

def evaluate_performance(simulated_pm25, observed_pm25, time_series, spinup_hours=config.SPINUP_HOURS, dataset_name="Tập Test"):
    """
    Tính toán chỉ số và vẽ biểu đồ Xác thực ngoài mẫu.
    Tự động trích xuất hằng số Spin-up từ config.py làm cấu hình mặc định.
    """
    print(f"\n🔍 ĐANG ĐÁNH GIÁ HIỆU SUẤT TRÊN {dataset_name.upper()}...")
    print(f"⏱️ Đã tự động cắt bỏ {spinup_hours} giờ khởi động lạnh (Spin-up time).")
    
    # 1. Cắt bỏ giai đoạn Spin-up để đảm bảo tính công bằng học thuật
    sim = np.array(simulated_pm25)[spinup_hours:]
    obs = np.array(observed_pm25)[spinup_hours:]
    times = time_series[spinup_hours:]
    
    # 2. Lọc bỏ các giá trị thực tế bị khuyết (NaN do trạm đo bảo trì)
    valid_mask = ~np.isnan(obs)
    sim_clean = sim[valid_mask]
    obs_clean = obs[valid_mask]
    
    if len(obs_clean) < 2:
        print("❌ Cảnh báo: Không đủ dữ liệu hợp lệ để đánh giá sau khi lọc NaN.")
        return None

    # ==========================================
    # KHỐI TOÁN HỌC: TÍNH TOÁN CÁC CHỈ SỐ
    # ==========================================
    
    # 1. MAE (Mean Absolute Error) - Sai số nền
    mae = np.mean(np.abs(sim_clean - obs_clean))
    
    # 2. RMSE (Root Mean Square Error) - Phạt nặng lỗi sai đỉnh
    rmse = np.sqrt(np.mean((sim_clean - obs_clean)**2))
    
    # 3. DA (Directional Accuracy) - Độ chính xác hướng
    # Tính đạo hàm bậc 1 (sự thay đổi nồng độ giữa 2 giờ liên tiếp)
    delta_sim = np.diff(sim_clean)
    delta_obs = np.diff(obs_clean)
    
    # Chỉ xét những thời điểm nồng độ thực tế có thay đổi (bỏ qua lúc đồ thị đi ngang)
    valid_dirs = (delta_obs != 0)
    
    # Nếu tích của 2 đạo hàm > 0 (tức là cùng dấu: cùng tăng hoặc cùng giảm) -> Đoán đúng hướng!
    correct_dirs = (delta_sim[valid_dirs] * delta_obs[valid_dirs]) > 0
    da = np.mean(correct_dirs) * 100 if np.any(valid_dirs) else 0.0

    # ==========================================
    # IN BÁO CÁO RA TERMINAL
    # ==========================================
    print("\n" + "=" * 60)
    print("🎯 BÁO CÁO XÁC THỰC MÔ HÌNH (OUT-OF-SAMPLE VALIDATION)")
    print("=" * 60)
    print(f"🔹 MAE  (Sai số trung bình ngày thường): {mae:.2f} µg/m³")
    print(f"🔹 RMSE (Khả năng bắt đỉnh ô nhiễm)   : {rmse:.2f} µg/m³")
    print(f"🔹 DA   (Độ chính xác hướng cảnh báo)  : {da:.2f} %")
    print("=" * 60 + "\n")
    
    # ==========================================
    # VẼ BIỂU ĐỒ CHỨNG MINH KẾT QUẢ (TIME-SERIES PLOT)
    # ==========================================
    plt.figure(figsize=(14, 6), facecolor='white')
    
    plt.plot(times[valid_mask], obs_clean, label='Quan trắc Thực tế (Trạm)', color='black', linewidth=1.5, marker='.', markersize=4, alpha=0.7)
    plt.plot(times[valid_mask], sim_clean, label='Dự báo PDE', color='red', linewidth=1.5, alpha=0.9)
    
    plt.title(f'Đối chiếu Nồng độ $PM_{{2.5}}$ - {dataset_name} (MAE: {mae:.2f} | DA: {da:.1f}%)', fontsize=14, fontweight='bold')
    plt.xlabel('Thời gian', fontsize=12)
    plt.ylabel('Nồng độ ($\mu g/m^3$)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Tô màu cảnh báo nền đỏ nếu nồng độ vượt ngưỡng an toàn (ví dụ > 50)
    plt.axhline(y=50, color='orange', linestyle='-.', label='Ngưỡng rủi ro (50 $\mu g/m^3$)')
    
    # Lưu ảnh đồ thị
    out_path = os.path.join(config.BASE_DIR, 'outputs', 'plots', 'validation_timeseries.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã xuất biểu đồ đối chiếu tại: {out_path}")
    
    # Trả về bộ từ điển (dictionary) chứa kết quả
    return {'mae': mae, 'rmse': rmse, 'da': da}