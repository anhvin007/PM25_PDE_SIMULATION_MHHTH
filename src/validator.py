# File: src/validator.py
import numpy as np

def calculate_metrics(simulated_pm25, observed_pm25):
    """
    Tính toán các chỉ số Xác thực ngoài mẫu (Out-of-sample Validation).
    """
    # 1. Loại bỏ các điểm dữ liệu thực tế bị khuyết (NaN) để không làm hỏng phép tính
    valid_idx = ~np.isnan(observed_pm25)
    sim = simulated_pm25[valid_idx]
    obs = observed_pm25[valid_idx]

    if len(obs) < 2:
        print("❌ Không đủ dữ liệu để tính toán Validation.")
        return

    # 2. Tính RMSE (Root Mean Square Error) - Phạt nặng lỗi bắt sai đỉnh
    rmse = np.sqrt(np.mean((sim - obs)**2))
    
    # 3. Tính MAE (Mean Absolute Error) - Sai số nền trung bình
    mae = np.mean(np.abs(sim - obs))

    # 4. Tính DA (Directional Accuracy) - Độ chính xác hướng dự báo
    da_correct = 0
    total_da = 0
    
    for t in range(1, len(obs)):
        delta_obs = obs[t] - obs[t-1]
        delta_sim = sim[t] - obs[t-1]
        
        if delta_obs != 0: # Bỏ qua những giờ nồng độ đứng im
            total_da += 1
            # Nếu tích của 2 vector cùng dấu (cùng tăng hoặc cùng giảm)
            if (delta_obs * delta_sim) > 0:
                da_correct += 1
                
    da = (da_correct / total_da * 100) if total_da > 0 else 0

    # In báo cáo Console chuẩn khoa học
    print("\n" + "=" * 50)
    print("🎯 BÁO CÁO XÁC THỰC MÔ HÌNH (OUT-OF-SAMPLE VALIDATION)")
    print("=" * 50)
    print(f"MAE  (Sai số tuyệt đối trung bình) : {mae:.2f} µg/m³")
    print(f"RMSE (Căn bậc hai sai số)          : {rmse:.2f} µg/m³")
    print(f"DA   (Độ chính xác hướng dự báo)   : {da:.2f} %")
    print("=" * 50 + "\n")
    
    return mae, rmse, da