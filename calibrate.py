# File: src/calibrate.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import optuna
import os

from src import config
from src.emission_model import EmissionModel
from src.pde_solver import PDESolver
from src.physics_engine import wind_to_uv, get_pg_diffusivity, get_washout_coeff

# Thiết lập đường dẫn cơ sở dữ liệu SQLite để làm bộ nhớ chốt (Checkpoint Storage)
db_dir = os.path.join(config.BASE_DIR, 'outputs')
os.makedirs(db_dir, exist_ok=True)
db_path = os.path.join(db_dir, 'optuna_calibration.db')
# Chuỗi kết nối chuẩn SQLite của Optuna
storage_url = f"sqlite:///{os.path.abspath(db_path)}"

print("🚀 BƯỚC 1: Đang tải dữ liệu huấn luyện lịch sử (Train Data)...")
train_df = pd.read_csv(config.TRAIN_DATA_PATH, index_col='time', parse_dates=True)
train_df = train_df.iloc[:720]  # Thử nghiệm trên 720 giờ (1 tháng mẫu)
spinup_hours = getattr(config, 'SPINUP_HOURS', 117)

def objective(trial):
    """ Hàm mục tiêu tính toán sai số PDE phục vụ Tối ưu hóa Bayes """
    trial_num = trial.number
    
    print(f"\n" + "-"*60)
    print(f"🔔 [TRIAL {trial_num:02d}] ⚙️ Khởi động cấu hình tìm kiếm mới...")
    
    # 1. Định nghĩa Không gian Siêu tham số liên tục
    s_base = trial.suggest_float('S_base', 0.1, 1.9)
    h_std = trial.suggest_float('H_std', 270, 730)
    lambda_dry = trial.suggest_float('Lambda_dry', 0.01, 0.5)
    
    print(f"   👉 Các tham số đề xuất từ AI Xác suất:")
    print(f"      🔹 S_base     = {s_base:.4f} µg/m³/h")
    print(f"      🔹 H_std      = {h_std:.2f} m")
    print(f"      🔹 Lambda_dry = {lambda_dry:.4f} h⁻¹")
    
    # Ghi đè cấu hình tạm thời vào hệ thống
    config.S_BASE = s_base
    config.H_STD = h_std
    config.LAMBDA_DRY = lambda_dry
    
    em = EmissionModel()
    solver = PDESolver()
    
    simulated, observed = [], []
    
    # 2. Vận hành vòng lặp PDE "Mù" (Tắt hoàn toàn Kalman Nudging)
    # Thêm mô tả trực quan vào thanh tiến trình nội tại của từng Trial
    desc_str = f"   📊 Chạy PDE Trial {trial_num:02d}"
    for current_time, row in tqdm(train_df.iterrows(), total=len(train_df), desc=desc_str, leave=False):
        ws, theta = row['wind_speed'], row['wind_direction']
        u, v = wind_to_uv(ws, theta)
        D = get_pg_diffusivity(ws, row['cloud_cover'], current_time.hour)
        Lambda_rain = get_washout_coeff(row['precipitation'])
        S_matrix = em.get_emission_matrix(current_time, row['blh'])
        
        # Giải tích vi phân bước nhảy 1 giờ
        C_new = solver.step(1.0, u, v, D, Lambda_rain, S_matrix, np.nan)
        
        simulated.append(C_new[config.OBS_I, config.OBS_J])
        observed.append(row['pm25'])
        
    # 3. Phân tích thống kê sai số hậu kỳ
    sim = np.array(simulated)[spinup_hours:]
    obs = np.array(observed)[spinup_hours:]
    valid = ~np.isnan(obs)
    
    if np.sum(valid) == 0:
        print(f"   ❌ [TRIAL {trial_num:02d}] Lỗi: Tập dữ liệu thực nghiệm không chứa mốc quan trắc hợp lệ!")
        return float('inf')
        
    rmse = np.sqrt(np.mean((sim[valid] - obs[valid])**2))
    
    print(f"   🎯 [TRIAL {trial_num:02d}] Kết thúc vòng tính toán. Sai số thu được:")
    print(f"      🔻 RMSE cục bộ = {rmse:.2f} µg/m³")
    
    # In ra kỷ lục tốt nhất hiện tại nếu có thông tin nghiên cứu lịch sử
    try:
        best_so_far = trial.study.best_value
        if rmse < best_so_far:
            print(f"      🎉 KỶ LỤC MỚI! Sai số giảm từ {best_so_far:.2f} ➔ {rmse:.2f} µg/m³")
    except ValueError:
        pass # Vòng đầu tiên chưa có best_value
        
    return rmse

def run_bayesian_calibration():
    print(f"\n🔍 BƯỚC 2: KHỞI ĐỘNG HỆ THỐNG TỐI ƯU HÓA BAYES (BAYESIAN OPTIMIZATION)")
    print(f"💾 File lưu trữ Checkpoint: {db_path}")
    print(f"⏱️ Tiêu chuẩn cô lập: Tự động gạt bỏ {spinup_hours} giờ khởi động lạnh (Spin-up).")
    
    # Khởi tạo Study tích hợp SQLite Storage
    # load_if_exists=True: Nếu thấy file DB cũ, tự động nạp lại trạng thái cũ để chạy tiếp!
    study = optuna.create_study(
        study_name="pde_global_calibration",
        storage=storage_url,
        direction='minimize',
        load_if_exists=True
    )
    
    # Thống kê số lượng vòng lặp đã hoàn thành trong quá khứ
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    total_trials = 1
    
    if completed_trials > 0:
        print(f"\n🔄 PHÁT HIỆN ĐIỂM CHỐT CŨ (BREAKPOINT)!")
        print(f"   ➔ Hệ thống đã hoàn thành {completed_trials}/{total_trials} vòng lặp trước đó.")
        print(f"   ➔ Sẽ tiếp tục giải từ Vòng lặp thứ {completed_trials + 1}...")
    else:
        print(f"   ➔ Không tìm thấy dữ liệu cũ. Khởi tạo tiến trình tìm kiếm mới hoàn toàn.")

    remaining_trials = total_trials - completed_trials
    
    if remaining_trials <= 0:
        print(f"\n✅ Dữ liệu tối ưu hóa đã hoàn tất đầy đủ {total_trials} vòng từ trước. Không cần chạy thêm!")
    else:
        try:
            # Vận hành tìm kiếm Bayes trên số vòng còn lại
            study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False)
        except KeyboardInterrupt:
            print("\n\n⚠️ NHẬN ĐƯỢC LỆNH NGẮT KHẨN CẤP (CTRL+C) TỪ NGƯỜI DÙNG!")
            print(f"💾 Toàn bộ {len(study.trials)} trạng thái tham số đã được khóa và ghi an toàn vào file SQLite.")
            print("🚀 Bạn có thể chạy lại file này bất kỳ lúc nào để tiếp tục tiến trình!")
            return

    # In bảng điểm danh dự cuối cùng
    print("\n" + "="*60)
    print("🏆 KẾT QUẢ TỐI ƯU HÓA TOÀN CỤC HOÀN CHỈNH (BEST CONFIGURATION)")
    print("="*60)
    print(f"   - Tổng số vòng lặp đã duyệt : {len(study.trials)}")
    print(f"   - S_BASE (Cường độ xả nền) : {study.best_params['S_base']:.4f} µg/m³/h")
    print(f"   - H_STD  (Trần khí quyển)   : {study.best_params['H_std']:.2f} m")
    print(f"   - LAMBDA_DRY (Lắng đọng khô): {study.best_params['Lambda_dry']:.4f} h⁻¹")
    print(f"   ➔ Kỷ lục RMSE nhỏ nhất đạt được: {study.best_value:.2f} µg/m³")
    print("="*60)
    print("💡 Hãy mở file src/config.py, cập nhật 3 hằng số trên trước khi chạy error_correction.py!")

if __name__ == "__main__":
    run_bayesian_calibration()