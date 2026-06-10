# File: evaluate_results.py
import pandas as pd
import os
from src import config
from src import validator

def run_evaluation():
    print("📈 KHỞI ĐỘNG TRẠM PHÂN TÍCH & CHẤM ĐIỂM (EVALUATION NODE)")
    
    # 1. Xác định đường dẫn tới file kết quả đã lưu
    csv_path = os.path.join(config.BASE_DIR, 'outputs', 'simulation_results.csv')
    
    if not os.path.exists(csv_path):
        print("❌ Không tìm thấy dữ liệu! Bạn phải chạy 'python main_pipeline.py' trước để máy tạo ra file kết quả.")
        return
        
    print(f"📂 Đang tải dữ liệu mô phỏng từ: {csv_path}")
    df = pd.read_csv(csv_path, index_col='time', parse_dates=True)
    
    # 2. Gọi thẳng hàm Validator để chấm điểm
    # Hàm này sẽ tự động loại trừ SPINUP_HOURS (117 giờ) cấu hình trong config
    results = validator.evaluate_performance(
        simulated_pm25=df['simulated_pm25'].values, 
        observed_pm25=df['pm25'].values,
        time_series=df.index,
        dataset_name="Toàn bộ tập Test"
    )
    
    if results:
        print("\n💡 Gợi ý: Bạn có thể chạy lại file này bao nhiêu lần tùy thích để tinh chỉnh đồ thị ")
        print("mà không cần phải đợi PDE chạy lại từ đầu!")

if __name__ == "__main__":
    run_evaluation()