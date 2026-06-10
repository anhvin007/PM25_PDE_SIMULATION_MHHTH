# File: test_geo_viz.py
import numpy as np
import matplotlib.pyplot as plt
import os
from src import config

def visualize_geo_matrix():
    # Đường dẫn tới file ma trận đã được tạo ra từ Phase 3
    file_path = os.path.join(config.BASE_DIR, 'data', 'geo', 'w_spatial_100x100.npy')
    
    if not os.path.exists(file_path):
        print("❌ Không tìm thấy file! Bạn hãy chạy file 'src/emission_model.py' trước để tạo ma trận nhé.")
        return

    print("Đang load ma trận từ bộ nhớ đệm...")
    w_spatial = np.load(file_path)

    # Cài đặt khung hình trực quan
    plt.figure(figsize=(10, 8), facecolor='white')
    
    # Dùng dải màu 'hot' (Đen -> Đỏ -> Vàng -> Trắng) để làm rực sáng các tuyến đường
    # origin='lower' để tọa độ (0,0) nằm ở góc dưới cùng bên trái (chuẩn Toán học)
    im = plt.imshow(w_spatial, cmap='hot', origin='lower', extent=[0, 100, 0, 100])
    
    # Thêm thanh chú thích màu sắc
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Trọng số Phát thải $\mathcal{W}_{spatial}$', fontsize=12, rotation=270, labelpad=20)
    
    # Trang trí đồ thị
    plt.title('Mạng Lưới Giao Thông Rasterize (100x100 Lưới)\nBán kính 2km quanh Trạm Quan trắc', fontsize=14, fontweight='bold')
    plt.xlabel('Trục X (Cột ô lưới)', fontsize=11)
    plt.ylabel('Trục Y (Hàng ô lưới)', fontsize=11)
    
    # Thêm lưới mờ để dễ nhìn
    plt.grid(color='white', linestyle='--', linewidth=0.2, alpha=0.3)
    
    # Lưu ảnh vào thư mục outputs
    out_path = os.path.join(config.BASE_DIR, 'outputs', 'plots', 'geo_matrix_preview.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ Đã lưu ảnh trực quan tại: {out_path}")
    
    # Hiển thị lên màn hình
    plt.show()

if __name__ == "__main__":
    visualize_geo_matrix()