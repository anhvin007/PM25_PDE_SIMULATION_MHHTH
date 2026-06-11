# File: src/traffic_animation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from src import config
from src.physics_engine import wind_to_uv

class TrafficEmissionVisualizer:
    def __init__(self):
        print("🎬 Đang khởi động Xưởng Render Video Giao thông Động lực học...")
        
        # 1. Khai báo đường dẫn
        self.cube_path = os.path.join(config.BASE_DIR, 'outputs', 'history_C_cube.npy')
        self.csv_path = os.path.join(config.BASE_DIR, 'outputs', 'simulation_results.csv')
        self.w_path = os.path.join(config.BASE_DIR, 'outputs', 'W_spatial.npy')
        self.vid_dir = os.path.join(config.BASE_DIR, 'outputs', 'videos')
        os.makedirs(self.vid_dir, exist_ok=True)
        
        # 2. Tải dữ liệu
        if not os.path.exists(self.cube_path):
            raise FileNotFoundError("❌ Không tìm thấy history_C_cube.npy! Hãy chạy main_pipeline.py trước.")
            
        self.C_cube = np.load(self.cube_path)
        self.df = pd.read_csv(self.csv_path, index_col='time', parse_dates=True)
        
        # Tải bộ khung xương bản đồ giao thông (để vẽ nét đứt mờ)
        if os.path.exists(self.w_path):
            self.W_spatial = np.load(self.w_path)
        else:
            self.W_spatial = None
            print("⚠️ Không tìm thấy W_spatial.npy. Sẽ chỉ hiển thị nồng độ bụi.")

    def get_traffic_status(self, hour):
        """ Xác định trạng thái nhịp sinh học giao thông để hiển thị Text """
        if 7 <= hour <= 9:
            return "🚗 KẸT XE SÁNG (Cao điểm)", "#ff3333" # Đỏ rực
        elif 17 <= hour <= 19:
            return "🚙 KẸT XE CHIỀU (Cao điểm)", "#ff3333" # Đỏ rực
        elif 22 <= hour or hour <= 4:
            return "🌙 ĐÊM KHUYA (Đường vắng)", "#00ccff" # Xanh dương nhạt
        elif 11 <= hour <= 13:
            return "🍱 GIỜ NGHỈ TRƯA (Ổn định)", "#ffcc00" # Vàng
        else:
            return "🚕 GIAO THÔNG BÌNH THƯỜNG", "#33cc33" # Xanh lá

    def render_video(self, start_idx=0, num_frames=72):
        print(f"⚙️ Đang tiến hành Render {num_frames} khung hình (frames)...")
        
        # Kích hoạt Dark Mode cho biểu đồ
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.set_facecolor('black')
        
        # Lấy giới hạn nồng độ bụi để scale màu (loại bỏ nhiễu cực đại)
        vmax = np.percentile(self.C_cube[start_idx:start_idx+num_frames], 99.5)
        # Nếu nồng độ quá nhỏ, ép vmax lên một chút để màu không bị bệt
        vmax = max(vmax, 1.0) 

        # --- LỚP 1: VẼ KHUNG XƯƠNG GIAO THÔNG (MAP SKELETON) ---
        if self.W_spatial is not None:
            # Che các ô không có đường đi (bằng 0) để chúng trong suốt
            w_masked = np.ma.masked_where(self.W_spatial == 0, self.W_spatial)
            # Vẽ mạng lưới đường bằng màu trắng mờ (alpha=0.25)
            ax.imshow(w_masked, cmap='gray', alpha=0.25, origin='lower', extent=[0, config.N_X, 0, config.N_Y])

        # --- LỚP 2: VẼ DUNG NHAM NỒNG ĐỘ BỤI (EMISSION HEATMAP) ---
        # Sử dụng cmap='inferno' (Đen -> Tím -> Đỏ -> Cam -> Vàng sáng)
        cax = ax.imshow(self.C_cube[start_idx], cmap='inferno', origin='lower', 
                        vmin=0, vmax=vmax, alpha=0.85, extent=[0, config.N_X, 0, config.N_Y])
        
        # Thêm lưới (Grid) mờ để dễ xác định không gian
        ax.grid(color='white', linestyle=':', linewidth=0.5, alpha=0.2)
        
        # --- LỚP 3: CÁC THÀNH PHẦN CHÚ THÍCH (TEXT & QUIVER) ---
        title = ax.set_title('', color='white', fontsize=14, fontweight='bold', pad=20)
        
        # Hộp hiển thị trạng thái kẹt xe
        status_box = ax.text(0.03, 0.96, '', transform=ax.transAxes, fontsize=13, fontweight='bold',
                             bbox=dict(facecolor='black', alpha=0.7, edgecolor='white', boxstyle='round,pad=0.5'))
        
        # Hộp hiển thị Thông tin Khí tượng
        weather_box = ax.text(0.03, 0.90, '', transform=ax.transAxes, color='white', fontsize=11,
                              bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
                              
        # Vector gió đại diện (Góc trên bên phải)
        quiver = ax.quiver(config.N_X * 0.9, config.N_Y * 0.9, 0, 0, color='cyan', scale=50, width=0.01)
        ax.text(config.N_X * 0.9, config.N_Y * 0.95, 'Hướng gió', color='cyan', fontsize=10, ha='center')

        # Thanh màu (Colorbar)
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Nồng độ $PM_{2.5}$ nguyên bản PDE ($\mu g/m^3$)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

        def update(frame):
            t = start_idx + frame
            row = self.df.iloc[t]
            current_time = self.df.index[t]
            
            # 1. Cập nhật Đám mây Bụi
            cax.set_data(self.C_cube[t])
            
            # 2. Cập nhật Trạng thái Giao thông
            status_text, status_color = self.get_traffic_status(current_time.hour)
            status_box.set_text(status_text)
            status_box.set_color(status_color)
            
            # 3. Cập nhật Khí tượng
            rain_str = f"Mưa: {row['precipitation']} mm" if row['precipitation'] > 0 else "Trời tạnh"
            weather_box.set_text(f"Gió: {row['wind_speed']} km/h ({row['wind_direction']}°)\n"
                                 f"BLH: {row['blh']} m | {rain_str}")
            
            # 4. Cập nhật Vector gió
            u, v = wind_to_uv(row['wind_speed'], row['wind_direction'])
            # Chuẩn hóa độ dài mũi tên, nhưng giữ nguyên hướng
            norm = np.sqrt(u**2 + v**2) if (u**2 + v**2) > 0 else 1.0
            quiver.set_UVC(u/norm * 5, v/norm * 5)
            
            # 5. Cập nhật Tiêu đề Thời gian
            title.set_text(f"BẢN ĐỒ DÒNG CHẢY Ô NHIỄM GIAO THÔNG\n{current_time.strftime('%A, %d/%m/%Y - %H:00')}")
            
            return cax, status_box, weather_box, title, quiver

        ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=False)
        
        out_path = os.path.join(self.vid_dir, 'traffic_emission_dynamics.gif')
        ani.save(out_path, writer='pillow', fps=5) # 6 FPS giúp quan sát rõ sự nhấp nháy của dòng xe
        plt.close()
        print(f"\n✅ ĐÃ XUẤT VIDEO THÀNH CÔNG TẠI: {out_path}")
        print("💡 Hãy mở file ảnh động (GIF) này lên. Các con đường sẽ phát sáng như nham thạch!")

if __name__ == "__main__":
    viz = TrafficEmissionVisualizer()
    # Chạy mô phỏng 72 giờ (3 ngày) để thấy rõ 3 chu kỳ ngày - đêm lặp lại
    viz.render_video(start_idx=117, num_frames=72)