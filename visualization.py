# File: src/visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import os

from src import config
from src.physics_engine import wind_to_uv

class AdvancedVisualizer:
    def __init__(self):
        print("🎨 Đang khởi động Xưởng Trực quan hóa Đa chiều...")
        
        # 1. Đảm bảo các thư mục đầu ra tồn tại
        self.plot_dir = os.path.join(config.BASE_DIR, 'outputs', 'plots')
        self.vid_dir = os.path.join(config.BASE_DIR, 'outputs', 'videos')
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.vid_dir, exist_ok=True)
        
        # 2. Tải khối dữ liệu 3D và file Khí tượng
        cube_path = os.path.join(config.BASE_DIR, 'outputs', 'history_C_cube.npy')
        csv_path = os.path.join(config.BASE_DIR, 'outputs', 'simulation_results.csv')
        
        if not os.path.exists(cube_path) or not os.path.exists(csv_path):
            raise FileNotFoundError("❌ Chưa có dữ liệu! Hãy chạy main_pipeline.py trước.")
            
        self.C_cube = np.load(cube_path)
        self.df = pd.read_csv(csv_path, index_col='time', parse_dates=True)
        
        # Thiết lập dải màu Cảnh báo Ô nhiễm (Xanh lá -> Vàng -> Đỏ -> Tím)
        colors = [(0, 'green'), (0.2, 'yellow'), (0.5, 'orange'), (0.8, 'red'), (1, 'purple')]
        self.aqi_cmap = LinearSegmentedColormap.from_list("AQI", colors)

    def plot_3d_surface(self, t_idx, title_suffix=""):
        """
        [Tính năng 5] Mô hình Khối 3D: Góc nhìn không gian 3 chiều về ngọn núi bụi.
        """
        C_matrix = self.C_cube[t_idx]
        time_label = self.df.index[t_idx].strftime('%d/%m/%Y %H:00')
        
        fig = plt.figure(figsize=(12, 8), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(np.arange(config.N_X), np.arange(config.N_Y))
        
        # Vẽ bề mặt 3D
        surf = ax.plot_surface(X, Y, C_matrix, cmap=self.aqi_cmap, 
                               edgecolor='none', alpha=0.9, vmin=0, vmax=100)
        
        ax.set_title(f'Bản đồ Địa hình Ô nhiễm $PM_{{2.5}}$ - {time_label}\n{title_suffix}', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Trục X (Lưới)')
        ax.set_ylabel('Trục Y (Lưới)')
        ax.set_zlabel('Nồng độ ($\mu g/m^3$)')
        
        # Giới hạn trục Z để dễ so sánh giữa các giờ
        ax.set_zlim(0, max(100, np.max(C_matrix)))
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Nồng độ $PM_{2.5}$')
        
        out_path = os.path.join(self.plot_dir, f'3d_surface_t{t_idx}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã xuất mô hình 3D tại: {out_path}")

    def plot_cross_section(self, t_idx):
        """
        [Tính năng 4] Biểu đồ Cắt ngang Phân phối (Đường cong Gauss bị xô lệch).
        Cắt ngang ma trận tại vị trí Y đi qua Trạm quan trắc.
        """
        C_matrix = self.C_cube[t_idx]
        time_label = self.df.index[t_idx].strftime('%d/%m/%Y %H:00')
        wind_dir = self.df.iloc[t_idx]['wind_direction']
        
        # Lấy một lát cắt ngang (Row) đi qua tâm Trạm đo
        cross_section = C_matrix[config.OBS_I, :]
        
        plt.figure(figsize=(10, 5), facecolor='white')
        
        # Tô màu gradient dưới đường cong
        x_vals = np.arange(config.N_X)
        plt.fill_between(x_vals, cross_section, color='orange', alpha=0.3)
        plt.plot(x_vals, cross_section, color='red', linewidth=2)
        
        # Đánh dấu vị trí trạm đo
        plt.axvline(x=config.OBS_J, color='blue', linestyle='--', label='Vị trí Trạm/Trường học')
        
        plt.title(f'Lát cắt Không gian Nồng độ $PM_{{2.5}}$ ngang qua Trạm (Giờ: {time_label})\nHướng gió đẩy: {wind_dir}°', 
                  fontsize=12, fontweight='bold')
        plt.xlabel('Tọa độ X (Biên trái -> Biên phải)')
        plt.ylabel('Nồng độ $\mu g/m^3$')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
        out_path = os.path.join(self.plot_dir, f'cross_section_t{t_idx}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã xuất Lát cắt 1D tại: {out_path}")

    # Mở file src/visualization.py và thêm hàm này vào bên trong class AdvancedVisualizer:

    def plot_time_series_comparison(self, start_idx=0, end_idx=None):
        """
        [Tính năng 6] Biểu đồ Chuỗi thời gian: Đối chiếu Nồng độ Thực tế và Dự báo.
        Tích hợp dải màu AQI và đánh dấu vùng loại trừ Spin-up.
        """
        if end_idx is None:
            end_idx = len(self.df)
            
        plot_df = self.df.iloc[start_idx:end_idx]
        spinup_hours = getattr(config, 'SPINUP_HOURS', 117)
        
        plt.figure(figsize=(15, 6), facecolor='white')
        
        # --- 1. VẼ DẢI MÀU CẢNH BÁO AQI (Tham khảo chuẩn EPA) ---
        max_y = max(100, plot_df['simulated_pm25'].max() + 20)
        plt.axhspan(0, 15, facecolor='#00e400', alpha=0.15, label='Tốt (0-15)')
        plt.axhspan(15, 35, facecolor='#ffff00', alpha=0.15, label='Trung bình (15-35)')
        plt.axhspan(35, 55, facecolor='#ff7e00', alpha=0.15, label='Kém (35-55)')
        plt.axhspan(55, max_y, facecolor='#ff0000', alpha=0.1, label='Xấu (>55)')
        
        # --- 2. VẼ ĐƯỜNG DỮ LIỆU ĐỘNG LỰC HỌC ---
        plt.plot(plot_df.index, plot_df['pm25'], label='Thực tế (Trạm quan trắc)', 
                 color='black', marker='.', linestyle='-', markersize=5, alpha=0.7)
        plt.plot(plot_df.index, plot_df['simulated_pm25'], label='Dự báo PDE', 
                 color='blue', linewidth=2, alpha=0.8)
        
        # --- 3. ĐÁNH DẤU VÙNG KHỞI ĐỘNG LẠNH (SPIN-UP) ---
        if start_idx < spinup_hours and len(plot_df) > spinup_hours:
            spinup_end_time = self.df.index[spinup_hours]
            # Tô xám vùng dữ liệu không được tính vào báo cáo sai số
            plt.axvspan(plot_df.index[0], spinup_end_time, color='gray', alpha=0.4, label='Vùng Khởi động (Spin-up)')
            plt.axvline(x=spinup_end_time, color='black', linestyle='--', linewidth=1.5)
            plt.text(spinup_end_time, max_y * 0.9, '  Hết Spin-up\n  (Bắt đầu tính điểm)', 
                     color='black', fontweight='bold', fontsize=10)
            
        plt.title('Đối chiếu Chuỗi Thời Gian $PM_{2.5}$ Thực tế và Mô hình Động lực học', 
                  fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Thời gian mô phỏng', fontsize=12)
        plt.ylabel('Nồng độ $PM_{2.5}$ ($\mu g/m^3$)', fontsize=12)
        
        # Tối ưu hóa Legend (Đẩy ra ngoài để không che mất đồ thị)
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Nghiêng chữ ngày tháng trục X cho dễ đọc
        plt.gcf().autofmt_xdate()
        
        out_path = os.path.join(self.plot_dir, 'time_series_comparison.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã xuất Biểu đồ Chuỗi thời gian tại: {out_path}")

    def generate_heatmap_animation(self, start_idx=0, num_frames=48):
        """
        [Tính năng 1, 2, 3] Đóng gói trọn bộ: Heatmap + Quiver Gió (Động) + Mưa rửa trôi nhiều cấp + BLH
        """
        print(f"🎬 Đang render Video Animation ({num_frames} frames)... Vui lòng đợi...")
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        
        # --- BẬT LƯỚI TỌA ĐỘ (GRID) ---
        # Đánh dấu tick mỗi 10 ô lưới (tương đương 200m mỗi ô lớn nếu dx=20m)
        ax.set_xticks(np.arange(0, config.N_X + 1, 10))
        ax.set_yticks(np.arange(0, config.N_Y + 1, 10))
        ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.4)
        
        # --- TẠO KHUNG VECTOR GIÓ ---
        Y, X = np.mgrid[0:config.N_Y, 0:config.N_X]
        step = 8 # Cứ 8 ô lưới vẽ 1 mũi tên gió để mật độ vừa phải
        
        vmax = np.percentile(self.C_cube[start_idx:start_idx+num_frames], 98)
        
        cax = ax.imshow(self.C_cube[start_idx], cmap=self.aqi_cmap, origin='lower', 
                        vmin=0, vmax=max(50, vmax))
        
        # Đánh dấu Trạm đo
        ax.plot(config.OBS_J, config.OBS_I, marker='*', color='cyan', markersize=14, markeredgecolor='black', label='Trạm đo / Trường học')
        
        # Thiết lập Quiver: Scale càng nhỏ mũi tên càng dài. Cố định scale=250 để thể hiện độ chênh lệch gió
        quiver = ax.quiver(X[::step, ::step], Y[::step, ::step], 
                           np.zeros_like(X[::step, ::step]), np.zeros_like(Y[::step, ::step]), 
                           color='white', alpha=0.9, scale=250, width=0.003, headwidth=4)
                           
        title = ax.set_title('', fontsize=13, fontweight='bold', pad=15)
        # Nhãn hiệu ứng mưa
        rain_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='cyan', fontsize=13, fontweight='bold', 
                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))
        
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label='Nồng độ $PM_{2.5}$ ($\mu g/m^3$)')
        ax.legend(loc='lower right')

        def update(frame):
            t = start_idx + frame
            row = self.df.iloc[t]
            time_str = self.df.index[t].strftime('%d/%m/%Y %H:00')
            
            # 1. Cập nhật Nồng độ Bụi
            cax.set_data(self.C_cube[t])
            
            # 2. Cập nhật Gió (Vector Động)
            speed = row['wind_speed']
            u, v = wind_to_uv(speed, row['wind_direction'])
            
            if speed > 0:
                norm = np.sqrt(u**2 + v**2)
                # Tỷ lệ thuận độ dài mũi tên với sức gió (speed)
                u_dyn, v_dyn = (u/norm) * speed, (v/norm) * speed 
            else:
                u_dyn, v_dyn = 0, 0
                
            U_matrix = np.full_like(X[::step, ::step], u_dyn, dtype=float)
            V_matrix = np.full_like(Y[::step, ::step], v_dyn, dtype=float)
            quiver.set_UVC(U_matrix, V_matrix)
            
            # 3. Cập nhật Hiệu ứng Mưa (Nhiều cấp độ)
            rain = row['precipitation']
            if rain > 0:
                if rain < 2.5: # Mưa nhỏ
                    rain_status = "🌦️ MƯA NHỎ"
                    bg_color = '#e6f2ff' # Xanh nhạt
                    rain_text.set_color('#80bfff')
                elif rain <= 7.6: # Mưa vừa
                    rain_status = "🌧️ MƯA VỪA"
                    bg_color = '#b3d9ff' # Xanh dương
                    rain_text.set_color('#00ace6')
                else: # Mưa to
                    rain_status = "⛈️ MƯA TO"
                    bg_color = '#66b3ff' # Xanh đậm, u ám
                    rain_text.set_color('#00ffcc')
                    
                rain_text.set_text(f'{rain_status}: {rain:.1f} mm')
                rain_text.set_visible(True)
                fig.patch.set_facecolor(bg_color) 
            else:
                rain_text.set_visible(False)
                fig.patch.set_facecolor('white')
            
            # 4. Cập nhật Tiêu đề có BLH
            title.set_text(f'Động lực học $PM_{{2.5}}$ TP.HCM | Lưới {config.N_X}x{config.N_Y} | Trạm: Tân Phú\n'
                           f'⏰ {time_str} | 💨 Gió: {speed:.1f} km/h ({row["wind_direction"]}°) | ☁️ Trần BLH: {row["blh"]}m')
            
            return cax, quiver, title, rain_text

        ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=False)
        
        out_path = os.path.join(self.vid_dir, 'pollution_dynamics.gif')
        ani.save(out_path, writer='pillow', fps=5) # 5 khung hình / giây cho mượt
        plt.close()
        print(f"🎞️ Đã xuất Video Động lực học Tích hợp tại: {out_path}")

if __name__ == "__main__":
    viz = AdvancedVisualizer()
    
# 1. Vẽ chuỗi thời gian (Lấy ví dụ 500 giờ đầu để nhìn rõ vùng Spin-up)
    viz.plot_time_series_comparison(start_idx=0, end_idx=200)
    
    # 2. Vẽ cắt ngang và 3D tại một thời điểm Ô nhiễm cao (Giả sử giờ thứ 200)
 # viz.plot_cross_section(t_idx=500)
 # viz.plot_3d_surface(t_idx=500, title_suffix="Điểm bùng phát khí thải")
    
    # 2. Render Video GIF 48 giờ liên tục (2 ngày)
    # Giúp quan sát chu kỳ ngày-đêm và gió đổi hướng
 # viz.generate_heatmap_animation(start_idx=400,num_frames=48)