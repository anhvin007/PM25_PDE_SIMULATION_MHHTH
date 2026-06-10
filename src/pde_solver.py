# File: src/pde_solver.py
import numpy as np
from src import config
from src.physics_engine import generate_nudging_kernel, wind_to_uv, get_pg_diffusivity, get_washout_coeff

class PDESolver:
    def __init__(self):
        # Khởi tạo ma trận nồng độ C_pred toàn số 0 ban đầu
        self.C = np.zeros((config.N_Y, config.N_X), dtype=np.float32)
        # Load sẵn ma trận Gaussian Kernel để dùng cho Đồng hóa dữ liệu
        self.nudging_kernel = generate_nudging_kernel()
        
    def step(self, dt_hour, u, v, D, Lambda_rain, S_matrix, obs_pm25):
        """
        Thực thi 1 giờ mô phỏng (Có tích hợp băm nhỏ CFL Micro-stepping)
        """
        dx, dy = config.DX, config.DY
        
        # 1. Kiểm tra Điều kiện ổn định CFL
        denominator = (abs(u)/dx) + (abs(v)/dy) + 2*D*((1/dx**2) + (1/dy**2))
        dt_max = 1.0 / denominator if denominator > 0 else 1.0
        
        # Quyết định phân mảnh bước thời gian (Micro-stepping)
        if dt_max >= dt_hour:
            M = 1
        else:
            M = int(np.ceil(dt_hour / dt_max))
        
        dt_sub = dt_hour / M
        
        # 2. Vòng lặp Vi mô (Micro-loop) giải phương trình PDE
        for _ in range(M):
            # Tính các ma trận dịch chuyển (Shifting) để làm sai phân
            # np.roll dịch ma trận đi 1 ô. Sau đó ép điều kiện biên Neumann (Đạo hàm biên = 0)
            C_left = np.roll(self.C, 1, axis=1);  C_left[:, 0] = self.C[:, 0]
            C_right = np.roll(self.C, -1, axis=1); C_right[:, -1] = self.C[:, -1]
            C_down = np.roll(self.C, 1, axis=0);   C_down[0, :] = self.C[0, :]
            C_up = np.roll(self.C, -1, axis=0);    C_up[-1, :] = self.C[-1, :]
            
            # --- TOÁN TỬ ĐỐI LƯU (Advection - Upwind Scheme) ---
            adv_x = np.maximum(u, 0) * (self.C - C_left)/dx + np.minimum(u, 0) * (C_right - self.C)/dx
            adv_y = np.maximum(v, 0) * (self.C - C_down)/dy + np.minimum(v, 0) * (C_up - self.C)/dy
            
            # --- TOÁN TỬ KHUẾCH TÁN (Diffusion - Central Difference) ---
            diff_x = (C_right - 2*self.C + C_left) / (dx**2)
            diff_y = (C_up - 2*self.C + C_down) / (dy**2)
            
            # --- CẬP NHẬT TRẠNG THÁI PDE (Euler Forward) ---
            # dC/dt = - Advection + Diffusion + Source - Sink
            delta_C = - (adv_x + adv_y) + D * (diff_x + diff_y) + S_matrix - (Lambda_rain * self.C) - (config.DRY_DEPOSITION * self.C)
            self.C = self.C + dt_sub * delta_C
            
            # Rào cản vật lý: Nồng độ không thể âm
            self.C = np.maximum(self.C, 0.0)

        # 3. ĐỒNG HÓA DỮ LIỆU (Data Assimilation / Nudging)
        # Chỉ thực hiện 1 lần vào cuối khung giờ (sau khi cộng dồn M vi bước)
        if not np.isnan(obs_pm25):
            # Tính sai số Innovation tại trạm đo
            epsilon = obs_pm25 - self.C[config.OBS_I, config.OBS_J]
            # Áp dụng Lực nắn Newtonian (Parametric Kalman Filter)
            self.C = self.C + config.G_NUDGE * epsilon * self.nudging_kernel
            
        return self.C.copy()

# Khối Test Nhanh Cơ Chế CFL
if __name__ == "__main__":
    solver = PDESolver()
    
    # Giả lập: Gió mạnh 15km/h (Gió Đông Bắc), Ban ngày Nắng gắt (Class A -> D rất lớn)
    u_test, v_test = wind_to_uv(15.0, 45) 
    D_test = get_pg_diffusivity(15.0, 10.0, 12)
    
    # Chạy thử 1 step
    S_dummy = np.full((100, 100), 10.0) # Hàm xả thải giả định
    C_out = solver.step(dt_hour=1.0, u=u_test, v=v_test, D=D_test, 
                        Lambda_rain=0, S_matrix=S_dummy, obs_pm25=30.0)
    
    print(f"✅ Động cơ PDE biên dịch thành công!")
    print(f"Trạng thái tâm ma trận sau 1 giờ: {C_out[50, 50]:.2f} µg/m³")