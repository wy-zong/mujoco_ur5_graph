# import numpy as np
# import roboticstoolbox as rtb
# from spatialmath import SE3
# import modern_robotics as mr

# from ..utils import MathUtils
# from .robot import Robot, get_transformation_mdh, wrap

# class SO101(Robot):
#     def __init__(self) -> None:
#         super().__init__()

#         # 修正：定義為 6 自由度 (5 Arm Joints + 1 Gripper Joint)
#         self._dof = 6
#         self.q0 = [0.0] * 6

#         # 1. 定義 DH 參數 (來源: argo-robot/scripts/model.py)
#         # 這裡只定義前 5 軸的運動學參數，第 6 軸(夾爪)不參與手臂 IK 解算
#         # 注意：RTB 模型我們建立 5 軸，但在 Robot 類別中我們管理 6 個數據
        
#         # Standard DH 參數 (alpha, a, d, theta)
#         self.alpha_array = [np.pi/2, 0.0, 0.0, -np.pi/2, 0.0]
#         self.a_array     = [0.0304, 0.116, 0.1347, 0.0, 0.0]
#         self.d_array     = [0.0542, 0.0, 0.0, 0.0, 0.0609]
#         self.theta_array = [0.0, 0.0, 0.0, 0.0, 0.0]
        
#         # 為了父類別計算相容性，補齊第 6 軸的虛擬參數 (夾爪)
#         self.alpha_array.append(0.0)
#         self.a_array.append(0.0)
#         self.d_array.append(0.0)
#         self.theta_array.append(0.0)
        
#         self.sigma_array = [0] * 6

#         # 2. 定義動力學參數 (來源: so101.urdf)
#         # 這裡填入 6 個馬達對應的連桿質量
#         m1 = 0.147   # base
#         m2 = 0.100   # shoulder
#         m3 = 0.103   # upper_arm
#         m4 = 0.104   # lower_arm
#         m5 = 0.079   # wrist
#         m6 = 0.087 + 0.012 # gripper_link + moving_jaw (近似)

#         ms = [m1, m2, m3, m4, m5, m6]
        
#         # 質心 (COM) 與 慣性 (Inertia) - 簡化範例
#         rs = [np.zeros(3) for _ in range(6)] 
#         Is = [np.eye(3) * 0.001 for _ in range(6)]
#         Jms = [0.01] * 6

#         # 3. 建立 Roboticstoolbox 模型 (只用於 IK 解算，所以建立 5 軸)
#         links = []
#         for i in range(5):
#             links.append(rtb.DHLink(d=self.d_array[i], alpha=self.alpha_array[i], a=self.a_array[i], 
#                                     offset=self.theta_array[i], mdh=False, 
#                                     m=ms[i], r=rs[i], I=Is[i], Jm=Jms[i], G=1.0))
        
#         # 這是用於運動學解算的內部模型 (5-DOF)
#         self.kinematics_robot = rtb.DHRobot(links, name="SO101_Arm")
        
#         # self.robot 用於父類別的某些功能，保持與 kinematics_robot 一致或包含夾爪視需求而定
#         # 在此專案架構下，通常 self.robot 用於 fkine/ikine
#         self.robot = self.kinematics_robot

#         # 4. 初始化父類別所需的動力學矩陣 (6 軸)
#         T = SE3()
#         for i in range(self._dof):
#             Ti = get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
#                                         self.sigma_array[i], 0.0)
#             self._Ms.append(Ti.A)
#             T = T * Ti
#             self._Ses.append(np.hstack((T.a, np.cross(T.t, T.a))))

#             Gm = np.zeros((6, 6))
#             Gm[:3, :3] = Is[i]
#             Gm[3:, 3:] = ms[i] * np.eye(3)
#             AdT = mr.Adjoint(mr.RpToTrans(np.eye(3), -rs[i]))
#             self._Gs.append(AdT.T @ Gm @ AdT)
#             self._Jms.append(Jms[i])

#         self._Ms.append(np.eye(4))

#     def fkine(self, q) -> SE3:
#         """
#         計算正運動學。
#         q: 長度為 6 的陣列 (包含夾爪)
#         """
#         # 只取前 5 個關節計算末端姿態
#         q_arm = q[:5]
#         return self.kinematics_robot.fkine(q_arm)

#     # def ikine(self, Twt: SE3) -> np.ndarray:
#     #     """
#     #     SO101 的逆運動學求解。
#     #     """
#     #     Tbe = self.cal_Tbe(Twt)
        
#     #     # 使用前 5 軸的當前角度作為猜測值
#     #     q_guess = self.q0[:5]
        
#     #     # 5 軸手臂無法完美達成 6D 姿態 (XYZ + RPY)，因此使用 mask 權重
#     #     # 優先保證位置 (X,Y,Z) 和 指向 (Pitch, Roll)，放寬 Yaw (繞 Z 軸旋轉)
#     #     # Mask: [x, y, z, rx, ry, rz]
#     #     sol = self.kinematics_robot.ikine_LM(Tbe, q0=q_guess, mask=[1, 1, 1, 1, 1, 0])
        
#     #     if sol.success:
#     #         # 回傳 6 軸數據：前 5 軸為 IK 解，第 6 軸維持當前夾爪狀態
#     #         return np.append(sol.q, self.q0[5])
#     #     else:
#     #         return np.array([])
        
#     def ikine(self, Twt: SE3) -> np.ndarray:
#         """
#         SO101 的逆運動學求解。
#         修正：直接傳入世界座標 Twt，讓 RTB 內部處理 Base Transform。
#         """
#         # 使用前 5 軸的當前角度作為猜測值
#         q_guess = self.q0[:5]
        
#         # [修正點] 
#         # 不要使用 self.cal_Tbe(Twt) 轉換成相對座標。
#         # 因為 self.kinematics_robot.base 已經在 Env 中透過 set_base 設定了世界座標。
#         # 直接傳入 Twt (Target in World Frame)，Solver 會自動計算 T_base_inv * Twt。
#         sol = self.kinematics_robot.ikine_LM(Twt, q0=q_guess, mask=[1, 1, 1, 1, 1, 0])
        
#         if sol.success:
#             # 回傳 6 軸數據：前 5 軸為 IK 解，第 6 軸維持當前夾爪狀態
#             return np.append(sol.q, self.q0[5])
#         else:
#             # [除錯用] 可以印出失敗訊息幫助定位
#             # print(f"[WARN] SO101 IK Failed. Target: {Twt.t}")
#             return np.array([])

#     def set_robot_config(self, q):
#         pass

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import modern_robotics as mr

from ..utils import MathUtils
from .robot import Robot, get_transformation_mdh, wrap

class SO101(Robot):
    def __init__(self) -> None:
        super().__init__()

        # 6 自由度: 5 Arm + 1 Gripper
        self._dof = 6
        self.q0 = [0.0] * 6

        # 1. DH 參數 (Standard DH)
        self.alpha_array = [np.pi/2, 0.0, 0.0, -np.pi/2, 0.0, 0.0]
        self.a_array     = [0.0304, 0.116, 0.1347, 0.0, 0.0, 0.0]
        self.d_array     = [0.0542, 0.0, 0.0, 0.0, 0.0609, 0.0]
        self.theta_array = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.sigma_array = [0] * 6

        # 2. 動力學參數
        ms = [0.147, 0.100, 0.103, 0.104, 0.079, 0.100]
        rs = [np.zeros(3) for _ in range(6)]
        Is = [np.eye(3) * 0.001 for _ in range(6)]
        Jms = [0.01] * 6

        # 3. 建立 RTB 模型 (前 5 軸用於 IK)
        links = []
        for i in range(5):
            links.append(rtb.DHLink(d=self.d_array[i], alpha=self.alpha_array[i], a=self.a_array[i], 
                                    offset=self.theta_array[i], mdh=False, 
                                    m=ms[i], r=rs[i], I=Is[i], Jm=Jms[i], G=1.0))
        
        self.kinematics_robot = rtb.DHRobot(links, name="SO101_Arm")
        self.robot = self.kinematics_robot

        # 4. 初始化父類別動力學矩陣
        T = SE3()
        for i in range(self._dof):
            Ti = get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                        self.sigma_array[i], 0.0)
            self._Ms.append(Ti.A)
            T = T * Ti
            self._Ses.append(np.hstack((T.a, np.cross(T.t, T.a))))

            Gm = np.zeros((6, 6))
            Gm[:3, :3] = Is[i]
            Gm[3:, 3:] = ms[i] * np.eye(3)
            AdT = mr.Adjoint(mr.RpToTrans(np.eye(3), -rs[i]))
            self._Gs.append(AdT.T @ Gm @ AdT)
            self._Jms.append(Jms[i])
        self._Ms.append(np.eye(4))

    def fkine(self, q) -> SE3:
        return self.kinematics_robot.fkine(q[:5])

    def ikine(self, Twt: SE3) -> np.ndarray:
        """
        使用微分運動學 (Jacobian DLS) 進行單步跟隨。
        這比 ikine_LM 快非常多，且在奇異點附近不會卡死，只會變慢。
        """
        # 取得當前 5 軸狀態
        q_curr = self.q0[:5]
        
        # 1. 計算當前末端姿態與誤差
        T_curr = self.kinematics_robot.fkine(q_curr)
        
        # 位置誤差 (World Frame)
        err_pos = Twt.t - T_curr.t
        
        # 姿態誤差 (使用旋轉矩陣差異近似角速度向量)
        # R_err = R_target * R_curr^T
        R_err = Twt.R @ T_curr.R.T
        
        # 轉成軸角 (Axis-Angle) 誤差向量
        tr = np.trace(R_err)
        angle = np.arccos(np.clip((tr - 1) / 2, -1, 1))
        if np.abs(angle) < 1e-6:
            err_rot = np.zeros(3)
        else:
            axis = np.array([R_err[2, 1] - R_err[1, 2],
                             R_err[0, 2] - R_err[2, 0],
                             R_err[1, 0] - R_err[0, 1]]) / (2 * np.sin(angle))
            err_rot = axis * angle

        # 組合空間誤差速度 v (6x1)
        # 這裡我們給姿態誤差較小的權重(0.3)，讓手臂優先滿足位置
        v = np.hstack([err_pos, err_rot * 0.3]) 

        # 2. 計算雅可比矩陣 (World Frame)
        J = self.kinematics_robot.jacob0(q_curr) # 6x5

        # 3. 求解關節速度 (Damped Least Squares)
        # dq = (J.T * J + lambda^2 * I)^-1 * J.T * v
        # 這是 argo-robot 專案中 RobotUtils.dls_right_pseudoinv 的核心邏輯
        dls_lambda = 0.1  # 阻尼係數，越大越穩定但越慢
        
        hessian = J.T @ J + (dls_lambda**2) * np.eye(5)
        gradient = J.T @ v
        dq = np.linalg.solve(hessian, gradient)

        # 4. 限制最大速度 (防暴衝)
        dq = np.clip(dq, -0.5, 0.5)

        # 5. 更新角度
        q_next = q_curr + dq
        
        # 回傳完整 6 軸 (前5軸更新 + 第6軸維持原樣)
        return np.append(q_next, self.q0[5])

    def set_robot_config(self, q):
        pass