
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("/home/wuc120/imitation_learning_lerobot/imitation_learning_lerobot/assets/scenes/flag.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)  # 以最簡單方式前進模擬
