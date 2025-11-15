import mujoco                                                                                                                                                  
m = mujoco.MjModel.from_xml_path("/home/wuc120/benchmarking_cloth/bcm/assets/mujoco3_cloth.xml")  # 或整個 scene.xml
print("nbody=", m.nbody)
for i in range(m.nbody):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and ("cloth" in name or "flex" in name):
        print(i, name)