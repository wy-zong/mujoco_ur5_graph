from setuptools import setup, find_packages

setup(
    name="imitation_learning_lerobot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # 在這裡列依賴，例如：
        # "numpy",
        # "torch",
        "spatialmath-python",
        "mujoco",
        "roboticstoolbox-python",
        "numpy>=1.26.4",
        "modern-robotics",
        "placo==0.9.14",
        "transformers==4.51.3",
        "num2words",
        "accelerate",
        "loop_rate_limiters",
        "h5py",
        "joycon-python",
        "hidapi",
        "PyGLM"

    
    ],
)