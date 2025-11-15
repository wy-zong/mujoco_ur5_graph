# debug_env_registry.py
from imitation_learning_lerobot.envs import EnvFactory
from imitation_learning_lerobot.envs import Env

print("Leaf subclasses of Env:")
for sub in Env.__subclasses__():
    print("  -", sub.__name__, "name attr =", getattr(sub, "name", None), " _name =", getattr(sub, "_name", None))

print("\nEnv registry keys (after register_all via __init__.py import):")
print(list(EnvFactory._strategies.keys()))

print("\nPick a class by key:")
env_cls = EnvFactory.get_strategies("grasp_cloth")
print("grasp_cloth ->", env_cls)