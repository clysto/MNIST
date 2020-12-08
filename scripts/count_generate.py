from os import path
import glob

base_fp = "data/generate"
for i in range(10):
    fps = glob.glob(path.join(base_fp, str(i), "*.jpg"))
    print(f"{i}: {len(fps)}")
