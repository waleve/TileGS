import argparse
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData

def analyze_opacity(ply_path, bins=100):
    # 读取 PLY 文件
    plydata = PlyData.read(ply_path)
    if "opacity" not in plydata.elements[0].data.dtype.names:
        print(f"[Error] 'opacity' field not found in {ply_path}")
        return

    # 提取 raw opacity
    opacities = np.asarray(plydata.elements[0]["opacity"]).astype(np.float32)
    
    # 判断是否需要 sigmoid
    if opacities.min() < 0 or opacities.max() > 1:
        print("[WARN] Opacity values exceed [0, 1], applying sigmoid...")
        opacities = 1 / (1 + np.exp(-opacities))

    # 输出统计信息
    print(f"[INFO] Loaded {ply_path}")
    print(f"[STATS] Number of Gaussians: {len(opacities)}")
    print(f"[STATS] Opacity mean: {opacities.mean():.6f}")
    print(f"[STATS] Opacity min: {opacities.min():.6f}, max: {opacities.max():.6f}")
    print(f"[STATS] Opacity < 0.01: {(opacities < 0.01).sum()} ({(opacities < 0.01).mean() * 100:.2f}%)")
    print(f"[STATS] Opacity < 0.05: {(opacities < 0.05).sum()} ({(opacities < 0.05).mean() * 100:.2f}%)")

    # 绘制直方图
    plt.figure(figsize=(8, 5))
    plt.hist(opacities, bins=bins, color='steelblue', edgecolor='black')
    plt.title('Opacity Distribution')
    plt.xlabel('Opacity')
    plt.ylabel('Number of Gaussians')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze opacity distribution from a PLY file.")
    parser.add_argument("ply_path", type=str, help="Path to the PLY file (.ply)")
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins")
    args = parser.parse_args()

    analyze_opacity(args.ply_path, bins=args.bins)
