import matplotlib.pyplot as plt
import numpy as np

def load_and_plot_depth(npz_file):
    data = np.load(npz_file)
    
    depth_image = data['arr_0']
    
    plt.figure(figsize=(6, 6))
    plt.imshow(depth_image, cmap="viridis")  # 使用伪彩色
    plt.colorbar(label="Depth Value")  # 颜色条
    plt.title(f"Depth Image from {npz_file}")
    plt.axis("off")
    plt.savefig("depth_test.png")

sample_npz = "./depth_images/episode_3871trajectory_2554_sample0.npz"
load_and_plot_depth(sample_npz)
