import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 定义标签和数据
labels = ['echinus', 'fish', 'starfish', 'holothurian', 'scallop', 'crab', 'plastic trash', 'diver',
          'cuttlefish', 'rov', 'turtle', 'jellyfish', 'metal trash', 'shrimp', 'wood trash', 'fabric trash',
          'fishing trash', 'paper trash', 'rubber trash']
sizes = [99732, 60108, 45345, 26545, 18295, 13337, 8326, 6395, 5857, 5770, 4123, 3315, 1126, 653, 340, 314, 196, 193,
         142]

# 使用tab20颜色映射，并为超出10个的颜色循环使用颜色
n_colors = len(labels)
cmap = cm.get_cmap('tab20')
colors = [cmap(i % 20) for i in range(n_colors)]  # 使用tab20的完整颜色范围

# 设置图的大小
plt.figure(figsize=(14, 8))

# 创建柱状图，并指定每个柱子的颜色
bars = plt.bar(labels, sizes, color=colors)


# 为每个柱子添加数据标签
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # 避免在高度为0的柱子上添加标签
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                     '{}'.format(height),
                     ha='center', va='bottom', fontsize=10)


autolabel(bars)

# 添加标题和坐标轴标签
plt.title('Category structure of the Poseidon-300K')
plt.xlabel('Categories')
plt.ylabel('Number')

# 旋转x轴标签以便于阅读
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

# 优化布局以避免标签被裁剪
plt.tight_layout()

# 显示图形
plt.show()
plt.savefig('/opt/data/private/fcf/mmdetection/tools/plot/columnmap.jpg', dpi=300)