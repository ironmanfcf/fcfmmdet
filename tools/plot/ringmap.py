import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 定义标签和值
labels = ['echinus', 'fish', 'starfish', 'holothurian', 'scallop', 'crab', 'plastic trash', 'diver',
          'cuttlefish', 'rov', 'turtle', 'jellyfish', 'metal trash', 'shrimp', 'wood trash', 'fabric trash',
          'fishing trash', 'paper trash', 'rubber trash']  # 你的标签列表
sizes = [99732, 60108, 45345, 26545, 18295, 13337, 8326, 6395, 5857, 5770, 4123, 3315, 1126, 653, 340, 314, 196, 193, 142]  # 你的值列表

#映射颜色列表
# 创建一个颜色映射实例，这里我们使用'viridis'（一个适用于科学数据的颜色映射）
cmap = plt.cm.get_cmap('tab20')

# 计算每个柱子的颜色（假设我们想要使用颜色映射的完整范围）
colors = cmap(np.linspace(0, 1, len(sizes)))

# 计算百分比
sizes_percent = [s / sum(sizes) * 100 for s in sizes]

# 创建一个explode列表（这里使用相同值来避免留白，但你也可以根据需要调整）
explode = [0.0001 for _ in sizes]  # 所有扇区都使用相同的explode值来减少留白

# 创建饼图，但不显示标签
fig1, ax1 = plt.subplots(figsize=(14, 6))
#patches, texts, autotexts = ax1.pie(sizes, autopct='%1.2f%%', startangle=90,
patches, texts, autotexts = ax1.pie(sizes, autopct=lambda p: '', startangle=90,
                                    colors=colors,
                                    explode=explode,
                                    wedgeprops=dict(width=0.6, edgecolor='w'))
for autotext in autotexts:
    autotext.set_fontsize(9.5)

# 设置轴为等比例
ax1.axis('equal')

# 去除x轴和y轴的标签和刻度
ax1.set_xticks([])
ax1.set_yticks([])

# 手动设置百分比标签，仅当扇区大小大于或等于阈值时
# for i, txt in enumerate(autotexts):
#     if sizes[i] >= 5000:  # 阈值设置为500
#         txt.set_text(f'{sizes_percent[i]:.1f}%')
#     else:
#         txt.set_text('')  # 不显示百分比
# 定义阈值
threshold = 10000

# 手动设置标签和百分比
for i, p in enumerate(patches):
    # 计算标签位置
    x, y = p.center
    # 设置自定义标签
    if sizes[i] >= threshold:
        autotexts[i].set_text(f'  {labels[i]}\n  {sizes_percent[i]:.1f}%')  # 设置百分比

# 创建一个图例，模拟饼图的每个扇区
labels_with_percent = [f'{label} ({percent:.2f}%)' for label, percent in zip(labels, sizes_percent)]
legend_elements = [Patch(facecolor=colors[i], label=label)
                   for i, label in enumerate(labels_with_percent)]

# 添加图例，并调整位置以免超出画面范围
# 这里bbox_to_anchor的x值是一个负数，意味着图例会向左移动
ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.1, 0.6), borderaxespad=0.,
           ncol=2)  # 设置为两列

# 显示图形
# 添加标题和坐标轴标签
plt.title('Category structure of the Poseidon-100K')
plt.show()