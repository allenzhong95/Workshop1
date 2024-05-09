import numpy as np
import matplotlib.pyplot as plt


dt = [[94.7, 98.0, 61.7],
      [94.9, 98.3, 60.9],
      [95.6, 98.3, 68.3],
      [95.3, 97.5, 73.8],
      [94.6, 96.3, 77.4],
      [94.0, 95.3, 80.2],
      [93.5, 94.6, 82.3],
      [93.0, 93.9, 84.0],
      [92.3, 93.0, 85.5],
      [91.6, 92.0, 86.6],
      [90.8, 91.1, 87.6],
      [89.8, 89.9, 88.4],
      [88.6, 88.6, 88.7],
      [87.4, 87.3, 89.0],
      [86.3, 86.0, 89.3],
      [84.9, 84.5, 89.4]]

dt2 = [[90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1],
       [90.1, 92.0, 71.1]]

dt = np.array(dt)
dt2 = np.array(dt2)
my_color = np.array([[224, 32, 32], [192, 48, 192], [32, 160, 64]])/255.0
my_line_width = 1.5
plt.style.use('ggplot')
ind = np.arange(1, 2.6, 0.1)
label = ['OS', 'OS*', 'UNK']
th = '2'


if __name__ == '__main__':
    ax_all = []
    # ind = np.arange(len(dt))
    fig, ax = plt.subplots(figsize=(1, 2))
    # ax
    for dt_i in [0, 2]:
    # for dt_i in range(len(dt[0])):
        ax_tmp, = ax.plot(ind, dt[:, dt_i], '-', marker='*', color=my_color[dt_i, :], linewidth=my_line_width, label=label[dt_i])
        ax_all.append(ax_tmp)
        ax_tmp2, = ax.plot(ind, dt2[:, dt_i], '--', color=my_color[dt_i, :], linewidth=my_line_width)
    ax.legend(handles=ax_all, fontsize=16, loc='lower right')
    # ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('OS, OS* and UNK w.r.t A->D', fontsize=20)
    ax.tick_params(labelsize=14)
    plt.show()
    fig.savefig('/home/bzhang3/zhong/OSBP/Office_31/analysis/para_alpha_ad.png', bbox_inches='tight', dpi=256)
