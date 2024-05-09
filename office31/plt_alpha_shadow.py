import numpy as np
import matplotlib.pyplot as plt
import PIE2.result_analysis as re_ana

data = ['amazon', 'dslr', 'webcam']
dt2 = [76.9, 79.9]
# dt2 = [91.9, 68.2]
my_color = np.array([[224, 32, 32], [32, 160, 64], [192, 48, 192]])/255.0
my_line_width = 3.5
plt.style.use('ggplot')
ind = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0']
# ind = ['1.0', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5']
label = ['OS', 'OS*', 'UNK']


def get_data():
    os_, os_star, unk = [], [], []
    for beta_ii in ind:
        dt_dir0 = '/home/bzhang3/zhong/OSBP/Office_31/result2/relu_alpha'+beta_ii+'/'+data[0]+'.mat_'+data[2]+'.mat'
        re = re_ana.find_loc_2(dt_dir0, cycle=3, loc=100, avg_num=1)
        os_.append(np.array(re)[:, 0])
        os_star.append(np.array(re)[:, 1])
        unk.append(np.array(re)[:, 2])
    mean_os = np.mean(os_, 1)
    std_os = np.std(os_, 1)
    upper_os = mean_os + std_os
    lower_os = mean_os - std_os
    mean_os_star = np.mean(os_star, 1)
    std_os_star = np.std(os_star, 1)
    upper_os_star = mean_os_star + std_os_star
    lower_os_star = mean_os_star - std_os_star
    mean_unk = np.mean(unk, 1)
    std_unk = np.std(unk, 1)
    upper_unk = mean_unk + std_unk
    lower_unk = mean_unk - std_unk
    return [mean_os, upper_os, lower_os], [mean_os_star, upper_os_star, lower_os_star], [mean_unk, upper_unk, lower_unk]


if __name__ == '__main__':
    os_, os_star, unk = get_data()
    ax_all = []
    fig, ax = plt.subplots(figsize=(1, 2))
    ax.patch.set_facecolor("#EEEEEE")
    ax_os, = ax.plot(ind, os_[0]*100, '-', marker='*', markersize=15, color=my_color[0, :], linewidth=my_line_width, label=label[0])
    ax_all.append(ax_os)
    ax_os_star, = ax.plot(ind, os_star[0]*100, '-', marker='o', markersize=15, color=my_color[1, :], linewidth=my_line_width, label=label[1])
    ax_all.append(ax_os_star)
    ax_unk, = ax.plot(ind, unk[0]*100, '-', marker='s', markersize=15, color=my_color[2, :], linewidth=my_line_width, label=label[2])
    ax_all.append(ax_unk)

    # ax.fill_between(ind, os_[2]*100, os_[1]*100, color=my_color[0, :], alpha=0.4)
    # ax.fill_between(ind, unk[2]*100, unk[1]*100, color=my_color[2, :], alpha=0.4)
    # ax.fill_between(ind, os_star[2]*100, os_star[1]*100, color=my_color[1, :], alpha=0.4)

    ax.axhline(dt2[0], linestyle='-.', color=my_color[0, :], linewidth=my_line_width)
    # ax.axhline(dt2[1], linestyle='-.', color=my_color[1, :], linewidth=my_line_width)
    ax.legend(handles=ax_all, fontsize=18, loc='lower right')
    # ax.legend(handles=ax_all, fontsize=18, loc='upper right')
    # ax.set_ylim(80, 100)
    ax.set_xlim(1, 10)
    ax.set_xlabel('alpha', fontsize=24)
    ax.set_ylabel('OS and UNK w.r.t A->D', fontsize=24)
    ax.tick_params(labelsize=20)
    plt.show()
    fig.savefig('/home/bzhang3/zhong/OSBP/fig/alpha/alpha_ad2.png', bbox_inches='tight', dpi=256)
    # fig.savefig('/home/bzhang3/zhong/OSBP/fig/alpha/alpha_ad2_shadow.png', bbox_inches='tight', dpi=256)
