import numpy as np
import matplotlib.pyplot as plt

data = ['amazon', 'dslr', 'webcam']
my_color = np.array([[224, 32, 32], [32, 160, 64], [159, 79, 79], [192, 48, 192]])/255.0
# my_color = np.array([[224, 32, 32], [255, 192, 0], [32, 160, 64], [48, 96, 192], [159, 79, 79], [192, 48, 192]])/255.0
my_line_width = 3.5
num_epochs = 201
epochs = np.arange(1, num_epochs+1)
plt.style.use('ggplot')
beta = ['0', '0.1', '0.2']
# beta = ['0', '0.05', '0.1', '0.15', '0.2']
label = ['beta=0', 'beta=0.1', 'beta=0.2', 'no beta']
# label = ['beta=0', 'beta=0.05', 'beta=0.1', 'beta=0.15', 'beta=0.2', 'no beta']
# show_os = True
show_os = False


def get_data():
    # dt_ = [[]]*len(label)
    dt_ = [[], [], [], []]
    for beta_ii in beta:
        dt_dir0 = '/home/bzhang3/zhong/OSBP/Office_31/result2/relu_beta'+beta_ii+'/'+data[0]+'.mat_'+data[2]+'.mat/'
        if show_os:
            for th in ['0', '1', '2']:
                dt_[beta.index(beta_ii)].append(np.genfromtxt(dt_dir0+th+'th_result.txt'))
        else:
            for th in ['0', '1', '2']:
                dt_[beta.index(beta_ii)].append(np.genfromtxt(dt_dir0+th+'th_loss.txt')-float(beta_ii))
    dt_dir0 = '/home/bzhang3/zhong/OSBP/Office_31/result2/no_relu_nobeta/' + data[0] + '.mat_'+data[2] + '.mat/'
    if show_os:
        for th in ['0', '1', '2']:
            dt_[-1].append(np.genfromtxt(dt_dir0+th+'th_result.txt'))
    else:
        for th in ['0', '1', '2']:
            dt_[-1].append(np.genfromtxt(dt_dir0+th + 'th_loss.txt'))
    if show_os:
        means = np.mean(dt_, 1)[:, :, 0]
        stds = np.std(dt_, 1)[:, :, 0]
    else:
        means = np.mean(dt_, 1)[:, :, 2]
        stds = np.std(dt_, 1)[:, :, 2]
    upper = means + stds
    lower = means - stds
    return means, lower, upper


if __name__ == '__main__':
    dt_mean, lower, upper = get_data()
    ax_all = []
    fig, ax = plt.subplots(figsize=(1, 2))
    ax.patch.set_facecolor('#EEEEEE')
    for dt_i in range(len(dt_mean)):
        ax_tmp, = ax.plot(epochs, dt_mean[dt_i]*100, '-', color=my_color[dt_i, :], linewidth=my_line_width, label=label[dt_i])
        ax_all.append(ax_tmp)
        ax.fill_between(epochs, lower[dt_i]*100, upper[dt_i]*100, color=my_color[dt_i, :], alpha=0.2)
    if show_os:
        ax.legend(handles=ax_all, fontsize=24, loc='lower right')
    else:
        ax.legend(handles=ax_all, fontsize=24, loc='lower left')
    # ax.set_ylim(70, 100)
    ax.set_xlim(0, 201)
    ax.set_xlabel('Epoch', fontsize=28)
    if show_os:
        ax.set_ylabel('OS w.r.t A->W', fontsize=28)
    else:
        ax.set_ylabel('Loss w.r.t A->W', fontsize=28)
    ax.tick_params(labelsize=24)
    plt.show()
    if show_os:
        fig.savefig('/home/bzhang3/zhong/OSBP/fig/beta/beta_ad_os.pdf', bbox_inches='tight', dpi=256)
    else:
        fig.savefig('/home/bzhang3/zhong/OSBP/fig/beta/beta_ad_loss.pdf', bbox_inches='tight', dpi=256)
