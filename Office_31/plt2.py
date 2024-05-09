import numpy as np
import matplotlib.pyplot as plt

data = ['amazon', 'dslr', 'webcam']
my_color = np.array([[224, 32, 32], [255, 192, 0], [32, 160, 64], [48, 96, 192], [159, 79, 79], [192, 48, 192]])/255.0
my_line_width = 1.5
num_epochs = 201
epochs = np.arange(1, num_epochs+1)
plt.style.use('ggplot')
beta = ['0', '0.025', '0.05', '0.075', '0.1']
label = ['beta=0', 'beta=0.025', 'beta=0.05', 'beta=0.075', 'beta=0.1', 'no beta']
th = '0'
# show_os = True
show_os = False


def get_data():
    dt_ = []
    for beta_ii in beta:
        dt_dir0 = '/home/bzhang3/zhong/OSBP/Office_31/result/relubeta'+beta_ii+'/'+data[0]+'.mat_'+data[1]+'.mat/'
        if show_os:
            dt_.append(np.genfromtxt(dt_dir0+th+'th_result.txt'))
        else:
            dt_.append(np.genfromtxt(dt_dir0+th+'th_loss.txt')-float(beta_ii))
    dt_dir0 = '/home/bzhang3/zhong/OSBP/Office_31/result/no_relunobeta/' + data[0] + '.mat_'+data[1] + '.mat/'
    if show_os:
        dt_.append(np.genfromtxt(dt_dir0+th+'th_result.txt'))
    else:
        dt_.append(np.genfromtxt(dt_dir0 + th + 'th_loss.txt'))
    return dt_


if __name__ == '__main__':
    dt = get_data()
    ax_all = []
    fig, ax = plt.subplots(figsize=(1, 2))
    for dt_i in range(len(dt)):
        if show_os:
            ax_tmp, = ax.plot(epochs, (dt[dt_i][:, 0])*100, '-', color=my_color[dt_i, :], linewidth=my_line_width, label=label[dt_i])
        else:
            ax_tmp, = ax.plot(epochs, dt[dt_i][:, 2], '-', color=my_color[dt_i, :], linewidth=my_line_width, label=label[dt_i])
        ax_all.append(ax_tmp)
    if show_os:
        ax.legend(handles=ax_all, fontsize=16, loc='lower right')
    else:
        ax.legend(handles=ax_all, fontsize=16, loc='lower left')
    ax.set_xlabel('Epoch', fontsize=20)
    if show_os:
        ax.set_ylim(75, 100)
        ax.set_ylabel('OS w.r.t A->D', fontsize=20)
    else:
        ax.set_ylabel('Loss w.r.t A->D', fontsize=20)
    ax.tick_params(labelsize=14)
    plt.show()
    if show_os:
        fig.savefig('/home/bzhang3/zhong/OSBP/Office_31/analysis/beta_ad_os.png', bbox_inches='tight', dpi=256)
    else:
        fig.savefig('/home/bzhang3/zhong/OSBP/Office_31/analysis/beta_ad_loss.png', bbox_inches='tight', dpi=256)
