import numpy as np
import matplotlib.pyplot as plt

data = ['amazon', 'dslr', 'webcam']
dt_dir0 = '/home/bzhang3/zhong/OSBP/Office_31/result/no_relu03/'+data[0]+'.mat_'+data[2]+'.mat/'
dt_dir1 = '/home/bzhang3/zhong/OSBP/Office_31/result/relu03/'+data[0]+'.mat_'+data[2]+'.mat/'
th = '1'
data1_loss = np.genfromtxt(dt_dir0+th+'th_loss.txt')
data1_acc = np.genfromtxt(dt_dir0+th+'th_result.txt')
data2_loss = np.genfromtxt(dt_dir1+th+'th_loss.txt')
data2_acc = np.genfromtxt(dt_dir1+th+'th_result.txt')

my_color = np.array([[224, 32, 32], [255, 192, 0], [32, 160, 64], [48, 96, 192], [192, 48, 192]])/255.0
my_line_width = 1.5
num_epochs = 201
epochs = np.arange(1, num_epochs+1)
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(1, 2))
no_relu, = ax.plot(epochs, data1_acc[:, 0], '-', color=my_color[3, :], linewidth=my_line_width, label='Open set difference')
relu, = ax.plot(epochs, data2_acc[:, 0], '-', color=my_color[0, :], linewidth=my_line_width, label='Non-negative open set difference')
ax.legend(handles=[relu, no_relu], fontsize=16, loc='lower right')
# ax.set_ylim(0.7, 1)
ax.set_xlabel('Epoch', fontsize=20)
ax.tick_params(labelsize=14)
ax.set_ylabel('OS w.r.t A->W', fontsize=20)

fig1, ax1 = plt.subplots(figsize=(1, 2))
no_relu, = ax1.plot(epochs, data1_loss[:, 2], '-', color=my_color[3, :], linewidth=my_line_width, label='Open set difference')
relu, = ax1.plot(epochs, data2_loss[:, 2], '-', color=my_color[0, :], linewidth=my_line_width, label='Non-negative open set difference')
ax1.legend(handles=[relu, no_relu], fontsize=16, loc='lower left')
ax1.set_xlabel('Epoch', fontsize=20)
ax1.tick_params(labelsize=14)
ax1.set_ylabel('Risk w.r.t A->W', fontsize=20)

plt.show()
# fig.savefig('aw_os.png', bbox_inches='tight', dpi=256)
# fig1.savefig('aw_loss.png', bbox_inches='tight', dpi=256)
