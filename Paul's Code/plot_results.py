import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic as bs
import shared_data as sd

def fix_range(data):

    mi = 0.0333333
    ma = 0.4

    data[:,0] = (data[:,0] * (ma - mi)) + mi
    data[:,1] = (data[:,0] * (ma - mi)) + mi

    return data

#select which images from each simulation should be used to do analysis
def get_selected_images(num_tests, overlap, ims_in_sim):

    selected_images = []
    for s in range(num_tests):
        if overlap > ims_in_sim:
            ims = np.random.choice(np.arange(ims_in_sim), overlap, True)
        else:
            ims = np.random.choice(np.arange(ims_in_sim), overlap, False)
        rot = np.random.randint(0,8,overlap)
        selected_images.extend(list((s*ims_in_sim*8) + (ims*8 + rot)))

    return np.array(selected_images)


def main():

    np.random.seed(1234) 

    res = 512
    sim_res = 512
    ma = 6

    redshift = 0

    skip = 15

    data0 = np.loadtxt(f"output/Results_res{res}_sim{sim_res}_z{redshift}_{res//256*32}k.txt")
    data1 = np.loadtxt(f"output/Results_res{res}_sim{sim_res}_z{redshift}_{res//256*32}k_test.txt")

    #d0 = list(data0[2::15])
    #d1 = list(data1[2::15])
    d0 = list(data0)
    d1 = list(data1)
    d0.extend(d1)
    data = np.array(d0)

    data = data[np.argsort(data[:,0], kind='mergesort')]

    #data = fix_range(data)

    true = 1/data[:,0]
    predicted = 1/data[:,1]
    error = data[:,2] / (data[:,1] * data[:,1])

    #cut = (true < ma) & (true > 4.0)
    #print(sim_res, res, ma)
    #print("Error", np.average(np.abs(true[cut] - predicted[cut])))
    #print("Uncer", np.average(error[cut]))

    fig, ax1 = sd.set_plot_params(nrows=1, figsize=(5,2.5))

    #ax1 = ax[0]
    #ax2 = ax[1]

    #ax1.get_shared_x_axes().join(ax1, ax2)

    for i in range(31):
        new_err = np.std(predicted[i*15:i*15+10])
        error[i*15] = np.sqrt(np.square(error[i*15]) + np.square(new_err))

    ax1.errorbar(true[::15], predicted[::15] - true[::15], yerr=error[::15], fmt='k.', alpha=1, linewidth=2, markersize=8, label='Power Spectra')
    ax1.legend(fontsize=14)

    ax1.plot([0,30],[0,0], 'k--', alpha=0.5)
    ax1.set_xlim((2,12))
    ax1.set_ylim((-4,4))
    ax1.set_xlabel("True WDM Mass [keV]", fontsize=20)
    #ax2.ylabel("Predicted - True WDM Mass [keV]", fontsize=20)
    ax1.set_ylabel("Predicted - True", fontsize=20)
    fig.savefig(f"../GRIDS/analysis/plots/results_better/power_spec/postdoc_plot1.pdf", bbox_inches='tight')

    #ax2.set_xlabel("True WDM Mass [keV]")
    #ax1.set_ylabel("Predicted WDM Mass [keV]")

    #box = dict(boxstyle="round",ec=(0.5, 0.5, 0.5), fc=(0.9, 0.9, 0.9))
    #ax.text(3.2,28.1,f"Resolutions: {sim_res}$^3$; {res}$^2$", bbox=box)

    #inax = ax.inset_axes([0.06,0.43,0.4,0.56])
    #cut = true < 5
    #inax.set_xlim((2.4,5.1))
    #inax.set_ylim((2.4,5.1))
    #inax.plot([2.5,5],[2.5,5], 'k--', alpha=0.5)
    #inax.errorbar(true[cut][::15], predicted[cut][::15], yerr=error[cut][::15], fmt='k.', linewidth=2, markersize=8)

    #inax.set_aspect(0.7143)

    #inax.tick_params('both', which='minor', length=4, direction='in', bottom=True, top=True, left=True, right=True)
    #inax.tick_params('both', which='major', length=8, direction='in', bottom=True, top=True, left=True, right=True)

    #for axis in ['top','bottom','left','right']:
    #    inax.spines[axis].set_linewidth(1)

    #ax.plot([2.5,5],[2.5,2.5],'k-',linewidth=0.5)
    #ax.plot([2.5,5],[5,5],'k-',linewidth=0.5)
    #ax.plot([2.5,2.5],[2.5,5],'k-',linewidth=0.5)
    #ax.plot([5,5],[2.5,5],'k-',linewidth=0.5)

    pixels = 512
    sim_res = 512
    overlap = 1

    path = f'/home/j.rose/Projects/CAMELS/GRIDS/CNN/output/results_better_{sim_res}/'
    data0 = np.loadtxt(f"{path}/results_train_WDM_Nbody_all_steps_500_500_o3_{pixels}_better_{sim_res}_test_WDM_Nbody_all_steps_500_500_o3_{pixels}_z{redshift}.0_LH_{sim_res}.txt")
    data1 = np.loadtxt(f"{path}/results_train_WDM_Nbody_all_steps_500_500_o3_{pixels}_better_{sim_res}_test_WDM_Nbody_all_steps_500_500_o3_{pixels}_z{redshift}.0_LH_{sim_res}_test.txt")

    num_test_in_data0 = 10
    ims_in_sim = len(data0) // (num_test_in_data0*8)
    num_test = num_test_in_data0 + (len(data1)//8//ims_in_sim)
    data = np.array(list(data0) + list(data1))
  
    idx = np.argsort(data[:,0], kind='mergesort')[::-1]
    data = data[idx]

    selected_images = get_selected_images(num_test, overlap, ims_in_sim)

    true = 1/data[selected_images,0]
    predicted = 1/data[selected_images,1]
    error = data[selected_images,2] * (predicted * predicted)

    fig, ax2 = sd.set_plot_params(nrows=1, figsize=(5,2.5))

    ax2.errorbar(true, predicted - true, yerr=error, fmt='.', alpha=1, markersize=8, linewidth=2, color='k', label='Field Level') 
    ax2.plot([0,30],[0,0],'k--', alpha=0.5)
    ax2.set_ylim((-4,4))
    ax2.legend(fontsize=14)
    ax2.set_xlabel("True WDM Mass [keV]", fontsize=20)
    #ax2.ylabel("Predicted - True WDM Mass [keV]", fontsize=20)
    ax2.set_ylabel("Predicted - True", fontsize=20)
    ax2.set_xlim((2,12))
    ax2.set_ylim((-4,4))

    #fig.add_subplot(111, frameon=False)
    #plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #plt.grid(False)
    #plt.xlabel("True WDM Mass [keV]", fontsize=20)
    #plt.ylabel("Predicted - True WDM Mass [keV]", fontsize=20)

    fig.savefig(f"../GRIDS/analysis/plots/results_better/power_spec/postdoc_plot2.pdf", bbox_inches='tight')

    return


if __name__=="__main__":
    main()
