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

def main():

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

    cut = (true < ma) & (true > 4.0)
    print(sim_res, res, ma)
    print("Error", np.average(np.abs(true[cut] - predicted[cut])))
    print("Uncer", np.average(error[cut]))

    fig, ax = sd.set_plot_params()

    for i in range(31):
        new_err = np.std(predicted[i*15:i*15+10])
        error[i*15] = np.sqrt(np.square(error[i*15]) + np.square(new_err))

    ax.errorbar(true[::15], predicted[::15], yerr=error[::15], fmt='k.', alpha=1, linewidth=2, markersize=8)

    ax.plot([0,30],[0,30], 'k--', alpha=0.5)
    ax.set_xlim((0,16))
    ax.set_ylim((0,16))

    ax.set_xlabel("True WDM Mass [keV]")
    ax.set_ylabel("Predicted WDM Mass [keV]")

    path = f'/home/j.rose/Projects/CAMELS/GRIDS/CNN/output/results_better_{sim_res}/'
    pixels = 512
    sim_res = 512

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

    ax.errorbar(1/true, 1/predicted, yerr=error/predicted/predicted, fmt='.', alpha=alpha)

    fig.savefig(f"../GRIDS/analysis/plots/results_better/power_spec/results_power_spec_comparison.pdf", bbox_inches='tight')
    return


if __name__=="__main__":
    main()
