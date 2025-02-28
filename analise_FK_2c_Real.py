#%%
import numpy as np
import os
import matplotlib.pyplot as plt
from FK_2c_cmaes import writeNii
import json
import argparse

# Modified savePlot function without ground truth
def savePlot(values, yLab, title, file_name):
    plt.figure(figsize=(12, 7))
    for i in range(len(values)):
        plt.scatter([res["nsamples"][i]] * len(values[i]), values[i], color="tab:blue", marker=".")
    plt.ylabel(yLab)
    plt.xlabel("Generation")
    plt.title(yLab + " ---  " + title)
    plt.savefig(os.path.join(plots_path, file_name))
    plt.close()

# Setup argument parser
parser = argparse.ArgumentParser(description='Process file paths.')
parser.add_argument('base_path', type=str, help='Base path for the data files')
parser.add_argument('data_path', type=str, help='Path for the ground truth values')
args = parser.parse_args()

# Use the paths from the arguments
base_path = args.base_path
data_path = args.data_path

# Create a directory for plots
plots_path = os.path.join(base_path, 'plots')
os.makedirs(plots_path, exist_ok=True)

#%%
# Load results
res = np.load(f"{base_path}/results.npy", allow_pickle=True).item()

#%%
# Find the minimum total loss and optimal parameters
lossDir = res["lossDir"]
minLoss = 1
for i in range(len(lossDir)):
    for j in range(len(lossDir[i])):
        if lossDir[i][j]["lossTotal"] < minLoss:
            minLoss = lossDir[i][j]["lossTotal"]
            opt = lossDir[i][j]["allParams"]
            bestLossDir = lossDir[i][j]

print("minLoss", minLoss)
print("opt", opt)
print("bestLossDir", bestLossDir)

#%%
# Initialize arrays for losses and parameters
lossEdema, lossEnhancing, lossNecrotic, times, xs, resfactor = [], [], [], [], [], []
for i in range(len(res["lossDir"])):
    lossEdema_, lossEnhancing_, lossNecrotic_, times_, xs_, resfactor_ = [], [], [], [], [], []
    for j in range(len(res["lossDir"][i])):
        lossEdema_.append(res["lossDir"][i][j]["lossEdema"])
        lossEnhancing_.append(res["lossDir"][i][j]["lossEnhancing"])
        lossNecrotic_.append(res["lossDir"][i][j]["lossNecrotic"])
        times_.append(res["lossDir"][i][j]["time"])
        xs_.append(res["lossDir"][i][j]["allParams"])
        resfactor_.append(res["lossDir"][i][j]["resolution_factor"])
    lossEdema.append(lossEdema_)
    lossEnhancing.append(lossEnhancing_)
    lossNecrotic.append(lossNecrotic_)
    times.append(times_)
    xs.append(xs_)
    resfactor.append(resfactor_)

times = np.array(times)/60
xs = np.array(xs)

#%%
# Function to plot values
def savePlot(values, yLab, title, file_name):
    plt.figure(figsize=(12, 7))
    for i in range(len(values)):
        plt.scatter([res["nsamples"][i]] * len(values[i]), values[i], color="tab:blue", marker=".")
    plt.ylabel(yLab)
    plt.xlabel("Generation")
    plt.title(yLab + " ---  " + title)
    plt.savefig(os.path.join(plots_path, file_name))
    plt.close()

# Plotting parameters
title = "Samples: " + np.max(res["nsamples"]).astype(str) + " - cumulative time: " + str(np.round(np.sum(times), 2)) + " - parallel time: " + str(np.round(res["time_min"], 1)) + "min"

# Plotting time, resolution factor, and loss components
savePlot(times, "time [min]", title, "time_plot.png")
savePlot(resfactor, "resolution_factor", title, "resolution_factor_plot.png")
savePlot(lossEdema, "lossEdema", title, 'lossEdema.png')
savePlot(lossEnhancing, "lossEnhancing", title, 'lossEnhancing.png')
savePlot(lossNecrotic, "lossNecrotic", title, 'lossNecrotic.png')
# Plotting parameters
savePlot(xs[:,:,3], "rho", title, 'rho.png')
savePlot(xs[:,:,4], "Dw", title, 'Dw.png')
savePlot(xs[:,:,5], "lambda_np", title, 'lambda_np.png')
savePlot(xs[:,:,6], "sigma_np", title, 'sigma_np.png')
savePlot(xs[:,:,7], "D_s", title, 'D_s.png')
savePlot(xs[:,:,8], "lambda_s", title, 'lambda_s.png')

# Add more plots for other parameters as needed

#%%
# Plot for time error bar
plt.figure(figsize=(12, 7))
plt.title(title)
plt.errorbar(res["nsamples"], np.mean(times, axis=1), yerr=np.std(times, axis=1), fmt='o', color='black', label="time")
plt.ylabel("time [min]")
plt.xlabel("nsamples")
plt.legend()

#%%
# Plot for combined loss
plt.figure(figsize=(12, 7))
combinedLoss = (np.array(lossEdema) + np.array(lossEnhancing) + np.array(lossNecrotic))/3
plt.errorbar(res["nsamples"], np.mean(combinedLoss, axis=1), yerr=np.std(combinedLoss, axis=1), label="combinedLoss")
plt.ylabel("loss")
plt.xlabel("number of samples")
plt.legend()


# %%
# Function to plot the 3D graph without ground truth
def plot3D(x, y, z, title, file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a colormap based on the z values
    colors = plt.cm.jet(np.linspace(0, 1, len(z)))

    # Plot each segment with a different color
    for i in range(1, len(x)):
        ax.plot(x[i-1:i+1], y[i-1:i+1], z[i-1:i+1], color=colors[i])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.savefig(os.path.join(plots_path, file_name))
    plt.close()

# Your existing code for parameters
params = np.array(res["xs0s"])
x = params.T[0]   
y = params.T[1]
z = params.T[2]

# Call the plotting function without ground truth values
plot3D(x, y, z, "Origin", "3d_plot.png")


# # %%
# from TumorGrowthToolkit.FK_2c import Solver
# import nibabel as nib
# # Get probability images for Grey and White matter
# GM =  nib.load("/Users/michal/Documents/TumorGrowthToolkit/synthetic_gens/synthetic_runs1T_FK_2c/synthetic1T_run0/gm_data.nii.gz").get_fdata()
# WM =  nib.load("/Users/michal/Documents/TumorGrowthToolkit/synthetic_gens/synthetic_runs1T_FK_2c/synthetic1T_run0/wm_data.nii.gz").get_fdata()
# params = np.array(res["xs0s"])
# print(f'Running in res factor:{res["lossDir"][i][j]["resolution_factor"]}')
# x = params[-1]
# parameters = {
#     'Dw': x[4],
#     'rho': x[3],
#     'lambda_np': x[5],
#     'sigma_np': x[6],
#     'D_s': x[7],
#     'lambda_s': x[8],
#     'gm': GM,
#     'wm': WM,
#     'NxT1_pct': x[0],
#     'NyT1_pct': x[1],
#     'NzT1_pct': x[2],
#     'resolution_factor': res["lossDir"][-1][-1]["resolution_factor"]
# }


# # Run the FK_solver and plot the results
# fk_solver = Solver(parameters)
# result = fk_solver.solve()
# writeNii(result['final_state'], base_path)


#%%

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def plot_and_save_field_comparisons(base_path, plots_path, x, data_path):
    file_suffixes = ['_S.nii.gz', '_P.nii.gz', '_N.nii.gz']
    titles = ['Nutrients (S)', 'Proliferative cells (P)', 'Necrotic cells (N)']
    result_arrays = []
    min_x, max_x = (10, 10)
    min_y, max_y = (10, 10)
    
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_cmap',
        ['#abd9e9', '#2c7bb6', '#fdae61'],  # light blue, brown, orange
        N=256
    )

    # Load result arrays
    for suffix in file_suffixes:
        path = os.path.join(base_path, suffix)
        if os.path.exists(path):
            img = nib.load(path)
            result_arrays.append(img.get_fdata())

    # Load additional data
    segmentation = nib.load(os.path.join(data_path, "segm.nii.gz")).get_fdata()
    pet = nib.load(os.path.join(data_path, "FET.nii.gz")).get_fdata()
    wm_data = nib.load(os.path.join(data_path, "t1_wm.nii.gz")).get_fdata()
    gm_data = nib.load(os.path.join(data_path, "t1_gm.nii.gz")).get_fdata()

    # Extract tumor regions
    FLAIR = np.where(np.logical_or(segmentation == 3, segmentation == 1), 1, np.nan)
    necrotic = np.where(segmentation == 4, 1, np.nan)
    enhancing = np.where(np.logical_or(segmentation == 1, segmentation == 4), 1, np.nan)

    # Inferred origin
    inferred_origin = (
        int(x[1] * result_arrays[0].shape[1]) - min_y,
        int(x[0] * result_arrays[0].shape[0]) - min_x,
        int(x[2] * result_arrays[0].shape[2])
    )

    #z_slice = np.argmax(np.sum(result_arrays[2], axis=(0, 1)))  # Default slice
    #z_slice = int(x[2] * result_arrays[0].shape[2])
    z_slice = np.argmax(np.sum(np.where(segmentation == 1,1,0), axis=(0, 1))) +5

    # Plotting setup with an additional row for inputs
    fig, axes = plt.subplots(2, len(result_arrays), figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1]})
    th_plot = 0.05

    for i in range(len(result_arrays)):
        # Plot result arrays
        vol = result_arrays[i][min_x:-max_x, min_y:-max_y, z_slice].T
        vol_display = np.array(np.where(vol > th_plot, vol, np.nan))
        axes[0, i].imshow(vol_display, cmap=cmap)
        axes[0, i].invert_yaxis()
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Result: {titles[i]}')
        axes[0, i].set_facecolor('white')

        # Overlay wm_data and gm_data contours on result arrays
        axes[0, i].contourf(gm_data[min_x:-max_x, min_y:-max_y, z_slice].T, levels=[0.5, 1], colors='gray', alpha=0.35)

        # Mark inferred origin on result arrays
        #axes[0, i].plot(inferred_origin[1], inferred_origin[0], 'o', color='orange')  # orange dot
                
    # Tumor Segmentation
    axes[1, 0].contourf(gm_data[min_x:-max_x, min_y:-max_y, z_slice].T, levels=[0.5, 1], colors='gray', alpha=0.35)
    axes[1, 0].imshow(FLAIR[min_x:-max_x, min_y:-max_y, z_slice].T, cmap="Blues", alpha=0.7, vmin=0, vmax=1)
    axes[1, 0].imshow(enhancing[min_x:-max_x, min_y:-max_y, z_slice].T, cmap="Greens", alpha=1, vmin=0, vmax=1)
    axes[1, 0].imshow(necrotic[min_x:-max_x, min_y:-max_y, z_slice].T, cmap="Reds", alpha=1, vmin=0, vmax=1)
    axes[1, 0].axis('off')
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title(f'Segmentation')
    # Create a legend
    orange_patch = mpatches.Patch(color='Blue', label='Edema')
    red_patch = mpatches.Patch(color='Red', label='Necrotic core')
    brown_patch = mpatches.Patch(color='Green', label='Enhancing core')
    axes[1, 0].legend(handles=[orange_patch, red_patch, brown_patch])
    
    # PET Signal
    vol = pet[min_x:-max_x, min_y:-max_y, z_slice].T
    vol_display = np.array(np.where(vol > th_plot, vol, np.nan))
    axes[1, 1].contourf(gm_data[min_x:-max_x, min_y:-max_y, z_slice].T, levels=[0.5, 1], colors='gray', alpha=0.35)
    pet_plot = axes[1, 1].imshow(vol_display, cmap=cmap, alpha=1)
    axes[1, 1].invert_yaxis()
    axes[1, 1].axis('off')
    axes[1, 1].set_title(f'FET-PET')

    # Add colorbar for PET signal
    colorbar_axes = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # Adjust these values as needed
    cbar = fig.colorbar(pet_plot, cax=colorbar_axes)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # Set custom tick labels

    axes[1, 2].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join(plots_path, "field_comparisons.png"))
    plt.close()

plot_and_save_field_comparisons(base_path, plots_path, x,data_path)

# %%
