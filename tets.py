import numpy as np
import os
import nibabel as nib
import time
from scipy import ndimage
import nibabel as nib
import matplotlib.pyplot as plt
import cmaesFK
import tools

WM = nib.load("data_mehdi/wm.nii.gz").get_fdata()
GM = nib.load("data_mehdi/gm.nii.gz").get_fdata()

plt.title("White and Gray matter")
zSlice = 75
plt.imshow(GM[:, :, zSlice],  cmap="Greys", alpha=1 *GM[:, :, 75])
plt.imshow(WM[:, :, zSlice],  cmap="Greys", alpha=0.5*WM[:, :, 75])
plt.show()

segmentation = nib.load("data_mehdi/tumor_autoseg.nii.gz").get_fdata()

edema = np.logical_or(segmentation == 2, segmentation == 4)
necrotic = segmentation == 1
enhancing = segmentation == 3

# plotting
zSlice = int(ndimage.center_of_mass(edema)[2])
plt.title("Tumor Segmentation Necrotic (Red) Enhancing (Orange) and Edema (Blue)")
plt.imshow(GM[:, :, zSlice],  cmap="Greys", alpha=0.5 *GM[:, :, zSlice])
plt.imshow(WM[:, :, zSlice],  cmap="Greys", alpha=0.25*WM[:, :, zSlice])
plt.imshow(0.99 * enhancing[:, :, zSlice],  cmap="Oranges", alpha=0.8 * enhancing[:, :, zSlice], label = "enhancing", interpolation="none")
plt.imshow(0.99 * necrotic[:, :, zSlice],  cmap="Reds", alpha=0.8 * necrotic[:, :, zSlice], label = "necrotic", interpolation="none")
plt.imshow(0.99 * edema[:, :, zSlice],  cmap="Blues", alpha=0.8* edema[:, :, zSlice], label = "edema", interpolation="none")
plt.show()

## load the PET dara
# pet = nib.load("exampleData/FET.nii.gz").get_fdata()
# pet = pet / np.max(pet)
#
# plt.title("PET data")
# plt.imshow(GM[:, :, zSlice],  cmap="Greys", alpha=0.5 *GM[:, :, zSlice])
# plt.imshow(WM[:, :, zSlice],  cmap="Greys", alpha=0.25*WM[:, :, zSlice])
# plt.imshow(pet[:, :, zSlice],  cmap="Greens", alpha=0.8* pet[:, :, zSlice], interpolation="none")
# plt.show()


settings = {}

# initial parameters for the total time is fix to T=100
settings["rho0"] = 0.06 # initial proliferation rate
settings["dw0"] = 0.001 # initial diffusion coefficient
settings["thresholdT1c"] = 0.675 # initial threshold for enhancing tumor
settings["thresholdFlair"] = 0.25 # initial threshold for edema

# center of mass
com = ndimage.center_of_mass(edema)

# set initial position of the tumor at the center of mass
settings["NxT1_pct0"] = float(com[0] / np.shape(edema)[0])
settings["NyT1_pct0"] = float(com[1] / np.shape(edema)[1])
settings["NzT1_pct0"] = float(com[2] / np.shape(edema)[2])

# possible ranges for the parameters from left to right
# NxT1_pct, NyT1_pct, NzT1_pct, , rho, dw, thresholdT1c, thresholdFlair
settings["parameterRanges"] = [[0, 1], [0, 1], [0, 1], [0.0001, 0.225], [0.001, 3], [0.5, 0.85], [0.001, 0.5]]

# multiprocessing
settings["workers"] = 12

# initial sigma - hyper parameter for the solver
settings["sigma0"] = 0.02

# resolution factor for the forward model
# It can change with generations: key = from relative generations , value = resolution factor
# i.e. {0:0.6, 0.5:1.0} means that the first half of the run is with 60% resolution and the second with 100%
settings["resolution_factor"] = { 0: 0.3, 0.7: 0.5}

# number of generations for the algorithm
settings["generations"] = 12 # there are 9 samples in each step

solver = cmaesFK.CmaesSolver(settings, WM, GM, edema, enhancing, 0, necrotic)
resultTumor, resultDict = solver.run()

# save results
resultpath = "data_mehdi/"
os.makedirs(resultpath, exist_ok=True)
np.save(resultpath + "gen_"+ str(settings["generations"]) + "_settings.npy", settings)
np.save(resultpath + "gen_"+ str(settings["generations"]) + "_results.npy", resultDict)

#### Saving outputs
write_name = "gen_"+ str(settings["generations"])+"_result.nii.gz"
write_path = os.path.join(resultpath, write_name)
src_nib = nib.load('./data_mehdi/t1.nii.gz')
dst_nib = nib.Nifti1Image(resultTumor, src_nib.affine, src_nib.header)
nib.save(dst_nib, write_path)






# plot final
zSlice = int(ndimage.center_of_mass(edema)[2])

plt.title("Final Tumor Estimation")
plt.imshow(GM[:, :, zSlice],  cmap="Greys", alpha=0.5 *GM[:, :, zSlice])
plt.imshow(WM[:, :, zSlice],  cmap="Greys", alpha=0.25*WM[:, :, zSlice])
plt.imshow(resultTumor[:, :, zSlice],  cmap="Reds", alpha=0.9* resultTumor[:, :, zSlice])

print("Total runtime: ", round(resultDict["time_min"], 1), "min")