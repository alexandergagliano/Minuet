import numpy as np
import h5py
import os
import pickle
import glob 

trainPath = '/pscratch/sd/a/agaglian/galaxyData/split_files/Train/'
valPath = '/pscratch/sd/a/agaglian/galaxyData/split_files/Val/'

trainFiles = glob.glob(os.path.join(trainPath, '*.hdf5'))
valFiles = glob.glob(os.path.join(valPath, '*.hdf5'))

dataset = {'train':trainFiles, 'val':valFiles}

print(f"Found {len(trainFiles)} training files.")
print(f"Found {len(valFiles)} validation files.")

keys = ['SPEC_Z', 'PHOTO_Z', 'PHOTO_ZERR', 'MASS_BEST', 'MASS_ERR']

stat_dict = {}

for key, val in dataset.items():

    print(f"Storing information for {key}ing data...")

    stat_dict[f'{key}_means'] = []
    stat_dict[f'{key}_stds'] = []

    zbest = []
    mass_best = []
    sfr_best = []

    for file in val:
        with h5py.File(file, 'r') as f:
            temp_zbest = f['PHOTO_Z'][:]
            temp_zspec = f['SPEC_Z'][:]
            print(temp_zbest)

            temp_zbest[temp_zspec > 0] = temp_zspec[temp_zspec > 0]
            temp_zbest = temp_zbest[temp_zbest > 0]
            temp_zbest = temp_zbest[temp_zbest < 1]

            temp_mass = f['MASS_BEST'][:]
            print(temp_mass)
            temp_mass = temp_mass[temp_mass > -5]
            temp_mass = temp_mass[temp_mass < 15]

            temp_sfr = f['SFR_BEST'][:]
            print(temp_sfr)

            zbest.append(temp_zbest)
            mass_best.append(temp_mass)
            sfr_best.append(temp_sfr)

    zbest = np.concatenate(zbest)
    mass_best = np.concatenate(mass_best)
    sfr_best = np.concatenate(sfr_best)

    stat_dict[f'{key}_means'].append(np.nanmean(zbest))
    stat_dict[f'{key}_means'].append(np.nanmean(mass_best))
    stat_dict[f'{key}_means'].append(np.nanmean(sfr_best))

    stat_dict[f'{key}_stds'].append(np.nanstd(zbest))
    stat_dict[f'{key}_stds'].append(np.nanstd(mass_best))
    stat_dict[f'{key}_stds'].append(np.nanstd(sfr_best))

    print("Stored.")

print("Successfully retrieved all stats. saving....")

with open('/global/cfs/cdirs/lsst/groups/TD/SN/SNANA/SURVEYS/LSST/USERS/agaglian/galaxyAutoencoder/data/norm_values.pkl', 'wb') as handle:
    pickle.dump(stat_dict, handle, protocol=4)#pickle.HIGHEST_PROTOCOL)
