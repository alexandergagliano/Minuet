import h5py
import numpy as np
import os
import random
import glob

# Define constants
OUTPUT_DIR = '/n/holylabs/LABS/iaifi_lab/Users/agagliano/galaxyAutoencoder/split_files/'
CHUNK_SIZE = 10_000  
SEED = 42  

files = sorted(glob.glob(os.path.join('/n/netscratch/iaifi_lab/Lab/agagliano/', f"matched*.hdf5")))

print(files)

#assume the same number of entries per file
dataset_info = {}
with h5py.File(files[0], 'r') as f:
    datasets_to_copy = list(f.keys())  # List of all datasets to copy from the original files
    Ndata = f['ID'].shape[0]
    for dataset in datasets_to_copy:
        shape = f[dataset].shape[1:]  # Shape without the number of samples
        dtype = f[dataset].dtype
        dataset_info[dataset] = {'shape': shape, 'dtype': dtype}

Ndata_tot = Ndata*len(files)

#split into train val test
train_frac = 0.6
val_frac = 0.2
test_frac = 1 - train_frac - val_frac

# Set random seed for reproducibility
random.seed(SEED)

# Generate indices for the entire dataset and shuffle
samples = np.arange(Ndata_tot)
np.random.shuffle(samples)

# Split indices into train, validation, and test
train_end = int(train_frac * Ndata_tot)
val_end = int((train_frac + val_frac) * Ndata_tot)

samples_train = samples[:train_end]
samples_val = samples[train_end:val_end]
samples_test = samples[val_end:]

print("len samples_test:")
print(len(samples_test))
print(samples_test)

# Function to create new HDF5 files for each split
def create_split_files(prefix, dataset_info, num_files):
    split_files = []
    for i in range(num_files):
        file_path = os.path.join(OUTPUT_DIR + f'/{prefix}', f"{prefix.lower()}_split_{i}.hdf5")
        split_file = h5py.File(file_path, 'w')
        
        # Create empty datasets for each dataset in original files
        for dataset, info in dataset_info.items():
            shape = info['shape']  # Use pre-fetched shape
            dtype = info['dtype']  # Use pre-fetched dtype
            split_file.create_dataset(dataset, shape=(0,) + shape, maxshape=(None,) + shape, dtype=dtype, chunks=(CHUNK_SIZE,) + shape)
        
        split_files.append(split_file)
    return split_files

# Create split files for train, val, and test
# Calculate the number of files for each split proportionally
#total_files = len(files)  # Total number of input files
events_per_file = 50_000 

# Calculate the total number of files needed for each split
train_files_count = len(samples_train) // events_per_file + (1 if len(samples_train) % events_per_file else 0)
test_files_count = len(samples_test) // events_per_file + (1 if len(samples_test) % events_per_file else 0)
val_files_count = len(samples_val) // events_per_file + (1 if len(samples_val) % events_per_file else 0)

print("Ndata_tot:", Ndata_tot)

print(train_files_count)
print(val_files_count)
print(test_files_count)

# Create split files for train, val, and test with proportional number of files
train_files = create_split_files('Train', dataset_info, train_files_count)
val_files = create_split_files('Val', dataset_info, val_files_count)
test_files = create_split_files('Test', dataset_info, test_files_count)

# Function to write data to corresponding split files in chunks
def write_to_split_files(split_files, split_indices, dataset_names, chunk_size=CHUNK_SIZE, events_per_file=50_000):
    # Debugging: Check the total number of indices
    print(f"Total number of indices: {len(split_indices)}")

    # Iterate over each split file to distribute indices
    for i, split_file in enumerate(split_files):
        start_idx = i * events_per_file
        end_idx = min((i + 1) * events_per_file, len(split_indices))

        # Handle the remaining indices for the last file
        file_indices = split_indices[start_idx:end_idx]

        # Debugging: Print number of entries per file
        print(f"{split_file.filename}: Writing {len(file_indices)} indices (expected: up to {events_per_file}).")

        # Check that the file is not empty
        if len(file_indices) == 0:
            continue

        # Write data for each dataset in chunks
        for dataset_name in dataset_names:
            for input_file in files:  # Use the correct input file from the list
                with h5py.File(input_file, 'r') as h5_file:
                    dataset = h5_file[dataset_name]

                    # Determine global indices that belong to this file
                    local_indices = [idx % Ndata for idx in file_indices if idx // Ndata == files.index(input_file)]
                    if not local_indices:
                        continue

                    # Write in chunks
                    for j in range(0, len(local_indices), chunk_size):
                        chunk_indices = local_indices[j:j + chunk_size]
                        chunk_data = dataset[np.sort(chunk_indices)]

                        # Resize the dataset in the split file and write data
                        current_size = split_file[dataset_name].shape[0]
                        new_size = current_size + len(chunk_data)
                        split_file[dataset_name].resize((new_size,) + split_file[dataset_name].shape[1:])  # Resize with full shape tuple
                        split_file[dataset_name][current_size:new_size, ...] = chunk_data  # Write the data chunk

# Write data to each split
print("Writing to train files...")
write_to_split_files(train_files, samples_train, datasets_to_copy)
for f in train_files:
    f.close()

print("Writing to validation files...")
write_to_split_files(val_files, samples_val, datasets_to_copy)
for f in val_files:
    f.close()

print("Writing to test files...")
print(test_files)
print(datasets_to_copy)
write_to_split_files(test_files, samples_test, datasets_to_copy)
for f in test_files:
    f.close()

print("Data split completed successfully!")
