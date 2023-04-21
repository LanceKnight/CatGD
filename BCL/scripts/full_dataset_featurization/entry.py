import os

# Convert idx to sdf
os.system('python convert_idx_to_sdf.py')



# remove duplicates


# Filter out and neuralize molecule
os.system('python filter.py')

# add ids

# Create features
os.system('python scripts/data_preparation.py')
# os.system(f'python scripts/randomize.py')
