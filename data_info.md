# Data Information

The data is formatted in the *openNsx* format: https://vh-lab.github.io/NDR-matlab/reference/lib/NPMK/openNSx.m/

Checkout Blackrock Neurotech's NPMK Library for even more info: https://github.com/BlackrockNeurotech/NPMK/tree/master


Data is stored within the `data` folder in the following format

    data/
    │   .gitkeep
    │
    ├───patients
    │   ├───202014
    │   │       ns2eeg_202014.mat
    │   │
    │   ├───202016
    │   │       ns2eeg_202016.mat
    │   │
    │   └───202202
    │           ns2eeg_202202.mat
    │
    └───results
            seeg_results.csv

# Note

`seeg_results.csv` contains manually recorded values from running each of all 8 microwires per [Behnke Fried/Micro Inner Wire Bundle Electrodes](https://adtechmedical.com/sites/default/files/inline-files/AT10036-1-B%2C%20Rev.%20E%20BF%20and%20WB%20Depth%20Electrodes%20%28EU%29.pdf) (described as micro-contacts in documentation)

Each row per region represents a single micro-contact. 

TODO: Add indices required, preferably automate the training on Hippocampus and OFC using information on patient data. [Private for HIPPA reasons]

# IMPORTANT

Data is not accessible for public access. Please contact Leonardo.Ferrisi@utah.edu for further info.
                