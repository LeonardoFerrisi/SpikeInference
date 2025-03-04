# Data Information

The data is formatted in the *openNsx* format: https://vh-lab.github.io/NDR-matlab/reference/lib/NPMK/openNSx.m/

Checkout Blackrock Neurotech's NPMK Library for even more info: https://github.com/BlackrockNeurotech/NPMK/tree/master


Data is stored within the `data` folder in the following format

    data/
    │   .gitkeep
    │   fake_lfp_data.csv
    │   fake_lfp_data1.csv
    │   tree.txt
    │   
    └───actual_data/
        │   electrode.mat
        │   spikes_1k.mat
        │   spikes_30k.mat
        │   try_sEEG_Data.mat
        │   try_spikes_Data.mat
        │   unit.mat
        │   waveform.mat
        │   
        └───patient1/
                indices_set1.mat
                indices_set2.mat
                spikes_set2_1k.mat
                spike_times_set1_1k.mat
                spike_times_set1_30k.mat
                spike_times_set2_30k.mat
                try_sEEG_Data.mat


# IMPORTANT

Data is not accessible publically. Please contact Leonardo.Ferrisi@utah.edu for further info.
                