# CONCEPT: Using Electric Field calculations to approximate ground truth using Genetic Algorithms

We model neuron spikes into a function that takes the action potentials (spiking data).

- We model all spikes as reaching the same peak, plug this into a current dipole equation
- We approximate extracellular conductance as well as dipole moment scaling factor `k`
- We utilize genetic algorithms to closely fit findings in LFP data


## Inspiration

- [Dipole characterization of single neurons from their extracellular action potentials](https://pubmed.ncbi.nlm.nih.gov/21667156/)
- [Biophysically detailed forward modeling of the neural origin of EEG and MEG signals](https://www.sciencedirect.com/science/article/pii/S1053811920309526)
- https://em.geosci.xyz/content/maxwell1_fundamentals/dipole_sources_in_homogeneous_media/electric_dipole_definition/index.html