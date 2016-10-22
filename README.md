# Jupyter Notebooks Related to Astronomical CCD Sensors

The following notebooks are available:
* Poisson: a pedagogical introduction to the electrostatics within a
CCD that leads to the buried channel where charge is stored.
* Fringing: a study of the mechanisms and phenomenology of fringing
in astronomical CCDs.
* LabData: utilities for studying lab sensor data with some results.

The LabData utilities are also available as a command-line script which
runs over a batch of FITS files to analyze them and produce plots, e.g.
```
./ccdplot --verbose itl.yaml
```
Processing and display is configured by options in YAML file. See comments
in this file for details on the available options.
