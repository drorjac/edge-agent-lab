# Data card

## UCI HAR: primary development dataset

Source: UCI, DOI [10.24432/C54S4K](https://doi.org/10.24432/C54S4K), CC BY 4.0.
Thirty volunteers aged 19–48 performed six activities with a waist-mounted
smartphone. Published inputs are preprocessed 128-sample windows at 50 Hz, with
50% overlap. Accelerometer and gyroscope are measured; body acceleration is a
processed component. This repository did not collect these recordings.

The archive is pinned by SHA256. Derived arrays and split metadata receive their
own checksums. The official nine test subjects are preserved. Validation uses
subjects 5, 15, 19, and 26. Counts and subject lists appear in the committed
[manifest](../reports/data_manifest.json).

Known limits include a small participant cohort, controlled activities,
preprocessing before publication, one smartphone configuration, correlated
windows, and limited deployment settings. Subject isolation reduces identity
leakage but does not solve device or population shift.

## Procedural simulation

The simulator generates periodic motion, simple gravity vectors, gyroscope
oscillations, and low-motion postures. Optional frequency, amplitude, noise, and
orientation variation implement domain randomization. Parameters are handwritten
and never fitted to test recordings. Simulated subjects use sentinel ID `-1`.
The simulator is an approximate learning prior, not a validated physical model.

## Synthetic corruptions of real recordings

Noise, gain changes, z-axis rotation, and gyroscope dropout transform copies of
measured arrays. They test specific sensitivities. Labels are assumed unchanged;
that assumption can fail for extreme transformations. These arrays are not
independently collected real-world samples.

## External transfer probe

When enabled, the HAR70+ probe uses independent lower-back accelerometer
recordings from older adults. It maps overlapping activity labels and fills the
absent gyroscope with zeros. Different population, placement, preprocessing,
label policy, and missing modalities make this an intentionally difficult
cross-dataset probe, not a directly comparable benchmark or clinical validation.
All adaptation steps and dropped windows are recorded. It never tunes model
selection or the primary training protocol.
