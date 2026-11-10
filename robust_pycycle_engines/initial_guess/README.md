# Compressor pressure ratios

The overall pressure ratio `opr` is an input, and the axial HPC supplies whatever the other stages
do not:

```
turbofan:   pi_hpc = opr / (pi_fan * pi_lpc * pi_centrif)
turboshaft: pi_hpc = opr / (pi_lpc * pi_centrif)
```

`pi_centrif` is the pressure ratio of a centrifugal compressor after the axial HPC, with polytropic
efficiency `eta_pcentrif` (default 0.85). It defaults to 1, meaning there is no centrifugal stage;
likewise an `opr` that leaves `pi_hpc = 1` means there is no axial HPC. A stage with a pressure
ratio of 1 does no work, and an `opr` below the product of the other stages raises an error. The
cycle models take their pressure ratios from the initial guess, so the two always agree, and model a
stage in pyCycle only when its pressure ratio is above 1. The turbofan model needs an axial HPC,
and the turboshaft model needs at least one HP compressor; without an LPC it also has no LPT.
