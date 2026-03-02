# Relaxometry on *ilumr*
Relaxometry, or Time-Domain NMR (TD-NMR), is the measurement and analysis of the different rates of relaxation in a sample. Since relaxation rates are affected by local molecular environments, structural environments, and diffusion, analysing the complex patterns of relaxation can give us information about a sample's molecular composition, macroscopic structures, and fluid transportation mechanisms.

**This repository contains Jupyter notebooks and the associated python pulse sequence files to run relaxometry experiments on [*ilumr*](https://www.resonint.com/ilumr) either programmatically or using a graphical user interface (GUI)**

For more information on relaxometry, head over to our [relaxometry blog post](https://www.resonint.com/post/relaxometry-on-ilumr) to look through our gallery of T1-T2 correlation maps and read our application note about using *ilumr* to characterise the T1, T2, and diffusion properties of samples for MRI phantom development.

## T1-T2 IRCPMG
The IRCPMG (Inversion Recovery Carr-Purcell-Meiboom-Gill) notebooks create plots of a sample's T1-T2 correlation map, T1 Inversion Recovery curve, and T2 Decay curve. 

`T1-T2 IRCPMG dev.ipynb` allows you to run the experiment programmatically. 
- Parameters are edited within the code blocks.
- Each block of code can be executed by clicking the box and pressing SHIFT+ENTER.
- Code blocks can be added for additional data processing and plotting.
  
`T1-T2 IRCPMG.ipynb` allows you to run the experiment through a GUI. 
- The experiment is set up using the [*matipo* Experiment Library](https://resonint-matipo-python.readthedocs-hosted.com/en/latest/api.html#experiment-building-tools).
- GUI is loaded by selecting **Run>Run all Cells** in the notebook.
- Parameters are altered using user input boxes.
- Experiment begins when the **Run** button is pressed.
- Plots are generated automatically.

To run these notebooks you will need to download `loglogmap_plot.py`. The notebooks use the IRCPMG pulse sequence in your *ilumr* system directory.

## T1-T2 SRCPMG
The SRCPMG (Saturation Recovery Carr-Purcell-Meiboom-Gill) notebooks create plots of a sample's T1-T2 correlation map, T1 Saturation Recovery curve, and T2 Decay curve.

`T1-T2 SRCPMG dev.ipynb` allows you to run the experiment programmatically.

`T1-T2 SRCPMG.ipynb` allows you to run the experiment through a GUI.

To run these notebooks you will need to download `loglogmap_plot.py`, `SRCPMG.py`, and `ilt_SRCPMG.py`. The additional files are required because *ilumr* is not shipped with SRCPMG in its system directory. 

## D-T2 PGSE_CPMG
The PGSE CPMG (Pulsed Gradient Spin Echo Carr-Purcell-Meiboom-Gill) notebook creates plots of a sample's Diffusion-T2 correlation map, diffusion attenuation curve, and T2 Decay curve.

`D-T2 PGSE_CPMG.ipynb` allows you to run the experiment programmatically.

To run this notebook you will need to download `PGSE_CPMG.py`. This additional file is required because *ilumr* is not shipped with PGSE_CPMG in its system directory. 
