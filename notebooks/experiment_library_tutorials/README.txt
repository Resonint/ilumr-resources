# Experiment Library Tutorials

The matipo Experiment Library is designed to make it easier to set up GUI's for MRI experiments in Jupyter notebooks. You can access the API reference for the library here: [Read the Docs](https://resonint-matipo-python.readthedocs-hosted.com/en/latest/api.html#experiment-building-tools)

Some key features of the experiment library are:

- State Preservation: The state of notebook experiments' inputs and plots is saved and loaded automatically, so you can resume where you left off.
- Auto-generated Input Widgets: Easily create input widgets for pulse sequence parameters.
- Interactive Plots
- Progress Bar: Display the estimated time a pulse sequence will take to run and updates the time as the experiment progresses. 
- Auto Layout: Plots, buttons, user inputs, and the progress bar are laid out automatically. 
- Data Saving: Notebook experiments are equipped with a "save" button that saves the sequence data and parameters to a hdf5 file. The "auto save" toggle button allows for automatic data saving each time the experiment runs. 
- Debugging and Error Reporting: Status logs and error messages with full tracebacks are now displayed in the new status card. 


## Experiment Class Tutorial 

This tutorial covers:
 
- Creating a custom experiment class
- Loading a pulse sequence
- Adding plots (Signal and Spectrum)
- Updating pulse sequence parameters
- Adding parameter input widgets
- Printing to the status box
- Customising the experiment and workspace name

