# Making MRI Music
The beeping sounds made by an MRI system are caused by the current in the gradient coils switching in the presence of a magnetic field. By controlling the frequency at which these gradients switch, we can control the pitch of the sounds. 
This notebook and pulse sequence can be used to play short tunes on the [*ilumr* benchtop MRI system](https://www.resonint.com/ilumr).

The Jupyter notebook, **music_notebook.ipynb**, is used to execute the **notes.py** pulse sequence program.

The python pulse sequence, **notes.py**, allows you to input an array of paired values (tuples) which specify the frequency and beats of each note you want to play. The "note" function then converts these values into gradient commands to create square waves.

You can read our [blog post](https://www.resonint.com/post/making-music-with-an-mri-system) to learn more about creating your own music on your *ilumr* MRI System.
