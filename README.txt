These are the replication files for 'Central Bank Communication and Higher Moments'.

All raw data is found in the Data folder.
	Including text data for Bank of England communications, financial markets data, and survey data.

All code for producing data, including cleaning of the raw data, is found in the Code folder.
	The Code folder takes the data from the Data folder and deposits it in the Output folder.
	The code is ordered in folders 1-6 and should be run in that order.
	All code is in python. The packages needed are imported at the beginning of each script.
	I was using python version 3.6

The Output folder contains all the information needed to produce the results in the paper.
	For those who want to use the topic distributions, for example, they needn't run the LDA process from the Code folder, they can simply find the results in the Output folder.

The DataVis folder takes the processed data from the Output folder, and creates the visualisations in the paper.
	All figures can be replicated by opening the DataVis folder, and navigating to the appropriate folder, opening the notebook and running it. The figure will then be saved in the \Figs directory.

In general, at the start of nearly all the scripts and notebooks, there is an option to change the working directory. The user should change this to where they have saved this replication folder.
