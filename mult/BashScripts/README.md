Generate bash scripts to queue up as jobs. Shows the command-line calls for the SLiM and Python scripts.

**makeMultDataProc.py:** produces bash scripts that runs SLiM simulations and then processes the output using procMultData.py
**makeMultNeutralDataProc.py:** same as above, but runs SLiM simulations without incompatibilities, which are used to normalize the CNNs
**makeMultRun.py:** trains the CNNs using PredictRegionsIOU.py (or others). Uses the output produced by the previous two scripts. The models are also saved and can be loaded to continue training, possibly with other parameters.