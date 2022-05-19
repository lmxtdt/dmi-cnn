Various Python Scripts.

**procMultData.py:** given a simInfo .csv file produced by SLiM simulations, processes data from all simulations mentioned in that file and outputs it to a .npz file.

**MultDataGen.py:** defines class(es) for data generators that feed data to the CNNs.

**PredictRegions.py:** trains and saves CNN that attempts to predict regions involved in incompatibilities via an output function that forms peaks.
**PredictRegions2.py:** as above, but with fewer layers and a different data generator to train CNN on the more severe incompatibilities. (note: not all that useful)
**PredictRegions3.py:** PredictRegions2.py, but with the normal data generator
**PredictRegionsIOU.py:** like PredictRegions.py, but with the option to train it using mean squared error ("mse") or the sum of intersection over union of the areas under the output curve ("sumIOU") (possibly inaccurately named)
**PredictRegionsMAE.py:** like PredictRegions3.py, but using mean absolute error to train the CNN

**plotMultPredIOU.py:** visualizes example input and output from a trained model. Given a saved model and input data to visualize, it outputs a CSV file and a figure showing the genotype frequencies along the chromosome, the model output, basic chi-squared analysis along the chromosome, and saliency (which isn't very useful for this model, but can be interesting for other CNNs).

**testMultClass.py:** evaluates the accuracy of model against chi-squared analysis using a few metrics. Prints output, does not save it.

Note on the different loss functions:
Training on MSE results in two possibilities: the CNN attempts to predict the output curve as desired; or the CNN only outputs 0 over the entire chromosome, which is useless. Training on IOU avoids the latter situation, training on MAE results into the latter situation. However, training with IOU may be overly sensitive, which is why the option to use MSE is also in the PredictRegionsIOU.py file: it may be initially trained on IOU and then re-loaded, re-compiled, and trained on MSE.