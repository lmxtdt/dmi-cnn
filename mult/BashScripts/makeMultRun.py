# -*- coding: utf-8 -*-

file = """#!/bin/bash -l
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --mail-type=END  
#SBATCH --mail-user=thoms352@umn.edu
#SBATCH --output={abbr}{num}.out
#SBATCH --error={abbr}{num}.err

cd ~/DMI || exit
source envml/bin/activate || exit

python3 PredictRegionsIOU.py \\
    "{trainGlob}" "{valGlob}" "{neutralGlob}" \\
    "{modelPath}" {version} {load} \\
    {kernel1} {kernel2} {filters1} {filters2} \\
    {epochs} \\
    {excludeStats} {loss}

deactivate

"""
abbr = "multRun"

num = 181 #199

#fixed stuff
#glob strings for .npz files for the different datasets
trainGlob = "multNpzTrain/n_1*.npz"
valGlob = "multNpzVal/n_06.npz"
#neutralGlob = "multNeutralNpz/*.npz"
neutralGlob = "multNpz/n_02.npz"

load = 1 #whether to load an old model & continue training it
filters1 = 32 #number filters in first few layers
epochs = 2  #number epochs to train
version = 2 #version of model to load;
            #1 less than the version that will be saved

loss = "mse"

excludeStats = 1
for kernel1 in [11, 19, 29]:
    for kernel2 in [5, 11, 19]:
        for filters2 in [32, 64]:
            numFormatted = "{:03}".format(num)
            
            modelPath = "multModelsIOU/m_k{:02}_k{:02}_f{:02}_f{:02}_e{}".format(
                            kernel1,
                            kernel2,
                            filters1,
                            filters2,
                            excludeStats)

            #or something
            contents = file.format(abbr = abbr,
                                   num = numFormatted,
                                   trainGlob = trainGlob,
                                   valGlob = valGlob,
                                   neutralGlob = neutralGlob,
                                   modelPath = modelPath,
                                   version = version,
                                   load = load,
                                   kernel1 = kernel1,
                                   kernel2 = kernel2,
                                   filters1 = filters1,
                                   filters2 = filters2,
                                   epochs = epochs,
                                   excludeStats = excludeStats,
                                   loss = loss
                                   )

            with open("{}{}.sh".format(abbr, numFormatted), "w") as fn:
                    fn.write(contents)
                    fn.close()

            #advance
            num += 1

print(num)
