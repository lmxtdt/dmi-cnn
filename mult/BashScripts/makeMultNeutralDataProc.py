# -*- coding: utf-8 -*-

file = """#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=24g
#SBATCH --mail-type=END  
#SBATCH --mail-user=thoms352@umn.edu
#SBATCH --output={abbr}{num}.out
#SBATCH --error={abbr}{num}.err

cd ~/DMI || exit

((rep = 0))

for j in {{1..{jmax}}}
    do ../build/slim -d "numPairs = 1" \\
                    -d "simInfoOutPath = '{simInfoOut}.csv'" \\
                    -d "ancPopOutPath = '{ancPopOut}${{rep}}.csv'" \\
                    -d "ancIndOutPath = '{ancIndOut}${{rep}}.txt'" \\
                    -d "saveIndData = {saveIndData}" \\
                    -d "fitOutPath = '{fitOut}${{rep}}.csv'" \\
                    -d "posOutPath = '{posOut}${{rep}}.csv'" \\
                    -d "neutral = 1" \\
                    multDMI.slim
    ((rep += 1))
done

source envml/bin/activate || exit

python3 procMultData.py {inCSV}.csv {outNPZ}

deactivate

"""

abbr = "multNeutralDataProc"
#runs simultations and also processes their output into .npz files

jmax = 1200 #number of simulations to do


num = 1 #11 #number to start numbering files at

saveIndData = 0 #whether to save individual-level ancestry

#produce however many files that are basically identical
for i in range(10):
    numFormatted = "{:02}".format(num)
    
    #various output paths
    simInfoOut = "multNeutralSimInfo/s_{}".format(numFormatted)
    ancPopOut = "multNeutralAncPopInfo/a_p_{}_".format(numFormatted)
    ancIndOut = "multNeutralAncIndInfo/a_i_{}_".format(numFormatted)
    fitOut = "multNeutralFit/f_{}_".format(numFormatted)
    posOut = "multNeutralPos/p_{}_".format(numFormatted)
    
    outNPZ = "multNeutralNpz/n_{}.npz".format(numFormatted)

    #write the file and save it
    contents = file.format(abbr = abbr,
                           num = numFormatted,
                           jmax = jmax,
                           simInfoOut = simInfoOut,
                           ancPopOut = ancPopOut,
                           ancIndOut = ancIndOut,
                           saveIndData = saveIndData,
                           fitOut = fitOut,
                           posOut = posOut,
                           inCSV = simInfoOut,
                           outNPZ = outNPZ
                           )

    with open("{}{}.sh".format(abbr, numFormatted), "w") as fn:
            fn.write(contents)
            fn.close()

    #advance
    num += 1


print(num)
