# -*- coding: utf-8 -*-

file = """#!/bin/bash -l
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=24g
#SBATCH --mail-type=END  
#SBATCH --mail-user=thoms352@umn.edu
#SBATCH --output={abbr}{num}.out
#SBATCH --error={abbr}{num}.err

cd ~/DMI || exit

((rep = 0))

for j in {{1..{jmax}}}
    do for nPairs in {{1..12}}
        do ../build/slim -d "numPairs = ${{nPairs}}" \\
                        -d "simInfoOutPath = '{simInfoOut}.csv'" \\
                        -d "ancPopOutPath = '{ancPopOut}${{rep}}.csv'" \\
                        -d "ancIndOutPath = '{ancIndOut}${{rep}}.txt'" \\
                        -d "saveIndData = {saveIndData}" \\
                        -d "fitOutPath = '{fitOut}${{rep}}.csv'" \\
                        -d "posOutPath = '{posOut}${{rep}}.csv'" \\
                        multDMI.slim
        ((rep += 1))
    done;
done

source envml/bin/activate || exit

python3 procMultData.py {inCSV}.csv {outNPZ}

deactivate

"""

abbr = "multDataProc"
#runs simulations and processes their data into .npz files

jmax = 100 #number simulations to run

num = 1 #51 #number to start numbering the files at

saveIndData = 0 #whether to save individual-level ancestry

#produce 50 nearly identical files
for i in range(50):
    numFormatted = "{:02}".format(num)

    #various output paths
    simInfoOut = "multSimInfo/s_{}".format(numFormatted)
    ancPopOut = "multAncPopInfo/a_p_{}_".format(numFormatted)
    ancIndOut = "multAncIndInfo/a_i_{}_".format(numFormatted)
    fitOut = "multFit/f_{}_".format(numFormatted)
    posOut = "multPos/p_{}_".format(numFormatted)
    outNPZ = "multNpz/n_{}.npz".format(numFormatted)

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
