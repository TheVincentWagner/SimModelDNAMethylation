#!/usr/bin/env python
# coding: utf-8

### This script performs a parameter estimation for our full simulation model (Model 2) of DNMT1 mediated DNA methylation 
### It was executed on a Linux based calculation cluster with 20 parallel processes and is written in Python 3. 

## Imports

import math
import sys
import numpy as np
import pandas as pd
import matplotlib
from scipy.linalg import solve
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from numpy import genfromtxt
import os.path
from matplotlib import cm
from matplotlib.cm import bone
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool
import csv


## Load all measured data stored at "measurements/i.csv" for i = 0,1,2,3 and calculate mean and variance of each 

def preprocMeasurements():

    # create a suitable container
    measurements = np.zeros((45,4))

    # store first and second moments lateron in here
    firstMomMeas = np.zeros(4)
    seconMomMeas = np.zeros(4)

    # loop over all datasets
    for i in range(4):
        measurementFileName = 'measurements/' + str(i) + '.csv' # define filename
        measurements[:,i] = np.reshape(pd.read_csv(measurementFileName,names=['col']).values,(45)) # read using pandas

        # negative values are set to 0   
        for j in range(45):
            if (measurements[j,i] < 0):
                measurements[j,i] = 0

        # Caculate the moments of the loaded distributions
        for j in range(45):
            firstMomMeas[i] = firstMomMeas[i] + j*measurements[j,i]

        for j in range(45):
            seconMomMeas[i] = seconMomMeas[i] + ((j-firstMomMeas[i])**2)*measurements[j,i]

    # retrun the moments
    return firstMomMeas, seconMomMeas, measurements

## Define the L2 Error Function

def L2ErrorFunc(methylations, experimentNo):

    # calculate the methylation histogram 
    methylHist, edges = np.histogram(methylations, np.linspace(0,45,46)-0.5, density=True)    # create an histogram of the data

    # container for L2 error
    L2 = 0
    
    # loop and sum over all histogram bins
    for j in range(45):
        L2 = L2 + (methylHist[j]-measurements[j,experimentNo])**2
        
    return L2


## Perform one methylation for a given DNA molecule "DNAmol" according to the site affinity vector "siteAffinity" 

def performMethylation(DNAmol,methylationsBool,DNAAffinity):
    
    invMethylationsBool = (-1)*(methylationsBool[DNAmol,:]-1) # vector with 0 for methylated and 1 for free sites

    specSiteAffinity = invMethylationsBool*siteAffinity # scalar product of site affinities and the just defined indexes
 
    methylRand = np.random.rand(1)*DNAAffinity[DNAmol] # random number to determine the methylation site via the look up method

    # The following section is a halfway efficient implementation of the look-up method for 44 possible outcomes

    sSAquarters = np.array([np.sum(specSiteAffinity[0:11]),np.sum(specSiteAffinity[11:22]),np.sum(specSiteAffinity[22:33]),np.sum(specSiteAffinity[33:44])])

    if (methylRand <= sSAquarters[0]):      # Check if it is one of the first 11 sites
        kStart = 0
        cumsSA = 0
        
    elif (methylRand <= sSAquarters[0] + sSAquarters[1]): # If not, check if it is one of the second 11 sites
        kStart = 11
        cumsSA = sSAquarters[0]
        
    elif (methylRand <= sSAquarters[0] + sSAquarters[1] + sSAquarters[2]): # ...
        kStart = 22
        cumsSA = sSAquarters[0] + sSAquarters[1]
        
    else:                                   # ...
        kStart = 33
        cumsSA = sSAquarters[0] + sSAquarters[1] + sSAquarters[2]
           
    k = kStart # set k to the just determined start index "kStart"
    
    while (k <= kStart+11):
        cumsSA = cumsSA + specSiteAffinity[k] # sum up the site affinities of still unmethylated sites until...

        if (methylRand <= cumsSA): # ...the sum is larger than the drawn random number

            methylationsBool[DNAmol,k] = 1 # mark site "k" at DNA molecule "DNAmol" as methylated
            DNAAffinity[DNAmol] = DNAAffinity[DNAmol] - siteAffinity[k] # reduce the DNA's overall methylation affinity by the site specific affinity of site "k"
            k = 50 # exit condition
            
        k = k+1
    
    return methylationsBool, DNAAffinity


## Execute one time step of the gillespie simulation 

def timeStep(time, state, DNAmol, methylationsBool, parameters, numDNA, DNAAffinity, procCounter, distrCounter):
    
    reacRand = np.random.rand(1)                             # random number to determine the reaction happening
    timeRand = np.random.rand(1)                             # random number to determine the time the specific reaction takes
    
    if (state == 0):                                         # If DNMT1 is undocked and open ...
        coeffSum = parameters[0]
        
        state = 1                                            # ... it can only dock ...
        DNAmol = int(np.round(np.random.rand(1)*numDNA-0.5)) # ... at a certain DNA molceule.
    
    elif (state == 1):                                       # If DNMT1 is docked and open, ...
        coeffSum = parameters[1] + parameters[2] + parameters[4]*DNAAffinity[DNAmol]
        
        if (reacRand <= (parameters[1]/coeffSum)):           
            state = 0                                        # it can either undock ...
            DNAmol = numDNA                                  # ... from its DNA ...
            
        elif (reacRand <= ((parameters[1]+parameters[2])/coeffSum)):
            state = 2                                        # ... or close ...
            
        else:         
            methylationsBool, DNAAffinity = performMethylation(DNAmol,methylationsBool,DNAAffinity) # ... or perform a  methylation ...
            distrCounter[DNAmol] = distrCounter[DNAmol] + 1  # ... in a distributive manner and ...
            state = 0                                        # ... undock ...
            DNAmol = numDNA                                  # ... from its DNA.
    
    elif (state == 2):                                       # If DNMT1 is closed and bound ...
        coeffSum = parameters[3] + parameters[5]*DNAAffinity[DNAmol]
        
        if (reacRand <= (parameters[3]/coeffSum)):
            state = 1                                        # ... it can either open again ...
        else:
            methylationsBool, DNAAffinity = performMethylation(DNAmol,methylationsBool,DNAAffinity) # ... or methylate ...
            procCounter[DNAmol] = procCounter[DNAmol] + 1    # ... in a processive manner.
            
    time = time + np.log(1/timeRand)/coeffSum                # Update the time
        
    return time, state, DNAmol, methylationsBool, DNAAffinity, procCounter, distrCounter


## Set-up for 4 Gillespie simulations each corresponding to one experimental data set  

def gillespieSim(logParameters):
    
    np.random.seed()                                 # Seed random number generator
    
    parameters = np.exp(logParameters)               # transformation from log-parameter-space to physical parameter-space
    
    sumL2Err = 0                                     # initialize error
    
    numDNA = 1000                                    # initialite 1000 DNA molecules for optimization
    
    numDNMT1 = int(numDNA*4.0)                       # number of DNMT1 molecules
    expTimes = [60.0,180.0,600.0,1800.0]             # experiment durations
    
    methylations = np.zeros((numDNA,4))              # one column for each experiment and one row for each DNA molecule
    methylationsBool = np.zeros((numDNA,44), dtype=int)  # store all methylated sites for each DNA molecule
    
    procCounter = np.zeros(numDNA)                   # counters to indicate the number of processive ...
    distrCounter = np.zeros(numDNA)                  # ... or distributive methylations of every DNA strand
    
    DNAAffinity = np.ones(numDNA)                    # the free sites of a DNA molecule define this methylation affinity 
    
    DNMT1Times = np.zeros(numDNMT1)                  # containers to store the time ...
    DNMT1States = np.zeros(numDNMT1, dtype=int)      # ... , state ...
    DNMT1DNAmol = np.ones(numDNMT1, dtype=int)*numDNA  # ... and the DNA strand the protein is docked to for every DNMT1

    for expStep in range(4):
        
        for dnmt1 in range(numDNMT1):
            time = DNMT1Times[dnmt1]                 # set time ...
            state = DNMT1States[dnmt1]               # ... and state ...
            DNAmol = DNMT1DNAmol[dnmt1]              # ... and DNA-molecule to where we were before the visualization step.
            
            while (time < expTimes[expStep]):        # Execute time steps till desired simulation time is reached.
                time,state,DNAmol,methylationsBool,DNAAffinity,procCounter,distrCounter = timeStep(time,state,DNAmol,methylationsBool,parameters,numDNA,DNAAffinity,procCounter,distrCounter)
            
            DNMT1Times[dnmt1] = time                 # Store time, ...
            DNMT1States[dnmt1] = state               # ... state,...
            DNMT1DNAmol[dnmt1] = DNAmol              # ... and DNA molecule the current DNMT1 is docked to. 
                
        methylations[:,expStep] = np.sum(methylationsBool,1) # compute methylation state 
                
        L2 = L2ErrorFunc(methylations[:,expStep], expStep) # compute the error...
        sumL2Err = sumL2Err + L2                     # ... and sum it up

    return sumL2Err


## Global optimization function based on sparse grids to minimize for smallest L2 error in parrallel

def iterOpt(startLogParameter,numIterations):

    np.random.seed() # Seed random for each process
               
    bestLogParamsAndErrors = np.zeros((numIterations,7)) # Container to store the best parameter set and corresponding error for each iteration

    # initialize first parameter set
    currP = startLogParameter                         # initialize the startparameter as first guess
    currE = 1000                                      # set the error high enough to surely update it after one iteration
    allE = np.zeros(39)                               # container to store all error of this iteration
    
    f = 0.33                                          # define the quantity that itself defines the size of the sparse grid ...
    g = f*2.0/3.0                                     # ... and two related values ...
    h = f*4.0/3.0                                     # ...
    
    # loop over all optimization iterations
    for i in range(numIterations):
        
        # choose 3 different parameter dimensions randomly to build the sparse grid
        dim0 = int(np.round(np.random.rand(1)*6-0.5))
        dim1 = dim0
        dim2 = dim0
        
        # make sure that they are indeed different
        while(dim1 == dim0):
            dim1 = int(np.round(np.random.rand(1)*6-0.5))

        while((dim2 == dim1) or (dim2 == dim0)):
            dim2 = int(np.round(np.random.rand(1)*6-0.5))
            
        # print the resulting dimensions
        print("DIMS: " + str(dim2) + " " + str(dim1) + " " + str(dim0))
        
        # create new parameters sets varied in the three just drawn random dimensions from the originl one in a sparse grid way
        p0 = np.copy(currP)
        p1 = np.copy(currP)
        p2 = np.copy(currP)
        p3 = np.copy(currP)
        p4 = np.copy(currP)
        p5 = np.copy(currP)
        p6 = np.copy(currP)
        p7 = np.copy(currP)
        p8 = np.copy(currP)
        p9 = np.copy(currP)
        p10 = np.copy(currP)
        p11 = np.copy(currP)
        p12 = np.copy(currP)
        p13 = np.copy(currP)
        p14 = np.copy(currP)
        p15 = np.copy(currP)
        p16 = np.copy(currP)
        p17 = np.copy(currP)
        p18 = np.copy(currP)
        p19 = np.copy(currP)
        p20 = np.copy(currP)
        p21 = np.copy(currP)
        p22 = np.copy(currP)
        p23 = np.copy(currP)
        p24 = np.copy(currP)
        p25 = np.copy(currP)
        p26 = np.copy(currP)
        p27 = np.copy(currP)
        p28 = np.copy(currP)
        p29 = np.copy(currP)
        p30 = np.copy(currP)
        p31 = np.copy(currP)
        p32 = np.copy(currP)
        p33 = np.copy(currP)
        p34 = np.copy(currP)
        p35 = np.copy(currP)
        p36 = np.copy(currP)
        p37 = np.copy(currP)
        p38 = np.copy(currP)

        
        # 8 3D corners
        p0[dim0] = p0[dim0]+f
        p0[dim1] = p0[dim1]+f
        p0[dim2] = p0[dim2]+f

        p1[dim0] = p1[dim0]-f
        p1[dim1] = p1[dim1]-f
        p1[dim2] = p1[dim2]-f

        p2[dim0] = p2[dim0]+f
        p2[dim1] = p2[dim1]-f
        p2[dim2] = p2[dim2]-f

        p3[dim0] = p3[dim0]-f
        p3[dim1] = p3[dim1]+f
        p3[dim2] = p3[dim2]+f
        
        p4[dim0] = p4[dim0]+f
        p4[dim1] = p4[dim1]+f
        p4[dim2] = p4[dim2]-f

        p5[dim0] = p5[dim0]-f
        p5[dim1] = p5[dim1]-f
        p5[dim2] = p5[dim2]+f

        p6[dim0] = p6[dim0]+f
        p6[dim1] = p6[dim1]-f
        p6[dim2] = p6[dim2]+f

        p7[dim0] = p7[dim0]-f
        p7[dim1] = p7[dim1]+f
        p7[dim2] = p7[dim2]-f
        
        # 12 inner 2D corners
        p8[dim0] = p8[dim0]+g
        p8[dim1] = p8[dim1]+g

        p9[dim0] = p9[dim0]-g
        p9[dim1] = p9[dim1]-g

        p10[dim0] = p10[dim0]+g
        p10[dim1] = p10[dim1]-g

        p11[dim0] = p11[dim0]-g
        p11[dim1] = p11[dim1]+g

        p12[dim2] = p12[dim2]+g
        p12[dim1] = p12[dim1]+g

        p13[dim2] = p13[dim2]-g
        p13[dim1] = p13[dim1]-g

        p14[dim2] = p14[dim2]+g
        p14[dim1] = p14[dim1]-g

        p15[dim2] = p15[dim2]-g
        p15[dim1] = p15[dim1]+g

        p16[dim0] = p16[dim0]+g
        p16[dim2] = p16[dim2]+g

        p17[dim0] = p17[dim0]-g
        p17[dim2] = p17[dim2]-g

        p18[dim0] = p18[dim0]+g
        p18[dim2] = p18[dim2]-g

        p19[dim0] = p19[dim0]-g
        p19[dim2] = p19[dim2]+g
 
        # 12 outer 2D corners
        p20[dim0] = p20[dim0]+h
        p20[dim1] = p20[dim1]+h

        p21[dim0] = p21[dim0]-h
        p21[dim1] = p21[dim1]-h

        p22[dim0] = p22[dim0]+h
        p22[dim1] = p22[dim1]-h
        
        p23[dim0] = p23[dim0]-h
        p23[dim1] = p23[dim1]+h

        p24[dim2] = p24[dim2]+h
        p24[dim1] = p24[dim1]+h

        p25[dim2] = p25[dim2]-h
        p25[dim1] = p25[dim1]-h

        p26[dim2] = p26[dim2]+h
        p26[dim1] = p26[dim1]-h

        p27[dim2] = p27[dim2]-h
        p27[dim1] = p27[dim1]+h

        p28[dim0] = p28[dim0]+h
        p28[dim2] = p28[dim2]+h

        p29[dim0] = p29[dim0]-h
        p29[dim2] = p29[dim2]-h

        p30[dim0] = p30[dim0]+h
        p30[dim2] = p30[dim2]-h

        p31[dim0] = p31[dim0]-h
        p31[dim2] = p31[dim2]+h 

        # 6 1D "corners"   
        p32[dim0] = p32[dim0]+f
        p33[dim0] = p33[dim0]-f
        p34[dim1] = p34[dim1]+f
        p35[dim1] = p35[dim1]-f
        p36[dim2] = p36[dim2]+f
        p37[dim2] = p37[dim2]-f
        
        # p38 stays the origiaal parameter set at the center of the sparse grid
        
        
        # stack all parameter sets togetether
        parLogParameters = np.stack((p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,p36,p37,p38),axis = 0)
        
        # the system we used was equipped with 20 cores per node
        if __name__ == '__main__':
            pool = Pool(processes = 20)
            # run "gillespieSim" with all just varied parametersets in parallel using "pool.map"
            result = pool.map(gillespieSim, parLogParameters) 
            pool.close()
            pool.join()

        # loop over all 39 simulation results
        for p in range(39):
            allE[p] = result[p]                   # store all errors
                
        currE = np.min(allE)                      # take the minimal one
        currP = parLogParameters[np.argmin(allE)] # update the current parameter set to be the one corresoponding to best error
        
        bestLogParamsAndErrors[i,0:6] = currP     # keep best parameter set ...
        bestLogParamsAndErrors[i,6] = currE       # ... and corresponding error.

        # print best error and corresponding parameter set
        print(" ")
        print("Best Parameters: " + str(np.exp(currP)))
        print("Current Error: " + str(currE))
        
        
        # the next sparse grid will be smaller in extend
        f = f*0.995
        g = f*2.0/3.0
        h = f*4.0/3.0

        # print the new factor that determines the sparse grid's size
        print("Current factor: " + str(f))
    
    return bestLogParamsAndErrors

## Definition of our optimization contraint

def methylRatesConstraint(logParameters):
    return logParameters[5]-logParameters[4]

## Wrapper for "fmin_cobyla" with contraints

def localOpt(logParameters):
    locOpt = optimize.fmin_cobyla(gillespieSim, logParameters, [methylRatesConstraint])    
    return locOpt

## Find parameter sets from the history of good parameter sets that good and diverse at the same time

def findBestDiverseLogParams(numIterations):
    globalOpt = np.loadtxt('GlobalOptFull.csv', delimiter = ',') # load the parameters resulting from global optimization
    logParams = globalOpt[:,0:6]                                 # divide the file into parameters ...
    errors = globalOpt[:,6]                                      # ... and corresponding errors.

    # print the just loaded values
    print("Best Parametersets throughout global Optimization: ")
    print(np.exp(logParams))
    print("Corresponding Errors: ")
    print(errors)

    # copy the error to an value we can update
    udErrors = np.copy(errors)
    
    # since global optimization did not have our constraint k_d <= k_p, we enforce it here ...
    for p in range(numIterations):
        if (logParams[p,4] > logParams[p,5]):
            udErrors[p] = udErrors[p] + 1000.0                   # ... and punish parameter sets that do not fulfil it 

    numFinalParams = 20                                          # determine how many parametersets this function will return

    finalLogParams = np.zeros((numFinalParams,6))                # create the needed containers
    finalErrors = np.zeros(numFinalParams)                       # ...

    # Iteratively choose the best parameterset and punish similar parametersets according to their similarity afterwards
    for n in range(numFinalParams):                              # loop oevr all parameter sets
        minimum = np.min(udErrors)                               # find the minimium error ...
        argmin = np.argmin(udErrors)                             # ... and its corresponding parameret set
    
        finalLogParams[n,:] = logParams[argmin,:]                # store paramters ...
        finalErrors[n] = errors[argmin]                          # ... and error
    
        for p in range(numIterations):                           # loop over all parameter sets to correct the error
        
            quotVec = finalLogParams[n,:]/logParams[p,:]         # take the element-wise quotient of both parameter sets
        
            similarityF = 100                                    # weighting factor 
        
            # devide by the element of the quotient if > 1, multiply otherwise for all elements
            for i in range(6):
                if (quotVec[i] > 1):
                    similarityF = similarityF/quotVec[i]
                else:
                    similarityF = similarityF*quotVec[i]
                
            udErrors[p] = udErrors[p] + similarityF              # add the just calculated modifier to the error
        
    # print the parameter set that will be used as a starting point for local optimization
    print("Parameterssets for local optimization: ")
    print(np.exp(finalLogParams))

    print("Corresponding Errors")
    print(finalErrors)
    
    return finalLogParams

# Local optimization function in parallel 

def localParallelOpt(finalLogParams):

    print('Local Optimization Started')
    if __name__ == '__main__':
        pool = Pool(processes = 20)
        # call "localOpt" in parallel with "pool.map"
        finalOpt = pool.map(localOpt, finalLogParams)
        pool.close()
        pool.join()
    
    # save the results in a csv file
    np.savetxt('LocalOptFull.csv', finalOpt, delimiter=',')
    print(finalOpt)
    return finalOpt


### The main script
# load the preprocessed experimental data

print("Full Model Optimization Script")

# Call preprocessing script 
firstMomMeas, seconMomMeas, measurements = preprocMeasurements()
    
# measurement values indicating the relative specific methylation probability of every CpG site
siteAffinity = np.array([0.020951302,0.03158065,0.0058243,0.006972982,0.019074582,0.051577415,0.039314027,
                         0.005371299,0.039993529,0.009189452,0.021371946,0.016356577,0.013848892,0.041158389,
                         0.017683223,0.010338133,0.01760233,0.037049021,0.013881248,0.02960686,0.036369519,
                         0.026435852,0.012732568,0.011260314,0.015240252,0.012231031,0.014156285,0.012845818,
                         0.030351076,0.064196732,0.018783368,0.008542307,0.048649085,0.016890471,0.012813461,
                         0.043520466,0.035738554,0.015175538,0.0170199,0.014965216,0.01585504,0.032761689,
                         0.024623847,0.010095454])


# a good initial parameter guess based on our experience and expert knowledge
logParamInit = np.log(np.array([1.27,3.74,0.0726,0.001,0.0115,0.016]))

# setting the number of global optimization steps
numGlobalIterations = 300

# Call global optimization script and store the results in a csv file
globalOpt = iterOpt(logParamInit,numGlobalIterations)
np.savetxt('GlobalOptFull.csv', globalOpt, delimiter=',')

# Extract a good and at the same time diverse parameters set from the history of global optimitzation
finalParams = findBestDiverseLogParams(numGlobalIterations)

# Call local optimization script for the good and diverse parameter set
finalOpt = localParallelOpt(finalParams)

# Save the final resulting parameters in a csv file
np.savetxt('FinalOptFull.csv', finalOpt, delimiter=',')

# Run the "gillespieSimulation" function for each local minimum in parallel
if __name__ == '__main__':
        pool = Pool(processes = 20)
        finalOptErrors = pool.map(gillespieSim, finalOpt)
        pool.close()
        pool.join()

# Also store the errors corresponding to the local optima     
np.savetxt('FinalOptErrorsFull.csv', finalOptErrors, delimiter=',')
