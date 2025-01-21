#!/bin/bash

# Author:       Alessio Cecconi
# Date:         2021/09/21
# Description:  This script launches the JanusDFGVariantAnalysisStarter
#               Run this launcher with "-h" to understand the meaning of options you can pass.

## Import the shell functions to create Regular Expressions expressing constraints
#. ./constraintsFunctions.cfg

## Clean up the screen
clear

##################################################################
##################################################################
## script variables
TEST_FOLDER="./tests-janus-VariantAnalysisDFG"
#TEST_BASE_NAME="VARIANT"
#
#mkdir ${TEST_FOLDER}"/"${TEST_BASE_NAME}
#TEST_FOLDER=${TEST_FOLDER}"/"${TEST_BASE_NAME}

## Runtime environment constants
JANUS_VARIANT_MAINCLASS="minerful.JanusDFGVariantAnalysisStarter"

# Variant parameters
LOG_VAR_1=${TEST_FOLDER}/"Sepsis(ageAbove70).xes"
LOG_VAR_2=${TEST_FOLDER}/"Sepsis(ageUnder35).xes"
VARIANT_RESULTS_CSV=${TEST_FOLDER}/"Sepsis-result.csv"
#LOG_VAR_1=${TEST_FOLDER}/"BPIC13_incidents_orgline_A2.xes"
#LOG_VAR_2=${TEST_FOLDER}/"BPIC13_incidents_orgline_C.xes"
#VARIANT_RESULTS_CSV=${TEST_FOLDER}/"BPIC13-result.csv"
#LOG_VAR_1=${TEST_FOLDER}/"BPIC15_1.xes"
#LOG_VAR_2=${TEST_FOLDER}/"BPIC15_2.xes"
#VARIANT_RESULTS_CSV=${TEST_FOLDER}/"BPIC15-result.csv"
#LOG_VAR_1=${TEST_FOLDER}/"BPIC15_1f.xes"
#LOG_VAR_2=${TEST_FOLDER}/"BPIC15_2f.xes"
#VARIANT_RESULTS_CSV=${TEST_FOLDER}/"BPIC15f-result.csv"
#LOG_VAR_1=${TEST_FOLDER}/"Road_teraffic_fineamount_Above_and_equal50.xes"
#LOG_VAR_2=${TEST_FOLDER}/"Road_teraffic_fineamount_Below50.xes"
#VARIANT_RESULTS_CSV=${TEST_FOLDER}/"Road_teraffic-result.csv"
#LOG_VAR_1=${TEST_FOLDER}/"covid_most-freq-20_2019.xes"
#LOG_VAR_2=${TEST_FOLDER}/"covid_most-freq-20_2020.xes"
#VARIANT_RESULTS_CSV=${TEST_FOLDER}/"covid_most-freq-20-result.csv"

LOGS_ENCODING="xes"
P_VALUE=0.01              # p_value threshold to consider a difference statistically relevant
PERMUTATIONS=1000         # number of permutations of the permutation test
#NaN_LOG="-nanLogSkip" # NOT YET exposed in input
#NaN_LOG=""
#DIFFERENCE_POLICY="absolute" # {"absolute", "distinct"} # NOT YET IMPLEMENTED decide if considering the ABSOLUTE distance between the results or keep the DISTINCT sign/orientation of the relations, i.e., to keep the sign of the difference
DIFFERENCE_THRESHOLD=0.00 # minimum difference between a rule in the two variance to be considered relevant
#-oKeep keep values below the p-value and difference threshold
P_VALUE_ADJUSTMENT=hb     # pValue adjustment methods: Holm-Bonferroni, Benjamini-Hochberg {'none','hb','bh'}. Default is: 'hb'

##################################################################
java -cp Janus.jar $JANUS_VARIANT_MAINCLASS \
  -iLE1 $LOGS_ENCODING \
  -iLF1 $LOG_VAR_1 \
  -iLE2 $LOGS_ENCODING \
  -iLF2 $LOG_VAR_2 \
  -pValue $P_VALUE \
  -permutations $PERMUTATIONS \
  -oCSV $VARIANT_RESULTS_CSV \
  -differenceThreshold $DIFFERENCE_THRESHOLD \
  -pValueAdjustment $P_VALUE_ADJUSTMENT
#  --no-screen-print-out

##################################################################
#Change ; with ,
#sed -e 's/,/§/g'
