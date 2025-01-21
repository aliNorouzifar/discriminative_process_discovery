#!/bin/bash

clear

##################################################################
##################################################################
## script variables
JANUS_MEASURES_MAINCLASS="minerful.JanusMeasurementsStarter"

#INPUT_LOGS_FOLDER="/home/alessio/Data/Phd/my_code/MINERful/tests-janus3MisuraThor/RealLife/logs"
INPUT_LOGS_FOLDER="/home/alessio/Data/Phd/my_code/MINERful/logs-only-events"
LOG_ENCODING="xes"
RESULTS_FOLDER="/home/alessio/Data/Phd/my_code/MINERful/tests-janus3MisuraThor/RealLife/results"

# Janus/MINERful main classes
JAVA_BIN="/home/alessio/Software/jdk/jdk-11.0.10/bin/java"
PYTHON_BIN="/home/alessio/Data/Phd/my_code/PyVEnv/ClusterMind/bin/python3"
JANUS_JAR="/home/alessio/Data/Phd/my_code/MINERful/Janus.jar"
MINERFUL_JAR="/home/alessio/Data/Phd/code_3rd_party/MINERful/MINERful.jar"
JANUS_DISCOVERY_MAINCLASS="minerful.JanusOfflineMinerStarter"
JANUS_DISCOVERY_SUPPORT=0.0
#JANUS_DISCOVERY_CONFIDENCE=0.8
MINERFUL_DISCOVERY_MAINCLASS="minerful.MinerFulMinerStarter"
#MINERFUL_DISCOVERY_SUPPORT=0.8    # support threshold for the initial discovery of the constraints of the variances
MINERFUL_DISCOVERY_CONFIDENCE=0.0 # confidence threshold for the initial discovery of the constraints of the variances
PERRACOTTA_JAR="/home/alessio/Data/Phd/code_3rd_party/Perracotta/lib/java-getopt-1.0.14.jar:/home/alessio/Data/Phd/code_3rd_party/Perracotta/bin edu.virginia.cs.terracotta.InferenceEngine"
##################################################################
##################################################################
mkdir -p $RESULTS_FOLDER

for INPUT_LOG in "${INPUT_LOGS_FOLDER}"/*.xes; do
  filename=$(basename -- "$INPUT_LOG")
  echo ${filename%.*}

  JANUS_MODEL=${RESULTS_FOLDER}"/${filename%.*}[model-JANUS].json"
  MINERFUL_MODEL=${RESULTS_FOLDER}"/${filename%.*}[model-MINERFUL].json"
  PERRACOTTA_MODEL=${RESULTS_FOLDER}"/${filename%.*}[model-PERRACOTTA].json"

  JANUS_MEASURES=${RESULTS_FOLDER}"/${filename%.*}[measures-JANUS].csv"
  MINERFUL_MEASURES=${RESULTS_FOLDER}"/${filename%.*}[measures-MINERFUL].csv"
  PERRACOTTA_MEASURES=${RESULTS_FOLDER}"/${filename%.*}[measures-PERRACOTTA].csv"

  RESULTS="${RESULTS_FOLDER}/${filename%.*}-results.csv"
  TIME_RESULTS="${RESULTS_FOLDER}/${filename%.*}-times.csv"

  echo "THRESHOLD;JANUS;MINERful;PERRACOTTA" >"${RESULTS}"
  echo "1.0;1.0;1.0;1.0" >>"${RESULTS}"

  echo "THRESHOLD;JANUS_RULES;JANUS_TIME;MINERful_RULES;MINERful_TIME;PERRACOTTA_RULES;PERRACOTTA_TIME" >"${TIME_RESULTS}"
  echo "1.0;0;0;0;0;0;0" >>"${TIME_RESULTS}"

  PERRACOTTA_LOG="${INPUT_LOGS_FOLDER}/${filename%.*}.perracotta"
  PERRACOTTA_ORIGINAL_MODEL="./${filename%.*}.perracotta.appro"
  $PYTHON_BIN /home/alessio/Data/Phd/my_code/MINERful/tests-janus3MisuraThor/RealLife/peracotta-log-converter.py "${INPUT_LOG}" "${INPUT_LOGS_FOLDER}/${filename%.*}.csv" "${PERRACOTTA_LOG}"
  java -cp ${PERRACOTTA_JAR} -i "${PERRACOTTA_LOG}">/dev/null

  for LEVEL in {95..00..5}; do
    echo -n "${LEVEL}..."
    THRESHOLD=0."${LEVEL}"
    echo -n $THRESHOLD >>"${RESULTS}"
    echo -n "${THRESHOLD};" >>"${TIME_RESULTS}"

    # DISCOVERY Janus
    $JAVA_BIN -cp $JANUS_JAR $JANUS_DISCOVERY_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -c $THRESHOLD -s $JANUS_DISCOVERY_SUPPORT -i 0 -oJSON "${JANUS_MODEL}" -vShush -d 'none' >/dev/null
    # Measures Janus
    start=$(date +%s.%N)
    $JAVA_BIN -cp $JANUS_JAR $JANUS_MEASURES_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -iMF "${JANUS_MODEL}" -iME JSON -oCSV "${JANUS_MEASURES}" -d none -nanLogSkip -measure "Confidence" -detailsLevel log >/dev/null
    end=$(date +%s.%N)
    runtime=$(echo "$end - $start" | bc -l)
    echo -n "$(tail ${RESULTS_FOLDER}"/${filename%.*}[measures-JANUS][logMeasures].csv" -n 1 | tr -d '\r' | tr -d 'MODEL')" >>"${RESULTS}"
    echo -n "$(wc -l ${RESULTS_FOLDER}"/${filename%.*}[measures-JANUS][logMeasures].csv" | sed 's/\s.*$//');$runtime;" >>"${TIME_RESULTS}"

    # DISCOVERY MINERful
    $JAVA_BIN -cp $MINERFUL_JAR $MINERFUL_DISCOVERY_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -c $MINERFUL_DISCOVERY_CONFIDENCE -s $THRESHOLD -oJSON "${MINERFUL_MODEL}" -vShush -d 'none' >/dev/null
    # Measures MINERful
    start=$(date +%s.%N)
    $JAVA_BIN -cp $JANUS_JAR $JANUS_MEASURES_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -iMF "${MINERFUL_MODEL}" -iME JSON -oCSV "${MINERFUL_MEASURES}" -d none -nanLogSkip -measure "Confidence" -detailsLevel log >/dev/null
    end=$(date +%s.%N)
    runtime=$(echo "$end - $start" | bc -l)
    echo -n "$(tail ${RESULTS_FOLDER}"/${filename%.*}[measures-MINERFUL][logMeasures].csv" -n 1 | tr -d '\r' | tr -d 'MODEL')" >>"${RESULTS}"
    echo -n "$(wc -l ${RESULTS_FOLDER}"/${filename%.*}[measures-MINERFUL][logMeasures].csv" | sed 's/\s.*$//');$runtime;" >>"${TIME_RESULTS}"

    # DISCOVERY PERRACOTTA
    java -cp ${PERRACOTTA_JAR} -i "${PERRACOTTA_LOG}" -a $THRESHOLD>/dev/null
    $PYTHON_BIN /home/alessio/Data/Phd/my_code/MINERful/tests-janus3MisuraThor/RealLife/peracotta-model-converter.py  "${PERRACOTTA_MODEL}" "${PERRACOTTA_ORIGINAL_MODEL}" "./${filename%.*}.perracotta."*
    # Measures PERRACOTTA
    start=$(date +%s.%N)
    $JAVA_BIN -cp $JANUS_JAR $JANUS_MEASURES_MAINCLASS -iLF "${INPUT_LOG}" -iLE $LOG_ENCODING -iMF "${PERRACOTTA_MODEL}" -iME JSON -oCSV "${PERRACOTTA_MEASURES}" -d none -nanLogSkip -measure "Confidence" -detailsLevel log >/dev/null
    end=$(date +%s.%N)
    runtime=$(echo "$end - $start" | bc -l)
    echo "$(tail ${RESULTS_FOLDER}"/${filename%.*}[measures-PERRACOTTA][logMeasures].csv" -n 1 | tr -d '\n' | tr -d 'MODEL')" >>"${RESULTS}"
    echo "$(wc -l ${RESULTS_FOLDER}"/${filename%.*}[measures-PERRACOTTA][logMeasures].csv" | sed 's/\s.*$//');$runtime" >>"${TIME_RESULTS}"
  done

  rm "./${filename%.*}.perracotta."*
  rm "${RESULTS_FOLDER}/${filename%.*}["*
  echo "DONE"
done
