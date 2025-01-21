package minerful.reactive.variant;

import minerful.concept.TaskChar;
import minerful.concept.TaskClass;
import minerful.logparser.*;
import minerful.reactive.dfg.DFG;
import minerful.reactive.dfg.DFGTransition;
import minerful.reactive.params.JanusDFGVariantCmdParameters;
import minerful.reactive.params.JanusPrintParameters;
import org.apache.log4j.Logger;

import java.time.Duration;
import java.time.Instant;
import java.util.*;

/**
 * Class to organize the variant analysis based on DFG times
 */
public class DFGtimesVariantAnalysisCore {

    protected static Logger logger;

    {
        if (logger == null) {
            logger = Logger.getLogger(DFGtimesVariantAnalysisCore.class.getCanonicalName());
        }
    }

    private final XesLogParser logParser1;  // original log1 parser
    private final XesLogParser logParser2; // original log2 parser
    private final JanusDFGVariantCmdParameters janusVariantParams;  // input parameter of the analysis
    private final JanusPrintParameters janusViewParams;  // input parameter of the analysis

    private Map<Integer, TaskClass> indexToTaskMap;
    private Map<TaskClass, Integer> TaskToIndexMap;

    public DFGtimesVariantAnalysisCore(XesLogParser logParser_1, XesLogParser logParser_2, JanusDFGVariantCmdParameters janusParams, JanusPrintParameters janusViewParams) {
        this.logParser1 = logParser_1;
        this.logParser2 = logParser_2;
        this.janusVariantParams = janusParams;
        this.janusViewParams = janusViewParams;
    }

    /**
     * Launcher for the variants check using graph structure for DFGs
     */
    public List<DFGPermutationResult> checkWithGraph() {
        logger.info("DFG Variant Analysis start");

        Instant start = Instant.now();

        logger.info("Pre-processing processing...");
//        Encode tasks: using maps operations is too heavy during the permutation, better a matrix with known indices
        int tasksNumber = logParser1.getTaskCharArchive().size();

        indexToTaskMap = new HashMap<Integer, TaskClass>();
        TaskToIndexMap = new HashMap<TaskClass, Integer>();
        int taskIndex = 0;
        for (TaskChar t : logParser1.getTaskCharArchive().getTaskChars()) {
            TaskClass tc = t.taskClass;
            indexToTaskMap.put(taskIndex, tc);
            TaskToIndexMap.put(tc, taskIndex);
            taskIndex++;
        }
        logger.info("Pre-processing time: " + Duration.between(start, Instant.now()));

        logger.info("Permutation test with Graphs and XES parser...");
        start = Instant.now();

//        structure to store the initial difference
        //        matrix NxN where N is the number of tasks

        DFGEncodedLog eLog1 = new DFGEncodedLog(logParser1);
        DFGEncodedLog eLog2 = new DFGEncodedLog(logParser2);
        DFG dfg1 = DFG.buildDFGFromEncodedLog(eLog1);
        DFG dfg2 = DFG.buildDFGFromEncodedLog(eLog2);

//        DFG dfg1 = DFG.buildDFGFromXesLogParser(logParser1);
//        DFG dfg2 = DFG.buildDFGFromXesLogParser(logParser2);

        float[][][] initialDifferences = compareDFGsGraphs(dfg1, dfg2, tasksNumber);
        logger.info("expected safe number of permutations for Multiple Test adjustment: " + (int) (notZeroDiff(initialDifferences) / janusVariantParams.pValue));
//        initialize structure to store the intermediate results for all transitions
        //        matrix NxNx3 where N is the number of tasks and and 3 is the <AVG,MIN,MAX> of the transition
        //        the structure counts how many time the difference was greater than the one observed initially
        int[][][] relevantCounter = new int[tasksNumber][tasksNumber][3];

//        Permutation test & significance test
//        permutationTestGraphBased(initialDifferences, relevantCounter, tasksNumber, logParser1, logParser2);
        permutationTestGraphBasedEncoded(initialDifferences, relevantCounter, tasksNumber, eLog1, eLog2);
        List<DFGPermutationResult> result = significanceTestGraph(tasksNumber, initialDifferences, relevantCounter, dfg1, dfg2);

//        POST-PROCESSING
        logger.info("Required permutations for multiple Test adjustment: " + (int) (result.size() / janusVariantParams.pValue) + " [used:" + janusVariantParams.nPermutations + "]");
        if (janusVariantParams.pValueAdjustmentMethod != JanusDFGVariantCmdParameters.PValueAdjustmentMethod.none)
            result = pValueAdjustment(result);

        logger.info("Permutation test time: " + Duration.between(start, Instant.now()));

        return result;
    }

    /**
     * Adjust the pValues of the results of the permutation test to address the Multiple Testing problem.
     * Returns the filtered list of results.
     *
     * @param results
     * @return
     */
    private List<DFGPermutationResult> pValueAdjustment(List<DFGPermutationResult> results) {
        int m = results.size();
        if (m / janusVariantParams.pValue > janusVariantParams.nPermutations) {
            // the smallest adjusted pValue is pValueThreshold/m, thus the number of permutations must allow to reach such dimensions
            logger.warn("Not enough iterations for a safe Multiple Testing adjustment!");
        }
        logger.info("pValue correction using " + janusVariantParams.pValueAdjustmentMethod + " method...");
        //        Sort results by pValue
        results.sort(Comparator.comparingDouble(o -> o.pValue)); //increasing order
        //        Compute rank
        TreeMap<Double, Integer> rankMap = new TreeMap();
        int currentRank = 0;
        for (DFGPermutationResult currentResult : results) {
            currentRank++;
            if (!rankMap.containsKey(currentResult.pValue)) {
                rankMap.put(currentResult.pValue, currentRank);
            }
        }

        int removed = 0;
        boolean killSwitch = false;
        List<DFGPermutationResult> newResults = new LinkedList<>();

        switch (janusVariantParams.pValueAdjustmentMethod) {
            case hb:
                // Holm-Bonferroni
                for (DFGPermutationResult currentResult : results) {
                    if (killSwitch || currentResult.pValue >= janusVariantParams.pValue / (m + 1 - rankMap.get(currentResult.pValue))) {
                        killSwitch = true;
                        removed++;
                    } else {
                        newResults.add(currentResult);
                    }
                }
                break;
            case bh:
                // Benjamini–Hochberg
                for (DFGPermutationResult currentResult : results) {
                    if (killSwitch || currentResult.pValue >= rankMap.get(currentResult.pValue) / (float) m * janusVariantParams.pValue) {
                        killSwitch = true;
                        removed++;
                    } else {
                        newResults.add(currentResult);
                    }
                }
                break;
            default:
                throw new IllegalArgumentException("Unknown method code! use HB (Holm–Bonferroni) or BH (Benjamini–Hochberg)");
        }


        logger.info("Removed " + removed + " of " + m + " results");

        return newResults;
    }

    /**
     * Returns the number of non-zero differences for the given differences matrix
     *
     * @param differenceMatrix
     * @return
     */
    private int notZeroDiff(float[][][] differenceMatrix) {
        int counter = 0;
        for (float[][] row : differenceMatrix) {
            for (float[] col : row) {
                for (float value : col) {
                    if (value > 0) counter++;
                }
            }
        }
        return counter;
    }

    /**
     * Perform the permutation test on the given input graph DFGs
     *
     * @param initialDifferences
     * @param relevantCounter
     * @param tasksNumber
     * @param logParser1
     * @param logParser2
     */
    @Deprecated
    private void permutationTestGraphBased(float[][][] initialDifferences, int[][][] relevantCounter, int tasksNumber, XesLogParser logParser1, XesLogParser logParser2) {
        int log1len = logParser1.length();
        int log2len = logParser2.length();
        int totLen = log1len + log2len;

//        Merge log for shuffling
        XesLogParser mergedLogParser = XesLogParser.mergeParsersWithEquivalentTaskChars(logParser1, logParser2);

        XesLogParser newLog1Parser;
        XesLogParser newLog2Parser;
        DFG newDFG1;
        DFG newDFG2;

        int step = 25;
        for (int i = 0; i < janusVariantParams.nPermutations; i++) {
            if (!janusViewParams.suppressPermutationStatusPrint && i % step == 0)
                System.out.print("\rPermutation: " + i + "/" + janusVariantParams.nPermutations);  // Status counter "current trace/total trace"

//            Shuffle trace
            mergedLogParser.shuffleTraces();
//            generate the two new logs
            newLog1Parser = (XesLogParser) mergedLogParser.takeASlice(0, log1len);
            newLog2Parser = (XesLogParser) mergedLogParser.takeASlice(log1len, totLen);
//            compute the differences
            newDFG1 = DFG.buildDFGFromXesLogParser(newLog1Parser);
            newDFG2 = DFG.buildDFGFromXesLogParser(newLog2Parser);
            float[][][] currentDifferences = compareDFGsGraphs(newDFG1, newDFG2, tasksNumber);
            checkDifferencesAgainstReferenceGraph(currentDifferences, initialDifferences, relevantCounter);
        }
        if (!janusViewParams.suppressPermutationStatusPrint)
            System.out.println("\rPermutation: " + janusVariantParams.nPermutations + "/" + janusVariantParams.nPermutations);
    }

    /**
     * Perform the permutation test on the given input graph DFGs using an encoded Event Log (around x4 faster than using normal XESParser).
     *
     * @param initialDifferences
     * @param relevantCounter
     * @param tasksNumber
     * @param logParser1
     * @param logParser2
     */
    private void permutationTestGraphBasedEncoded(float[][][] initialDifferences, int[][][] relevantCounter, int tasksNumber, DFGEncodedLog logParser1, DFGEncodedLog logParser2) {
        int log1len = logParser1.length();
        int log2len = logParser2.length();
        int totLen = log1len + log2len;

//        Merge log for shuffling
        DFGEncodedLog mergedLogParser = logParser1.merge(logParser2);

        DFG newDFG1;
        DFG newDFG2;

        int step = 25;
        for (int i = 0; i < janusVariantParams.nPermutations; i++) {
            if (!janusViewParams.suppressPermutationStatusPrint && i % step == 0)
                System.out.print("\rPermutation: " + i + "/" + janusVariantParams.nPermutations);  // Status counter "current trace/total trace"

//            Shuffle trace
            mergedLogParser.shuffleTraces();
//            generate the two new logs & compute the differences
            newDFG1 = DFG.buildDFGFromEncodedLog(mergedLogParser.traces.subList(0, log1len));
            newDFG2 = DFG.buildDFGFromEncodedLog(mergedLogParser.traces.subList(log1len, totLen));
            float[][][] currentDifferences = compareDFGsGraphs(newDFG1, newDFG2, tasksNumber);
            checkDifferencesAgainstReferenceGraph(currentDifferences, initialDifferences, relevantCounter);
        }
        if (!janusViewParams.suppressPermutationStatusPrint)
            System.out.println("\rPermutation: " + janusVariantParams.nPermutations + "/" + janusVariantParams.nPermutations);
    }


    /**
     * Check the graph-based permutations results and return the significant differences in output
     *
     * @param tasksNumber
     * @param initialDifferences
     * @param relevantCounter
     * @param dfg1
     * @param dfg2
     * @return
     */
    public List<DFGPermutationResult> significanceTestGraph(int tasksNumber, float[][][] initialDifferences, int[][][] relevantCounter, DFG dfg1, DFG dfg2) {
        //        Significance test
        int nanResultsCounter = 0;
        List<DFGPermutationResult> result = new LinkedList<>();
        for (int i = 0; i < tasksNumber; i++) {
            for (int j = 0; j < tasksNumber; j++) {
                boolean flag = false;
                for (int k = 0; k < 3; k++) {
                    float currentPValue = (float) relevantCounter[i][j][k] / janusVariantParams.nPermutations;
                    if (currentPValue <= janusVariantParams.pValue) {
                        TaskClass tcI = indexToTaskMap.get(i);
                        TaskClass tcJ = indexToTaskMap.get(j);
                        DFGTransition time1 = dfg1.getTransition(tcI, tcJ);
                        DFGTransition time2 = dfg2.getTransition(tcI, tcJ);
                        if (time1 == null || time2 == null) {
                            flag = true;
                            continue;
                        }
                        switch (k) {
                            case 0:
                                result.add(new DFGPermutationResult(
                                        indexToTaskMap.get(i).toString(),
                                        indexToTaskMap.get(j).toString(),
                                        "AVG",
                                        currentPValue,
                                        initialDifferences[i][j][k],
                                        time1.getTimeAvg(),
                                        time2.getTimeAvg()
                                ));
                                break;
                            case 1:
                                result.add(new DFGPermutationResult(
                                        indexToTaskMap.get(i).toString(),
                                        indexToTaskMap.get(j).toString(),
                                        "MIN",
                                        currentPValue,
                                        initialDifferences[i][j][k],
                                        time1.getTimeMin(),
                                        time2.getTimeMin()
                                ));
                                break;
                            case 2:
                                result.add(new DFGPermutationResult(
                                        indexToTaskMap.get(i).toString(),
                                        indexToTaskMap.get(j).toString(),
                                        "MAX",
                                        currentPValue,
                                        initialDifferences[i][j][k],
                                        time1.getTimeMax(),
                                        time2.getTimeMax()
                                ));
                                break;
                        }
                    }
                }
                if (flag) {
                    nanResultsCounter++;
                }
            }
        }
        logger.info("NaN Relevant differences: " + nanResultsCounter);
        return result;
    }


    /**
     * Check the difference of the given graph-based DFG difference and update the counters
     *
     * @param currentDifferences
     * @param initialDifferences
     * @param relevantCounter
     */
    private void checkDifferencesAgainstReferenceGraph(float[][][] currentDifferences, float[][][] initialDifferences, int[][][] relevantCounter) {
        int tNum = currentDifferences.length;

        for (int i = 0; i < tNum; i++) {
            for (int j = 0; j < tNum; j++) {
                for (int k = 0; k < 3; k++) {
                    if (Float.isNaN(currentDifferences[i][j][k]) && !Float.isNaN(initialDifferences[i][j][k]))
                        relevantCounter[i][j][k]++;
                    else if (!Float.isNaN(currentDifferences[i][j][k]) && Float.isNaN(initialDifferences[i][j][k]))
                        relevantCounter[i][j][k]++;
                    else if (currentDifferences[i][j][k] >= initialDifferences[i][j][k])
                        relevantCounter[i][j][k]++;
                }
            }
        }
    }

    /**
     * Compare two DFGs and return the dime differences between their transition considering <AVG,MIN,MAX>
     * The union of all the task is considered.
     * The result is a NxNx3 matrix where N is the number of tasks and 3 are the aforementioned <AVG,MIN,MAX> of the transition
     *
     * @param dfg1
     * @param dfg2
     * @return
     */
    private float[][][] compareDFGsGraphs(DFG dfg1, DFG dfg2, int tasksTotalNumber) {
        float[][][] result = new float[tasksTotalNumber][tasksTotalNumber][3];

        for (int i = 0; i < tasksTotalNumber; i++) {
            TaskClass tcI = indexToTaskMap.get(i);
            for (int j = 0; j < tasksTotalNumber; j++) {
                TaskClass tcJ = indexToTaskMap.get(j);
                DFGTransition time1 = dfg1.getTransition(tcI, tcJ);
                DFGTransition time2 = dfg2.getTransition(tcI, tcJ);

                if (time1 != null & time2 != null) {
                    result[i][j][0] = Math.abs(time1.getTimeAvg() - time2.getTimeAvg());
                    result[i][j][1] = Math.abs(time1.getTimeMin() - time2.getTimeMin());
                    result[i][j][2] = Math.abs(time1.getTimeMax() - time2.getTimeMax());
                } else if (time1 == null & time2 == null) {
                    result[i][j][0] = 0.0F;
                    result[i][j][1] = 0.0F;
                    result[i][j][2] = 0.0F;
//                } else if (time1 == null & time2 != null) {
//                } else if (time1 != null & time2 == null) {
                } else {
                    result[i][j][0] = Float.NaN;
                    result[i][j][1] = Float.NaN;
                    result[i][j][2] = Float.NaN;
                }
            }
        }
        return result;
    }

}