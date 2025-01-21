package minerful.reactive.variant;

import com.google.common.collect.Ordering;
import com.google.common.collect.TreeMultimap;
import minerful.concept.ProcessModel;
import minerful.logparser.LogParser;
import minerful.logparser.LogTraceParser;
import minerful.reactive.measurements.MegaMatrixMonster;
import minerful.reactive.measurements.ReactiveMeasurementsOfflineQueryingCore;
import minerful.reactive.params.JanusMeasurementsCmdParameters;
import minerful.reactive.params.JanusVariantCmdParameters;
import minerful.reactive.params.JanusPrintParameters;
import org.apache.log4j.Logger;

import java.util.*;


/**
 * Class to organize the variant analysis
 */
public class ReactiveVariantAnalysisCore {

    protected static Logger logger;

    private final LogParser logParser_1;  // original log1 parser
    private final ProcessModel processSpecification1;  // original set of constraints mined from log1
    private final LogParser logParser_2; // original log2 parser
    private final ProcessModel processSpecification2;  // original set of constraints mined from log2
    private final JanusVariantCmdParameters janusVariantParams;  // input parameter of the analysis
    private final JanusPrintParameters janusViewParams;  // print behaciours parameters

    private float[][] lCodedIndex; // encoded log for efficient permutations. only constraints and traces indices are used
    private int processSpecificationUnionSize; // number of constraints in the specification union

    // RESULTS
    private Map<String, Float> spec1; // constraint->log measure, measurement of the union model over the first variant
    private Map<String, Float> spec2; // constraint->log measure, measurement of the union model over the second variant
    //

    public static final Map<String, String[]> HIERARCHY = new HashMap<String, String[]>() {{
        put("Participation", new String[]{});
        put("RespondedExistence", new String[]{"Participation($2)"}); //this link is ok only if the simplification works with equivalences of measures, otherwise it is not direct
        put("CoExistence", new String[]{"RespondedExistence($1,$2)", "RespondedExistence($2,$1)"});
        put("Succession", new String[]{"Response($1,$2)", "Precedence($1,$2)", "CoExistence($1,$2)"});
        put("Precedence", new String[]{"RespondedExistence($2,$1)"});
        put("Response", new String[]{"RespondedExistence($1,$2)"});
        put("AlternateSuccession", new String[]{"AlternateResponse($1,$2)", "AlternatePrecedence($1,$2)", "Succession($1,$2)"});
        put("AlternatePrecedence", new String[]{"Precedence($1,$2)"});
        put("AlternateResponse", new String[]{"Response($1,$2)"});
        put("ChainSuccession", new String[]{"ChainResponse($1,$2)", "ChainPrecedence($1,$2)", "AlternateSuccession($1,$2)"});
        put("ChainPrecedence", new String[]{"AlternatePrecedence($1,$2)"});
        put("ChainResponse", new String[]{"AlternateResponse($1,$2)"});
        put("NotCoExistence", new String[]{});
        put("NotSuccession", new String[]{"NotCoExistence($1,$2)"});
        put("NotChainSuccession", new String[]{"NotSuccession($1,$2)"});
    }}; // TODO only direct derivation for now, implement also simplification from combination of rules


    private Map<Integer, String> indexToConstraintMap;
    private Map<String, Integer> constraintToIndexMap;
    private List<Integer> permutableTracesIndexList;

    {
        if (logger == null) {
            logger = Logger.getLogger(ReactiveMeasurementsOfflineQueryingCore.class.getCanonicalName());
        }
    }

    /**
     * Constructor
     *
     * @param logParser_1
     * @param logParser_2
     * @param janusVariantParams
     */
    public ReactiveVariantAnalysisCore(LogParser logParser_1, ProcessModel processSpecification1, LogParser logParser_2, ProcessModel processSpecification2, JanusVariantCmdParameters janusVariantParams, JanusPrintParameters janusViewParams) {
        this.logParser_1 = logParser_1;
        this.processSpecification1 = processSpecification1;
        this.logParser_2 = logParser_2;
        this.processSpecification2 = processSpecification2;
        this.janusVariantParams = janusVariantParams;
        this.janusViewParams = janusViewParams;
    }

    /**
     * Launcher for variant analysis of two logs
     *
     * @return
     */
    public Map<String, Float> check() {
        logger.info("Variant Analysis start");
//        PREPROCESSING
        double before = System.currentTimeMillis();
        //        1. Models differences
//                NOTE USED FOR NOW
        //        setModelsDifferences(processSpecification1, processSpecification2);
        //        2. Models Union (total set of rules to check afterwards)
//        setModelsUnion(processSpecification1, processSpecification2);
        // total set of constraints to analyse, i.e., union of process specification 1 and 2
        ProcessModel processSpecificationUnion = ProcessModel.union(processSpecification1, processSpecification2);
        processSpecificationUnionSize = processSpecificationUnion.howManyConstraints();
        //        3. Encode log (create efficient log structure for the permutations)
        //        4. Precompute all possible results for the Encoded Log
        encodeLogsIndex(logParser_1, logParser_2, processSpecificationUnion);
        double after = System.currentTimeMillis();
        logger.info("Pre-processing time: " + (after - before));

//        PERMUTATION TEST
        before = System.currentTimeMillis();
        logger.info("Permutations processing...");
        int nPermutations;
        if (janusVariantParams.nPermutations <= 0) {
            nPermutations = (int) (processSpecificationUnionSize / janusVariantParams.pValue);
            logger.info("Number of required permutations: " + nPermutations);
//            TODO check that this number does not go beyond the possible permutations (unlikely, but theoretically possible)
        } else {
            nPermutations = janusVariantParams.nPermutations;
        }
        if (processSpecificationUnionSize / janusVariantParams.pValue > nPermutations) {
            // the smallest adjusted pValue is pValueThreshold/results.size(), thus the number of permutations must allow to reach such dimensions.
            // the worst case scenario is when all the hypotheses/constraints are statistically relevant
            logger.warn("Possible low number of iterations for a sound Multiple Testing adjustments! used:" + nPermutations + " safe upperbound expected:" + (int) (processSpecificationUnionSize / janusVariantParams.pValue));
        }
        Map<String, Float> results = permuteResultsIndex(nPermutations, true);
        after = System.currentTimeMillis();

//        POST-PROCESSING
        logger.info(" Permutations used:" + nPermutations + " minimum requirement for pValue adjustment:" + (int) (results.size() / janusVariantParams.pValue));
        if (janusVariantParams.pValueAdjustmentMethod != JanusVariantCmdParameters.PValueAdjustmentMethod.none) {
            if (results.size() / janusVariantParams.pValue > nPermutations) {
                // the smallest adjusted pValue is pValueThreshold/results.size(), thus the number of permutations must allow to reach such dimensions
                logger.warn("Not enough iterations for a sound Multiple Testing adjustments!");
            }
            pValueAdjustment(results, janusVariantParams.pValue, janusVariantParams.pValueAdjustmentMethod);
        }

        logger.info("Permutation test time: " + (after - before));
        return results;
    }

    /**
     * SIDE-EFFECT on result parameter!
     * <p>
     * A pValue correction method is applied on the result of the permutation test to mitigate the multiple testing problem.
     * <p>
     * The implemented methods are:
     * HB: Holm–Bonferroni (no assumptions, controls the FWER, more strict)
     * BH: Benjamini–Hochberg (independence or certain types of positive dependence, controls the FDR, more relaxed)
     *
     * @param results
     * @param pValueThreshold
     * @param method
     */
    private void pValueAdjustment(Map<String, Float> results, double pValueThreshold, JanusVariantCmdParameters.PValueAdjustmentMethod method) {
        logger.info("pValue adjustment using " + method + " correction method...");
        int m = results.size();
//        Sort results by pValue
        TreeMultimap<Float, String> sortedPvaluesResults = TreeMultimap.create(Ordering.natural(), Ordering.usingToString());
        for (String constraint : results.keySet()) {
            sortedPvaluesResults.put(results.get(constraint), constraint);
        }
//        Compute rank
        Map<Float, Integer> rankMap = new TreeMap();
        int currentRank = 0;
        for (float currentPvalue : sortedPvaluesResults.keySet()) {
            for (String line : sortedPvaluesResults.get(currentPvalue)) {
                currentRank++;
                if (!rankMap.containsKey(currentPvalue)) {
                    rankMap.put(currentPvalue, currentRank);
                }
            }
        }

        int removed = 0;
        boolean killSwitch = false;
        switch (method) {
            case hb:
                // Holm-Bonferroni
                for (float currentPvalue : sortedPvaluesResults.keySet()) {
                    if (killSwitch || currentPvalue >= pValueThreshold / (m + 1 - rankMap.get(currentPvalue))) {
                        killSwitch = true;
                        for (String constraint : sortedPvaluesResults.get(currentPvalue)) {
                            results.remove(constraint);
                            removed++;
                        }
                    }
                }
                break;
            case bh:
                // Benjamini–Hochberg
                for (float currentPvalue : sortedPvaluesResults.keySet()) {
                    if (killSwitch || currentPvalue >= rankMap.get(currentPvalue) / (float) m * pValueThreshold) {
                        killSwitch = true;
                        for (String constraint : sortedPvaluesResults.get(currentPvalue)) {
                            results.remove(constraint);
                            removed++;
                        }
                    }
                }
                break;
            default:
                throw new IllegalArgumentException("Unknown method code! use HB (Holm–Bonferroni) or BH (Benjamini–Hochberg)");
        }

        logger.info("Removed " + removed + " of " + m + " results");
    }


    /**
     * Permutation test in which is taken the encoded results and
     * check of the significance of the permutation test results
     *
     * @param nPermutations
     * @param nanCheck
     * @return Map from constraint to its pValue
     */
    private Map<String, Float> permuteResultsIndex(int nPermutations, boolean nanCheck) {
        int nConstraints = processSpecificationUnionSize;

        int log1Size = logParser_1.length();
        int log2Size = logParser_2.length();
        logger.info("[Tot traces:" + (log1Size + log2Size) + " Constraints:" + processSpecificationUnionSize + "]");


//        List<Integer> permutableTracesIndexList = new ArrayList<>();
//        for (Iterator<LogTraceParser> it = logParser_1.traceIterator(); it.hasNext(); ) {
//            permutableTracesIndexList.add(traceToIndexMap.get(it.next().printStringTrace()));
//        }
//        for (Iterator<LogTraceParser> it = logParser_2.traceIterator(); it.hasNext(); ) {
//            permutableTracesIndexList.add(traceToIndexMap.get(it.next().printStringTrace()));
//        }

        float[] pValues = new float[nConstraints];
        Set<Integer> blackList = new HashSet();

        float[] result1 = new float[nConstraints];
        float[] result2 = new float[nConstraints];
        float[] initialDifference = new float[nConstraints];

        int step = 25;
        for (int i = 0; i < nPermutations; i++) {
            if (!janusViewParams.suppressPermutationStatusPrint && i % step == 0)
                System.out.print("\rPermutation: " + i + "/" + nPermutations);  // Status counter "current trace/total trace"

            for (int c = 0; c < nConstraints; c++) {
                if (!janusVariantParams.oKeep && blackList.contains(c))
                    continue;
                int traceIndex = -1;
                int nanTraces1 = 0;
                int nanTraces2 = 0;
                for (int t : permutableTracesIndexList) {
                    traceIndex++;
                    if (traceIndex < log1Size) {
                        if (nanCheck & Float.isNaN(lCodedIndex[t][c])) {
                            nanTraces1++;
                            continue; // TODO expose in input
                        }
                        result1[c] += lCodedIndex[t][c];
                    } else {
                        if (nanCheck & Float.isNaN(lCodedIndex[t][c])) {
                            nanTraces2++;
                            continue; // TODO expose in input
                        }
                        result2[c] += lCodedIndex[t][c];
                    }
                }
                result1[c] = result1[c] / (log1Size - nanTraces1);
                result2[c] = result2[c] / (log2Size - nanTraces2);
                if (i == 0) initialDifference[c] = Math.abs(result1[c] - result2[c]);
                if (Math.abs(result1[c] - result2[c]) >= initialDifference[c]) {
                    pValues[c] += 1.0;
                }
                if (!janusVariantParams.oKeep && (pValues[c] / nPermutations) > janusVariantParams.pValue)
                    blackList.add(c); //if the constraints present a pValues greater than the threshold before the end of the permutations, we can discard it immediately
                result1[c] = 0.0f;
                result2[c] = 0.0f;

            }
//            permutation "0" are the original logs
            Collections.shuffle(permutableTracesIndexList);
        }
        if (!janusViewParams.suppressPermutationStatusPrint) {
            System.out.print("\rPermutation: " + nPermutations + "/" + nPermutations);
            System.out.println();
        }

        // Significance test in case of NEGATIVE/POSITIVE DISTANCE
        logger.info("Significance testing...");
        Map<String, Float> result = new HashMap<String, Float>(); // constraint: pValue
        for (int cIndex = 0; cIndex < nConstraints; cIndex++) {
            pValues[cIndex] = pValues[cIndex] / nPermutations;

            if (janusVariantParams.oKeep || pValues[cIndex] <= janusVariantParams.pValue) {
                result.put(indexToConstraintMap.get(cIndex), pValues[cIndex]);
            }
        }
        logger.info("Rules Number: " + nConstraints + " ; relevant: " + result.size() + " ; non-relevant: " + (nConstraints - result.size()));
        return result;
    }

    /**
     * Encode the input traces for efficient permutation.
     * the result is a Map where the keys are the hash of the traces and the content in another map with key:value as constrain:measure.
     * In this way we check only here the constraints in each trace and later we permute only the results
     * <p>
     * Transform the encoded map into a matrix where traces and constraints are referred by indices.
     * compute the encoding and return the reference mappings
     *
     * @param model
     * @param logParser_1
     * @param logParser_2
     */
    private void encodeLogsIndex(LogParser logParser_1, LogParser logParser_2, ProcessModel model) {
//        encode index
        lCodedIndex = new float[logParser_1.length() + logParser_2.length()][processSpecificationUnionSize]; // lCodedIndex[trace index][constraint index]

        indexToConstraintMap = new HashMap<>();
        constraintToIndexMap = new HashMap<>();

        //        encode
        encodeLog(logParser_1, model, 0);
        encodeLog(logParser_2, model, logParser_1.length());

        Set<String> constraintsRemovalCandidate = new HashSet<>();
        Set<String> constraintsList = constraintToIndexMap.keySet();

        //  hierarchical simplification
        if (janusVariantParams.simplify) {
            logger.info("Rules simplification...");

            for (String c : constraintsList) {
                if (constraintsRemovalCandidate.contains(c)) continue;
                String template = c.split("\\(")[0];
                // skip constraints with only one variable from simplification
                if (c.contains(",") == false || HIERARCHY.get(template) == null) continue;
                String cVar1 = c.split("\\(")[1].replace(")", "").split(",")[0];
                String cVar2 = c.split("\\(")[1].replace(")", "").split(",")[1];
                for (String d : HIERARCHY.get(template)) {
                    String derived = d.replace("$1", cVar1).replace("$2", cVar2);
                    if (constraintsList.contains(derived)) {
                        if (spec1.get(derived) - spec1.get(c) == 0 || spec2.get(derived) - spec2.get(c) == 0) {
                            constraintsRemovalCandidate.add(c);
                        }
                    }
                }
            }
            logger.info("Number of simplified constraints: " + (processSpecificationUnionSize - constraintsRemovalCandidate.size()));

            //            simplification of symmetric constraints [CoExistence, NotCoExistence]
            int initConstrNum = processSpecificationUnionSize - constraintsRemovalCandidate.size();

            for (String c : constraintsList) {
                if (constraintsRemovalCandidate.contains(c)) continue;
                // skip constraints with only one variable from simplification
                if (!c.contains(",")) continue;
                String template = c.split("\\(")[0];
                // only symmetric constraints
                if (!template.equals("CoExistence") && !template.equals("NotCoExistence")) continue;
                // skip constraints already labelled for removal
                if (constraintsRemovalCandidate.contains(c)) continue;


                String cVar1 = c.split("\\(")[1].replace(")", "").split(",")[0];
                String cVar2 = c.split("\\(")[1].replace(")", "").split(",")[1];
                String symmetricConstraint = template + "(" + cVar2 + "," + cVar1 + ")";
                if (constraintsList.contains(symmetricConstraint)) constraintsRemovalCandidate.add(symmetricConstraint);
            }
            logger.info("Number of simplified symmetric constraints: " + (initConstrNum - (processSpecificationUnionSize - constraintsRemovalCandidate.size())));
        }
        //  difference min cut
        if (!janusVariantParams.oKeep) {
            logger.info("Removing rules with not enough initial difference...");
            int initConstrNum = processSpecificationUnionSize - constraintsRemovalCandidate.size();

            for (String c : constraintsList) {
                if (constraintsRemovalCandidate.contains(c)) continue;
                float difference = Math.abs(spec1.get(c) - spec2.get(c));
//          if one is NaN, the rule is removed if the non-NaN value is below the difference threshold (like if NaN=0)
                if ((difference < janusVariantParams.differenceThreshold) ||
                        (Float.isNaN(spec1.get(c)) && spec2.get(c) < janusVariantParams.differenceThreshold) ||
                        (Float.isNaN(spec2.get(c)) && spec1.get(c) < janusVariantParams.differenceThreshold)) {
                    constraintsRemovalCandidate.add(c);
                }
            }
            logger.info("Number of removed constraints: " + (initConstrNum - (processSpecificationUnionSize - constraintsRemovalCandidate.size())));
        }
//        Measures below threshold
        if (janusVariantParams.measureThreshold > 0) {
            logger.info("Removing rules below threshold in both variants...");
            int initConstrNum = processSpecificationUnionSize - constraintsRemovalCandidate.size();

            String[] initialConstraintsList = new String[spec1.size()];
            spec1.keySet().toArray(initialConstraintsList);
            for (String c : constraintsList) {
                if (constraintsRemovalCandidate.contains(c)) continue;
                if (spec1.get(c) < janusVariantParams.measureThreshold && spec2.get(c) < janusVariantParams.measureThreshold) {
                    constraintsRemovalCandidate.add(c);
                }
            }
            logger.info("Number of removed constraints: " + (initConstrNum - (processSpecificationUnionSize - constraintsRemovalCandidate.size())));
        }

//        remove selected constraints and update the constraints indices map
        Set<Integer> constraintsRemovalCandidateIndices = new HashSet<>();
        for (String c : constraintsRemovalCandidate) {
            spec1.remove(c);
            spec2.remove(c);
            constraintsRemovalCandidateIndices.add(constraintToIndexMap.get(c));
            processSpecificationUnionSize--;
        }
        float[][] res = new float[lCodedIndex.length][lCodedIndex[0].length - constraintsRemovalCandidateIndices.size()];
        for (int t = 0; t < lCodedIndex.length; t++) {
            int i = 0;
            for (int c = 0; c < lCodedIndex[t].length; c++) {
                if (constraintsRemovalCandidateIndices.contains(c)) continue;
                res[t][i] = lCodedIndex[t][c];
                i++;
            }
        }

        Map<Integer, String> newIndexToConstraintMap = new HashMap<>();
        Map<String, Integer> newConstraintToindexMap = new HashMap<>();
        int currentIndex = 0;
        for (int c = 0; c < lCodedIndex[0].length; c++) {
            if (constraintsRemovalCandidateIndices.contains(c)) continue;
            newIndexToConstraintMap.put(currentIndex, indexToConstraintMap.get(c));
            newConstraintToindexMap.put(indexToConstraintMap.get(c), currentIndex);
            currentIndex++;
        }

        lCodedIndex = res;

        indexToConstraintMap = newIndexToConstraintMap;
        constraintToIndexMap = newConstraintToindexMap;


        permutableTracesIndexList = new ArrayList<>();
        Map<String, Integer> traceToIndexMap = new HashMap<>();

        int currentTrace = 0;
        for (Iterator<LogTraceParser> it = logParser_1.traceIterator(); it.hasNext(); ) {
            LogTraceParser tr = it.next();
            String stringTrace = tr.printStringTrace();
            traceToIndexMap.put(stringTrace, currentTrace);
            currentTrace++;
        }
        for (Iterator<LogTraceParser> it = logParser_2.traceIterator(); it.hasNext(); ) {
            LogTraceParser tr = it.next();
            String stringTrace = tr.printStringTrace();
            traceToIndexMap.put(stringTrace, currentTrace);
            currentTrace++;
        }
        for (Iterator<LogTraceParser> it = logParser_1.traceIterator(); it.hasNext(); ) {
            permutableTracesIndexList.add(traceToIndexMap.get(it.next().printStringTrace()));
        }
        for (Iterator<LogTraceParser> it = logParser_2.traceIterator(); it.hasNext(); ) {
            permutableTracesIndexList.add(traceToIndexMap.get(it.next().printStringTrace()));
        }
    }


    /**
     * Precompute the evaluation and encode a map where each distinct trace is linked to all the constraints measumentents
     *
     * @param logParser
     * @param model
     * @param resultIndex index from which to append the computed measures into the encoded result matrix
     * @return
     */
    private void encodeLog(LogParser logParser, ProcessModel model, int resultIndex) {
        JanusMeasurementsCmdParameters janusCheckingParams = new JanusMeasurementsCmdParameters(false, 0, true, false);
        ReactiveMeasurementsOfflineQueryingCore reactiveMeasurementsOfflineQueryingCore = new ReactiveMeasurementsOfflineQueryingCore(
                0, logParser, janusCheckingParams, janusViewParams, null, logParser.getTaskCharArchive(), null, model.bag);
        double before = System.currentTimeMillis();
        MegaMatrixMonster measures = reactiveMeasurementsOfflineQueryingCore.check();
        int constraintsNum = measures.getConstraintsNumber() - 1;
        double after = System.currentTimeMillis();

        logger.info("Total KB checking time: " + (after - before));

//      compute only the desired measure
        float[][] tracesMeasure = measures.retrieveSingleTraceMeasures(
                janusVariantParams.measure,
                janusCheckingParams.nanTraceSubstituteFlag,
                janusCheckingParams.nanTraceSubstituteValue);

//        constraints indices map
        int ic = constraintToIndexMap.size();
        for (String constraint : measures.getConstraintsNames()) {
            if (!constraintToIndexMap.containsKey(constraint)) {
                constraintToIndexMap.put(constraint, ic);
                indexToConstraintMap.put(ic, constraint);
                ic++;
            }
        }


//      fill result
        for (int currentTrace = 0; currentTrace < logParser.length(); currentTrace++) {
            System.arraycopy(tracesMeasure[currentTrace], 0, lCodedIndex[currentTrace + resultIndex], 0, constraintsNum);
        }

//      save log measures
        Map<String, Float> logMeasures = new HashMap<>(); // constraint->measurement
        boolean nanCheck = true;
        for (int c = 0; c < constraintsNum; c++) {
            int nanTraces = 0;
            float constraintResult = 0;
            for (int t = 0; t < logParser.length(); t++) {
                if (nanCheck & Float.isNaN(tracesMeasure[t][c])) {
                    nanTraces++;
                    continue; // TODO expose in input
                }
                constraintResult += tracesMeasure[t][c];
            }
            constraintResult = constraintResult / (logParser.length() - nanTraces);
            logMeasures.put(indexToConstraintMap.get(c), constraintResult);
        }

        if (resultIndex > 0) {
            spec2 = logMeasures;
        } else {
            spec1 = logMeasures;
        }

    }

    /**
     * Get the log level measurement of a given log parser using already encoded log measurements
     * *
     *
     * @param nanCheck
     * @return Map<String, Float>  constraint-name:measurement
     */
    private Map<String, Float> getMeasurementsOfOneVariant(boolean nanCheck, LogParser logParser, Map<String, Map<String, Float>> lCoded) {
        Map<String, Float> result = new HashMap<>(); // constraint->measurement
        int logSize = logParser.length();
        List<String> permutableTracesList = new LinkedList<>();
        for (Iterator<LogTraceParser> it = logParser.traceIterator(); it.hasNext(); ) {
            permutableTracesList.add(it.next().printStringTrace());
        }

        Set<String> constraints = lCoded.values().iterator().next().keySet();
        for (String c : constraints) {
            int nanTraces = 0;
            float constraintResult = 0;
            for (String t : permutableTracesList) {
                if (nanCheck & Float.isNaN(lCoded.get(t).get(c))) {
                    nanTraces++;
                    continue; // TODO expose in input
                }
                constraintResult += lCoded.get(t).get(c);

            }
            constraintResult = constraintResult / (logSize - nanTraces);
            result.put(c, constraintResult);
        }

        return result;
    }

    /**
     * the first variant
     * Get the original log level measurement of the first variant
     *
     * @param nanCheck
     * @return Map<String, Float>  constraint-name:measurement
     */
    public Map<String, Float> getMeasurementsVar1(boolean nanCheck) {
        return spec1;
    }

    /**
     * Get the original log level measurement of the second variant
     *
     * @param nanCheck
     * @return Map<String, Float>  constraint-name:measurement
     */
    public Map<String, Float> getMeasurementsVar2(boolean nanCheck) {
        return spec2;
    }

}
