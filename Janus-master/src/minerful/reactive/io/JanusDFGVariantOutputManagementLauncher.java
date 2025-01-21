package minerful.reactive.io;

import minerful.MinerFulOutputManagementLauncher;
import minerful.params.SystemCmdParameters;
import minerful.reactive.params.JanusDFGVariantCmdParameters;
import minerful.reactive.params.JanusPrintParameters;
import minerful.reactive.variant.DFGPermutationResult;
import minerful.reactive.variant.DFGtimesVariantAnalysisCore;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Class to handle the output of Janus
 */
public class JanusDFGVariantOutputManagementLauncher extends MinerFulOutputManagementLauncher {

    /**
     * reads the terminal input parameters and launch the proper output functions
     *
     * @param varParams
     * @param janusViewParams
     * @param systemParams
     */
    public void manageVariantOutput(
            List<DFGPermutationResult> variantResults,
            JanusDFGVariantCmdParameters varParams,
            JanusPrintParameters janusViewParams,
            SystemCmdParameters systemParams) {
        manageVariantOutput(variantResults, varParams, janusViewParams, systemParams, "LOG1VALUE", "LOG2VALUE");
    }

    /**
     * reads the terminal input parameters and launch the proper output functions
     *
     * @param varParams
     * @param janusViewParams
     * @param systemParams
     */
    public void manageVariantOutput(
            List<DFGPermutationResult> variantResults,
            JanusDFGVariantCmdParameters varParams,
            JanusPrintParameters janusViewParams,
            SystemCmdParameters systemParams,
            String log1Name,
            String log2Name) {
        File outputFile = null;

        // ************* CSV
        if (varParams.outputCvsFile != null) {
            outputFile = retrieveFile(varParams.outputCvsFile);
            logger.info("Saving variant analysis result as CSV in " + outputFile + "...");
            double before = System.currentTimeMillis();

            exportVariantResultsToCSV(variantResults, outputFile, log1Name, log2Name);

            double after = System.currentTimeMillis();
            logger.info("Total CSV serialization time: " + (after - before));
        }

        if (janusViewParams != null && !janusViewParams.suppressResultsPrintOut) {
            printVariantResultsToScreen(variantResults);
        }

        // ************* JSON
        if (varParams.outputJsonFile != null) {
            outputFile = retrieveFile(varParams.outputJsonFile);
            logger.info("Saving variant analysis result as JSON in " + outputFile + "...");

            double before = System.currentTimeMillis();

//            TODO
            logger.info("JSON output yet not implemented");

            double after = System.currentTimeMillis();
            logger.info("Total JSON serialization time: " + (after - before));
        }

    }

    private void printVariantResultsToScreen(List<DFGPermutationResult> variantResults) {
        //		header row
        System.out.println("--------------------");
        System.out.println("relevant transitions time differences: " + variantResults.size());
        System.out.println();

        for (DFGPermutationResult r : variantResults) {
            System.out.println(r.toString());
        }
        System.out.println();
    }


    private void exportVariantResultsToCSV(List<DFGPermutationResult> variantResults, File outputFile, String log1Name, String log2Name) {
        //		header row
        try {
            String[] headerDetailed = {"FROM", "TO", "PERSPECTIVE", "pVALUE", "DIFFERENCE", log1Name, log2Name};
            FileWriter fwDetailed = new FileWriter(outputFile);
            CSVPrinter printerDetailed = new CSVPrinter(fwDetailed, CSVFormat.DEFAULT.withHeader(headerDetailed).withDelimiter(';'));

            for (DFGPermutationResult r : variantResults) {
                printerDetailed.printRecord(new String[]{
                        r.sourceNode,
                        r.destinationNode,
                        r.kind,
                        String.format("%.3f", r.pValue),
                        String.format("%.3f", r.diff / 60), // minutes
                        String.format("%.3f", r.log1Value / 60), // minutes
                        String.format("%.3f", r.log2Value / 60) // minutes
                });
            }
            fwDetailed.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
