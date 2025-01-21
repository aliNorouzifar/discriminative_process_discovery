package minerful;

import minerful.concept.TaskCharArchive;
import minerful.logparser.LogEventClassifier;
import minerful.logparser.LogParser;
import minerful.logparser.XesLogParser;
import minerful.params.SystemCmdParameters;
import minerful.reactive.params.JanusDFGVariantCmdParameters;
import minerful.reactive.params.JanusPrintParameters;
import minerful.reactive.variant.DFGPermutationResult;
import minerful.reactive.variant.DFGtimesVariantAnalysisCore;
import minerful.utils.MessagePrinter;

import java.io.File;
import java.util.List;

/**
 * Class for launching Janus variant analysis on two logs
 */
public class JanusDFGVariantAnalysisLauncher {
    public static MessagePrinter logger = MessagePrinter.getInstance(JanusDFGVariantAnalysisLauncher.class);

    private JanusDFGVariantCmdParameters janusParams;
    private SystemCmdParameters systemParams;
    private JanusPrintParameters janusViewParams;

    private XesLogParser eventLog1;
    private XesLogParser eventLog2;

    public JanusDFGVariantAnalysisLauncher(JanusDFGVariantCmdParameters janusParams) {
        this.janusParams = janusParams;
        this.systemParams = new SystemCmdParameters();
        this.janusViewParams = new JanusPrintParameters();

        logger.info("Loading event logs...");
        this.eventLog1 = (XesLogParser) deriveLogParserFromLogFile(janusParams.inputLogLanguage1, janusParams.inputLogFile1, janusParams.eventClassification, null);
        this.eventLog2 = (XesLogParser) deriveLogParserFromLogFile(janusParams.inputLogLanguage2, janusParams.inputLogFile2, janusParams.eventClassification, eventLog1.getTaskCharArchive());
//        this is a bit redundant, but to make sure that both have the same alphabet we recompute the first parser with the alphabet of the second, which now has both alphabets
        this.eventLog1 = (XesLogParser) deriveLogParserFromLogFile(janusParams.inputLogLanguage1, janusParams.inputLogFile1, janusParams.eventClassification, eventLog2.getTaskCharArchive());
    }

    public JanusDFGVariantAnalysisLauncher(JanusDFGVariantCmdParameters janusParams, JanusPrintParameters viewParams, SystemCmdParameters systemParams) {
        this(janusParams);
        this.systemParams = systemParams;
        this.janusViewParams = viewParams;
    }

    /**
     * Returns the logParser of a given input log
     *
     * @param inputLanguage       file format of the input event log
     * @param inputLogFile        path to the input file of the event log
     * @param eventClassification
     * @return LogParser of the input log
     */
    public static LogParser deriveLogParserFromLogFile(JanusDFGVariantCmdParameters.LogInputEncoding inputLanguage, File inputLogFile, JanusDFGVariantCmdParameters.EventClassification eventClassification, TaskCharArchive taskCharArchive) {
        LogParser logParser = null;
        switch (inputLanguage) {
            case xes:
            case mxml:
                LogEventClassifier.ClassificationType evtClassi = fromInputParamToXesLogClassificationType(eventClassification);
                try {
                    logParser = new XesLogParser(inputLogFile, evtClassi, taskCharArchive);
                } catch (Exception e1) {
                    // TODO Auto-generated catch block
                    e1.printStackTrace();
                }

                // Let us try to free memory from the unused XesDecoder!
                System.gc();
                break;
            case strings:
                try {
//                    logParser = new StringLogParser(inputLogFile, LogEventClassifier.ClassificationType.NAME, taskCharArchive);
                    throw new UnsupportedOperationException("Only XES log supported");
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(1);
                }
                break;
            default:
                throw new UnsupportedOperationException("This encoding (" + inputLanguage + ") is not yet supported");
        }

        return logParser;
    }

    /**
     * Returns the classification type for a given event log encoding
     *
     * @param evtClassInputParam
     * @return
     */
    public static LogEventClassifier.ClassificationType fromInputParamToXesLogClassificationType(JanusDFGVariantCmdParameters.EventClassification evtClassInputParam) {
        switch (evtClassInputParam) {
            case name:
                return LogEventClassifier.ClassificationType.NAME;
            case logspec:
                return LogEventClassifier.ClassificationType.LOG_SPECIFIED;
            default:
                throw new UnsupportedOperationException("Classification strategy " + evtClassInputParam + " not yet implemented");
        }
    }

    public TaskCharArchive getAlphabetDecoder() {
        return eventLog2.getTaskCharArchive();
    }

    /**
     * analyse the time differences between the direct follow succession relations in the two event logs
     *
     * @return
     */
    public List<DFGPermutationResult> checkVariants() {
        DFGtimesVariantAnalysisCore variantAnalysisCore = new DFGtimesVariantAnalysisCore(
                eventLog1, eventLog2, janusParams, janusViewParams);
        return variantAnalysisCore.checkWithGraph();
    }

}
