package minerful.reactive.params;

import minerful.params.ParamsManager;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

import java.io.File;

public class JanusDFGVariantCmdParameters extends ParamsManager {
    //      DFG Variant specific
    public static final String INPUT_LOGFILE_1_PATH_PARAM_NAME = "iLF1";  // first log variant to analyse
    public static final String INPUT_LOGFILE_2_PATH_PARAM_NAME = "iLF2";  // second log variant to analyse
    public static final String P_VALUE_NAME = "pValue";  // p-value treshold for statistical relevance of the results. default: 0.01
    public static final Double DEFAULT_P_VALUE = 0.01;  // p-value treshold for statistical relevance of the results. default: 0.01
    public static final String DIFFERENCE_THRESHOLD_NAME = "differenceThreshold";  // threshold for the time(sec.) difference between the variants to be considered relevant. default= 0.0
    public static final Double DEFAULT_DIFFERENCE_THRESHOLD = 0.0;  // threshold for the time difference to consider it relevant, default: "0.0"
//    public static final String BEST_N_RESULTS_NAME = "bestNresults";  // number of rules in the TOP result list. default= 10
//    public static final Integer DEFAULT_BEST_N_RESULTS_VALUE = 10;  // number of rules in the TOP result list. default= 10
    public static final String N_PERMUTATIONS_PARAM_NAME = "permutations";
    public static final Integer DEFAULT_N_PERMUTATIONS = 1000;  // default number of permutations
    public static final String OUTPUT_FILE_CSV_PARAM_NAME = "oCSV";
    public static final String OUTPUT_FILE_JSON_PARAM_NAME = "oJSON";
    public static final String OUTPUT_KEEP_FLAG_NAME = "oKeep";
    public static final String P_VALUE_ADJUSTMENT_METHOD_PARAM_NAME = "pValueAdjustment";
    public static final JanusDFGVariantCmdParameters.PValueAdjustmentMethod DEFAULT_P_VALUE_ADJUSTMENT_METHOD = JanusDFGVariantCmdParameters.PValueAdjustmentMethod.hb;
    //      Log managing fom MINERful
    public static final EventClassification DEFAULT_EVENT_CLASSIFICATION = EventClassification.name;
    public static final LogInputEncoding DEFAULT_INPUT_LOG_ENCODING = LogInputEncoding.xes;
    public static final String INPUT_LOG_1_ENCODING_PARAM_NAME = "iLE1";  // second log variant to analyse
    public static final String INPUT_LOG_2_ENCODING_PARAM_NAME = "iLE2";  // second log variant to analyse
    public static final String EVENT_CLASSIFICATION_PARAM_NAME = "iLClassif";
    public static final String ENCODE_OUTPUT_TASKS_FLAG = "encodeTasksFlag";
    public enum LogInputEncoding {
        /**
         * For XES logs (also compressed)
         */
        xes,
        /**
         * For MXML logs (also compressed)
         */
        mxml,
        /**
         * For string-encoded traces, where each character is assumed to be a task symbol
         */
        strings;

    }

    public enum EventClassification {
        name, logspec
    }

    public enum PValueAdjustmentMethod {
        /**
         * Do not apply correction
         */
        none,
        /**
         * Holm-Bonferroni
         */
        hb,
        /**
         * Benjamini-Hochberg
         */
        bh
    }

    /**
     * file of the first log variant to analyse
     */
    public File inputLogFile1;
    /**
     * file of the second log variant to analyse
     */
    public File inputLogFile2;
    /**
     * Encoding language for the first input event log
     */
    public LogInputEncoding inputLogLanguage1;
    /**
     * Encoding language for the second input event log
     */
    public LogInputEncoding inputLogLanguage2;
    /**
     * Classification policy to relate events to event classes, that is the task names
     */
    public EventClassification eventClassification;
    /**
     * p-value treshold for statistical relevance of the results. default: 0.01
     */
    public double pValue;
    /**
     * threshold for the difference of the variants constraints measurement to be considered relevant, default: "0.0"
     */
    public double differenceThreshold;
    /**
     * number of permutations to perform, default: 1000
     */
    public int nPermutations;
    /**
     * output file in CSV format
     */
    public File outputCvsFile;
    /**
     * output file in JSON format
     */
    public File outputJsonFile;
    /**
     * keep the irrelevant results in output
     */
    public boolean oKeep;
    /**
     * Flag if the output tasks/events should be encoded (e.g., A B C D E...) or not (original names as in log)
     **/
    public boolean encodeOutputTasks;
    /**
     * method to adjust the pValues to address the Multiple test problem. Default=hb
     **/
    public JanusDFGVariantCmdParameters.PValueAdjustmentMethod pValueAdjustmentMethod;

    public JanusDFGVariantCmdParameters() {
        super();
        this.inputLogLanguage1 = DEFAULT_INPUT_LOG_ENCODING;
        this.inputLogLanguage2 = DEFAULT_INPUT_LOG_ENCODING;
        this.eventClassification = DEFAULT_EVENT_CLASSIFICATION;
        this.inputLogFile1 = null;
        this.inputLogFile2 = null;
        this.pValue = DEFAULT_P_VALUE;
        this.nPermutations = DEFAULT_N_PERMUTATIONS;
        this.outputCvsFile = null;
        this.outputJsonFile = null;
        this.oKeep = false;
        this.encodeOutputTasks = false;
        this.differenceThreshold = DEFAULT_DIFFERENCE_THRESHOLD;
        this.pValueAdjustmentMethod = DEFAULT_P_VALUE_ADJUSTMENT_METHOD;
    }

    public JanusDFGVariantCmdParameters(Options options, String[] args) {
        this();
        // parse the command line arguments
        this.parseAndSetup(options, args);
    }

    public JanusDFGVariantCmdParameters(String[] args) {
        this();
        // parse the command line arguments
        this.parseAndSetup(new Options(), args);
    }

    @Override
    protected void setup(CommandLine line) {
        this.inputLogFile1 = openInputFile(line, INPUT_LOGFILE_1_PATH_PARAM_NAME);
        this.inputLogFile2 = openInputFile(line, INPUT_LOGFILE_2_PATH_PARAM_NAME);

        this.inputLogLanguage1 = LogInputEncoding.valueOf(
                line.getOptionValue(
                        INPUT_LOG_1_ENCODING_PARAM_NAME,
                        this.inputLogLanguage1.toString()
                )
        );
        this.inputLogLanguage2 = LogInputEncoding.valueOf(
                line.getOptionValue(
                        INPUT_LOG_2_ENCODING_PARAM_NAME,
                        this.inputLogLanguage2.toString()
                )
        );

        this.eventClassification = EventClassification.valueOf(
                line.getOptionValue(
                        EVENT_CLASSIFICATION_PARAM_NAME,
                        this.eventClassification.toString()
                )
        );

        this.pValue = Double.parseDouble(
                line.getOptionValue(
                        P_VALUE_NAME,
                        Double.toString(this.pValue)
                )
        );

        this.differenceThreshold = Double.parseDouble(
                line.getOptionValue(
                        DIFFERENCE_THRESHOLD_NAME,
                        Double.toString(this.differenceThreshold)
                )
        );

        this.nPermutations = Integer.parseInt(
                line.getOptionValue(
                        N_PERMUTATIONS_PARAM_NAME,
                        Integer.toString(this.nPermutations)
                )
        );
        this.outputCvsFile = openOutputFile(line, OUTPUT_FILE_CSV_PARAM_NAME);
        this.outputJsonFile = openOutputFile(line, OUTPUT_FILE_JSON_PARAM_NAME);
        this.oKeep = line.hasOption(OUTPUT_KEEP_FLAG_NAME);
        this.inputLogFile1 = openInputFile(line, INPUT_LOGFILE_1_PATH_PARAM_NAME);
        this.inputLogFile2 = openInputFile(line, INPUT_LOGFILE_2_PATH_PARAM_NAME);

        this.encodeOutputTasks = line.hasOption(OUTPUT_KEEP_FLAG_NAME);

        this.pValueAdjustmentMethod = JanusDFGVariantCmdParameters.PValueAdjustmentMethod.valueOf(
                line.getOptionValue(
                        P_VALUE_ADJUSTMENT_METHOD_PARAM_NAME,
                        this.pValueAdjustmentMethod.toString()
                )
        );
    }

    @Override
    public Options addParseableOptions(Options options) {
        Options myOptions = listParseableOptions();
        for (Object myOpt : myOptions.getOptions())
            options.addOption((Option) myOpt);
        return options;
    }

    @Override
    public Options listParseableOptions() {
        return parseableOptions();
    }

    @SuppressWarnings("static-access")
    public static Options parseableOptions() {
        Options options = new Options();
        options.addOption(
                Option.builder(INPUT_LOGFILE_1_PATH_PARAM_NAME)
                        .hasArg().argName("path")
//                .isRequired(true) // Causing more problems than not
                        .longOpt("in-log-1-file")
                        .desc("path to read the log file from")
                        .type(String.class)
                        .build()
        );
        options.addOption(
                Option.builder(INPUT_LOGFILE_2_PATH_PARAM_NAME)
                        .hasArg().argName("path")
//                .isRequired(true) // Causing more problems than not
                        .longOpt("in-log-2-file")
                        .desc("path to read the log file from")
                        .type(String.class)
                        .build()
        );
        options.addOption(
                Option.builder(INPUT_LOG_1_ENCODING_PARAM_NAME)
                        .hasArg().argName("language")
                        .longOpt("in-log-1-encoding")
                        .desc("input encoding language " + printValues(LogInputEncoding.values())
                                + printDefault(fromEnumValueToString(DEFAULT_INPUT_LOG_ENCODING)))
                        .type(String.class)
                        .build()
        );
        options.addOption(
                Option.builder(INPUT_LOG_2_ENCODING_PARAM_NAME)
                        .hasArg().argName("language")
                        .longOpt("in-log-2-encoding")
                        .desc("input encoding language " + printValues(LogInputEncoding.values())
                                + printDefault(fromEnumValueToString(DEFAULT_INPUT_LOG_ENCODING)))
                        .type(String.class)
                        .build()
        );
        options.addOption(
                Option.builder(EVENT_CLASSIFICATION_PARAM_NAME)
                        .hasArg().argName("class")
                        .longOpt("in-log-evt-classifier")
                        .desc("event classification (resp., by activity name, or according to the log-specified pattern) " + printValues(EventClassification.values())
                                + printDefault(fromEnumValueToString(DEFAULT_EVENT_CLASSIFICATION)))
                        .type(String.class)
                        .build()
        );
        options.addOption(
                Option.builder(DIFFERENCE_THRESHOLD_NAME)
                        .hasArg().argName("number")
                        .longOpt("difference-threshold")
                        .desc("threshold for the difference of the variants constraints measurement to be considered relevant, default: 0.00")
                        .type(Double.class)
                        .build()
        );
        options.addOption(
                Option.builder(P_VALUE_NAME)
                        .hasArg().argName("number")
                        .longOpt("p-value")
                        .desc("p-value threshold for statistical relevance of the results. default: 0.01")
                        .type(Double.class)
                        .build()
        );
        options.addOption(
                Option.builder(N_PERMUTATIONS_PARAM_NAME)
                        .hasArg().argName("number")
                        .longOpt("number-of-permutations")
                        .desc("number of permutations to perform during the statistical test. default: 1000")
                        .type(Double.class)
                        .build()
        );
        options.addOption(
                Option.builder(OUTPUT_FILE_CSV_PARAM_NAME)
                        .hasArg().argName("path")
//                .isRequired(true) // Causing more problems than not
                        .longOpt("out-csv-file")
                        .desc("path to output CSV file")
                        .type(String.class)
                        .build()
        );
        options.addOption(
                Option.builder(OUTPUT_FILE_JSON_PARAM_NAME)
                        .hasArg().argName("path")
//                .isRequired(true) // Causing more problems than not
                        .longOpt("out-json-file")
                        .desc("path to output JSON file")
                        .type(String.class)
                        .build()
        );
        options.addOption(
                Option.builder(OUTPUT_KEEP_FLAG_NAME)
//                .isRequired(true) // Causing more problems than not
                        .longOpt("output-keep")
                        .desc("keep irrelevant results in output")
                        .type(Boolean.class)
                        .build()
        );
        options.addOption(
                Option.builder(ENCODE_OUTPUT_TASKS_FLAG)
//                .isRequired(true) // Causing more problems than not
                        .longOpt("flag-encoding-tasks")
                        .desc("Flag if the output tasks/events should be encoded")
                        .type(Boolean.class)
                        .build()
        );
        options.addOption(
                Option.builder(P_VALUE_ADJUSTMENT_METHOD_PARAM_NAME)
                        .hasArg().argName("language")
                        .longOpt("p-value-adjustment-method")
                        .desc("pValue adjustment methods: Holm-Bonferroni, Benjamini-Hochberg " + printValues(JanusDFGVariantCmdParameters.PValueAdjustmentMethod.values())
                                + printDefault(fromEnumValueToString(DEFAULT_P_VALUE_ADJUSTMENT_METHOD)))
                        .type(String.class)
                        .build()
        );
        return options;
    }
}
