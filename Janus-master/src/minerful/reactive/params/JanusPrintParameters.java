package minerful.reactive.params;

import minerful.params.ParamsManager;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

public class JanusPrintParameters extends ParamsManager {
    public static final String SUPPRESS_SCREEN_PRINT_OUT_RESULTS_PARAM_NAME = "suppressResultsPrintOut";
    public static final String SUPPRESS_SCREEN_PRINT_OUT_DISCOVERY_STATUS_PARAM_NAME = "suppressDiscoveryStatus";
    public static final String SUPPRESS_SCREEN_PRINT_OUT_MEASURES_STATUS_PARAM_NAME = "suppressMeasuresStatus";
    public static final String SUPPRESS_SCREEN_PRINT_OUT_PERMUTATION_STATUS_PARAM_NAME = "suppressPermutationStatus";

    public static final Boolean DEFAULT_DO_SUPPRESS_SCREEN_PRINT_OUT_RESULTS = false;
    public static final Boolean DEFAULT_DO_SUPPRESS_SCREEN_PRINT_OUT_DISCOVERY_STATUS = false;
    public static final Boolean DEFAULT_DO_SUPPRESS_SCREEN_PRINT_OUT_MEASURES_STATUS = false;
    public static final Boolean DEFAULT_DO_SUPPRESS_SCREEN_PRINT_OUT_PERMUTATION_STATUS = false;

    /**
     * Set this field to <code>true</code> to avoid the results to be printed on terminal.
     */
    public Boolean suppressResultsPrintOut;
    /**
     * Set this field to <code>true</code> to avoid the status bar of discovery to be printed.
     */
    public Boolean suppressDiscoveryStatusPrint;
    /**
     * Set this field to <code>true</code> to avoid the status bar of measuring to be printed.
     */
    public Boolean suppressMeasuresStatusPrint;
    /**
     * Set this field to <code>true</code> to avoid the status bar of permutations to be printed.
     */
    public Boolean suppressPermutationStatusPrint;

    /**
     *
     */
    public JanusPrintParameters() {
        super();
        suppressResultsPrintOut =DEFAULT_DO_SUPPRESS_SCREEN_PRINT_OUT_RESULTS;
        suppressDiscoveryStatusPrint = DEFAULT_DO_SUPPRESS_SCREEN_PRINT_OUT_DISCOVERY_STATUS;
        suppressMeasuresStatusPrint = DEFAULT_DO_SUPPRESS_SCREEN_PRINT_OUT_MEASURES_STATUS;
        suppressPermutationStatusPrint = DEFAULT_DO_SUPPRESS_SCREEN_PRINT_OUT_PERMUTATION_STATUS;
    }


    public JanusPrintParameters(Options options, String[] args) {
        this();
        // parse the command line arguments
        this.parseAndSetup(options, args);
    }

    public JanusPrintParameters(String[] args) {
        this();
        // parse the command line arguments
        this.parseAndSetup(new Options(), args);
    }

    @Override
    protected void setup(CommandLine line) {
        this.suppressResultsPrintOut = line.hasOption(SUPPRESS_SCREEN_PRINT_OUT_RESULTS_PARAM_NAME);
        this.suppressDiscoveryStatusPrint = line.hasOption(SUPPRESS_SCREEN_PRINT_OUT_DISCOVERY_STATUS_PARAM_NAME);
        this.suppressMeasuresStatusPrint = line.hasOption(SUPPRESS_SCREEN_PRINT_OUT_MEASURES_STATUS_PARAM_NAME);
        this.suppressPermutationStatusPrint = line.hasOption(SUPPRESS_SCREEN_PRINT_OUT_PERMUTATION_STATUS_PARAM_NAME);
    }

    @SuppressWarnings("static-access")
    public static Options parseableOptions() {
        Options options = new Options();
        options.addOption(
                Option.builder(SUPPRESS_SCREEN_PRINT_OUT_RESULTS_PARAM_NAME)
                        .longOpt("no-screen-print-out-results")
                        .desc("suppresses the print-out of results" +
                                printDefault(DEFAULT_DO_SUPPRESS_SCREEN_PRINT_OUT_RESULTS))
                        .type(Boolean.class)
                        .build()
        );
        options.addOption(
                Option.builder(SUPPRESS_SCREEN_PRINT_OUT_DISCOVERY_STATUS_PARAM_NAME)
                        .longOpt("no-screen-print-out-discovery-status")
                        .desc("suppresses the print-out of discovery status bar" +
                                printDefault(DEFAULT_DO_SUPPRESS_SCREEN_PRINT_OUT_DISCOVERY_STATUS))
                        .type(Boolean.class)
                        .build()
        );
        options.addOption(
                Option.builder(SUPPRESS_SCREEN_PRINT_OUT_MEASURES_STATUS_PARAM_NAME)
                        .longOpt("no-screen-print-out-measures-status")
                        .desc("suppresses the print-out of measurements status bar" +
                                printDefault(DEFAULT_DO_SUPPRESS_SCREEN_PRINT_OUT_MEASURES_STATUS))
                        .type(Boolean.class)
                        .build()
        );
        options.addOption(
                Option.builder(SUPPRESS_SCREEN_PRINT_OUT_PERMUTATION_STATUS_PARAM_NAME)
                        .longOpt("no-screen-print-out-permutation-status")
                        .desc("suppresses the print-out of permutation test status bar" +
                                printDefault(DEFAULT_DO_SUPPRESS_SCREEN_PRINT_OUT_PERMUTATION_STATUS))
                        .type(Boolean.class)
                        .build()
        );
        return options;
    }
}
