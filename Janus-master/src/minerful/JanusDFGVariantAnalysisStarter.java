package minerful;

import minerful.params.SystemCmdParameters;
import minerful.params.ViewCmdParameters;
import minerful.reactive.io.JanusDFGVariantOutputManagementLauncher;
import minerful.reactive.params.JanusDFGVariantCmdParameters;
import minerful.reactive.params.JanusPrintParameters;
import minerful.reactive.variant.DFGPermutationResult;
import minerful.reactive.variant.DFGtimesVariantAnalysisCore;
import minerful.utils.MessagePrinter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

import java.util.List;

/**
 * Class to start from terminal Janus variant analysis
 */
public class JanusDFGVariantAnalysisStarter extends MinerFulMinerStarter {
    public static MessagePrinter logger = MessagePrinter.getInstance(JanusDFGVariantAnalysisStarter.class);

    @Override
    public Options setupOptions() {
        Options cmdLineOptions = new Options();

        Options systemOptions = SystemCmdParameters.parseableOptions(),
//                outputOptions = OutputModelParameters.parseableOptions(),
//                postPrOptions = PostProcessingCmdParameters.parseableOptions(),
                viewOptions = ViewCmdParameters.parseableOptions(),
                janusViewOptions = JanusPrintParameters.parseableOptions(),
//                chkOptions = CheckingCmdParameters.parseableOptions(),
//                inputLogOptions = InputLogCmdParameters.parseableOptions(),
//                inpuModlOptions = InputModelParameters.parseableOptions(),
                janusOptions = JanusDFGVariantCmdParameters.parseableOptions();

        for (Object opt : systemOptions.getOptions()) {
            cmdLineOptions.addOption((Option) opt);
        }
//        for (Object opt : outputOptions.getOptions()) {
//            cmdLineOptions.addOption((Option) opt);
//        }
//        for (Object opt : postPrOptions.getOptions()) {
//            cmdLineOptions.addOption((Option) opt);
//        }
//        for (Object opt : viewOptions.getOptions()) {
//            cmdLineOptions.addOption((Option) opt);
//        }
        for (Object opt : janusViewOptions.getOptions()) {
            cmdLineOptions.addOption((Option) opt);
        }
//        for (Object opt : chkOptions.getOptions()) {
//            cmdLineOptions.addOption((Option) opt);
//        }
//        for (Object opt : inputLogOptions.getOptions()) {
//            cmdLineOptions.addOption((Option) opt);
//        }
//        for (Object opt : inpuModlOptions.getOptions()) {
//            cmdLineOptions.addOption((Option) opt);
//        }
        for (Object opt : janusOptions.getOptions()) {
            cmdLineOptions.addOption((Option) opt);
        }

        return cmdLineOptions;
    }

    public static void main(String[] args) {
        JanusDFGVariantAnalysisStarter checkStarter = new JanusDFGVariantAnalysisStarter();
        Options cmdLineOptions = checkStarter.setupOptions();

        SystemCmdParameters systemParams =
                new SystemCmdParameters(
                        cmdLineOptions,
                        args);
//        OutputModelParameters outParams =
//                new OutputModelParameters(
//                        cmdLineOptions,
//                        args);
//        PostProcessingCmdParameters preProcParams =
//                new PostProcessingCmdParameters(
//                        cmdLineOptions,
//                        args);
//        CheckingCmdParameters chkParams =
//                new CheckingCmdParameters(
//                        cmdLineOptions,
//                        args);
//        InputLogCmdParameters inputLogParams =
//                new InputLogCmdParameters(
//                        cmdLineOptions,
//                        args);
//        InputModelParameters inpuModlParams =
//                new InputModelParameters(
//                        cmdLineOptions,
//                        args);
//        ViewCmdParameters viewParams =
//                new ViewCmdParameters(
//                        cmdLineOptions,
//                        args);
        JanusPrintParameters janusViewParams =
                new JanusPrintParameters(
                        cmdLineOptions,
                        args);
        JanusDFGVariantCmdParameters janusParams =
                new JanusDFGVariantCmdParameters(
                        cmdLineOptions,
                        args);

        MessagePrinter.configureLogging(systemParams.debugLevel);

        if (systemParams.help) {
            systemParams.printHelp(cmdLineOptions);
            System.exit(0);
        }
        double execTimeStart = System.currentTimeMillis();

        JanusDFGVariantAnalysisLauncher variantAnalysis = new JanusDFGVariantAnalysisLauncher(janusParams, janusViewParams, systemParams);
        List<DFGPermutationResult> results = variantAnalysis.checkVariants();

        new JanusDFGVariantOutputManagementLauncher().manageVariantOutput(
                results,
                janusParams,
                janusViewParams,
                systemParams,
                janusParams.inputLogFile1.getName().substring(0, janusParams.inputLogFile1.getName().lastIndexOf('.')),
                janusParams.inputLogFile2.getName().substring(0, janusParams.inputLogFile2.getName().lastIndexOf('.'))
        );

        double execTimeEnd = System.currentTimeMillis();
        logger.info("Total execution time: " + (execTimeEnd - execTimeStart));
    }

}
