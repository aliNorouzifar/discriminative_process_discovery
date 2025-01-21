package minerful.logparser;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.deckfour.xes.in.XMxmlGZIPParser;
import org.deckfour.xes.in.XMxmlParser;
import org.deckfour.xes.in.XParser;
import org.deckfour.xes.in.XesXmlGZIPParser;
import org.deckfour.xes.in.XesXmlParser;
import org.deckfour.xes.model.XLog;
import org.deckfour.xes.model.XTrace;

import minerful.concept.AbstractTaskClass;
import minerful.concept.TaskCharArchive;
import minerful.io.encdec.TaskCharEncoderDecoder;

public class XesLogParser extends AbstractLogParser implements LogParser {
    protected XParser parser;
    protected XesEventClassifier xesEventClassifier;
    protected List<XLog> xLogs = null;

	protected XesLogParser(TaskCharEncoderDecoder taChaEncoDeco,
			TaskCharArchive taskCharArchive,
			List<LogTraceParser> traceParsers,
			Integer startingTrace,
			Integer subLogLength,
			XParser parser,
			XesEventClassifier xesEventClassifier,
			List<XLog> xLogs) {
		super(taChaEncoDeco, taskCharArchive, traceParsers, startingTrace, subLogLength);
		this.parser = parser;
		this.xesEventClassifier = xesEventClassifier;
		this.xLogs = xLogs;
	}

    private void init(
    		LogEventClassifier.ClassificationType evtClassType,
			Integer startingTrace,
			Integer subLogLength) {
        this.traceParsers = new ArrayList<LogTraceParser>();
        this.taChaEncoDeco = new TaskCharEncoderDecoder();
        this.parser = new XesXmlParser();
        this.xesEventClassifier = new XesEventClassifier(evtClassType);

        super.init(startingTrace, subLogLength);
    }

    public XesLogParser(File xesFile,
    		LogEventClassifier.ClassificationType evtClassType) throws Exception {
    	this(xesFile, evtClassType, 0, 0, null);
    }

	public XesLogParser(File xesFile,
						LogEventClassifier.ClassificationType evtClassType,
						TaskCharArchive taskCharArchive) throws Exception {
		this(xesFile, evtClassType, 0, 0, taskCharArchive);
	}

    public XesLogParser(
    		File xesFile,
    		LogEventClassifier.ClassificationType evtClassType,
			Integer startingTrace,
			Integer subLogLength,
			TaskCharArchive taskCharArchive) throws Exception {
    	this.init(evtClassType, startingTrace, subLogLength);

        if (!this.parser.canParse(xesFile)) {
        	this.parser = new XesXmlGZIPParser();
        	if (!this.parser.canParse(xesFile)) {
        		this.parser = new XMxmlParser();
            	if (!this.parser.canParse(xesFile)) {
            		this.parser = new XMxmlGZIPParser();
                	if (!this.parser.canParse(xesFile)) {
                		throw new IllegalArgumentException("Unparsable log file: " + xesFile.getAbsolutePath());
                	}
            	}
        	}
        }

        super.archiveTaskChars(this.parseLog(xesFile), taskCharArchive);

        super.postInit();
	}


    public XesLogParser(XLog xLog,
    		LogEventClassifier.ClassificationType evtClassType) {
    	this(xLog, evtClassType, 0, 0);
    }

    public XesLogParser(
    		XLog xLog,
    		LogEventClassifier.ClassificationType evtClassType,
			Integer startingTrace,
			Integer subLogLength) {
    	this.init(evtClassType, startingTrace, subLogLength);

    	super.archiveTaskChars(this.parseLog(xLog), null);

    	super.postInit();
    }

    @Override
	protected Collection<AbstractTaskClass> parseLog(File xesFile) throws Exception {
        this.xLogs = parser.parse(xesFile);

        for (XLog xLog : xLogs) {
        	this.parseLog(xLog);
        }

        return this.xesEventClassifier.getTaskClasses();
	}

    protected Collection<AbstractTaskClass> parseLog(XLog xLog) {
        XesTraceParser auXTraPar = null;

    	this.xesEventClassifier.addXesClassifiers(xLog.getClassifiers(), xLog);

        for (XTrace trace : xLog) {
        	auXTraPar = new XesTraceParser(trace, this);
        	this.traceParsers.add(auXTraPar);
        }
        return this.xesEventClassifier.getTaskClasses();
    }

 	@Override
	public LogEventClassifier getEventClassifier() {
		return this.xesEventClassifier;
	}

	public XLog getFirstXLog() {
		return this.xLogs.get(0);
	}

	@Override
	protected AbstractLogParser makeACopy(
			TaskCharEncoderDecoder taChaEncoDeco,
			TaskCharArchive taskCharArchive,
			List<LogTraceParser> traceParsers,
			Integer startingTrace,
			Integer subLogLength) {
		return new XesLogParser(taChaEncoDeco, taskCharArchive, traceParsers, startingTrace, subLogLength, parser, xesEventClassifier, xLogs);
	}

    /**
     * Returns a XesLogParser reading the union of the input logs (parsers)
     * BEWARE valid only if the TaskCharArchive and TaskCharEncoderDecoder of the inputs are equivalent!
	 * Otherwise, an exception is risen.
     *\
     * @param xlp1
     * @param xlp2
     * @return
     */
    public static XesLogParser mergeParsersWithEquivalentTaskChars(XesLogParser xlp1, XesLogParser xlp2) {
        if (!xlp1.taChaEncoDeco.equals(xlp2.taChaEncoDeco)){
			throw new IllegalArgumentException("The tasks encoders of the input parsers are different");
		}
		TaskCharEncoderDecoder taChaEncoDeco = xlp1.taChaEncoDeco; // TODO merge the two input logs
        TaskCharArchive taskCharArchive = xlp1.taskCharArchive; // TODO merge the two input logs
        List<LogTraceParser> traceParsers = new ArrayList<>();
        traceParsers.addAll(xlp1.traceParsers);
        traceParsers.addAll(xlp2.traceParsers);
        Integer startingTrace = 0;
        Integer subLogLength = xlp1.length() + xlp2.length();
        XParser parser = new XesXmlParser();
        XesEventClassifier xesEventClassifier= xlp1.xesEventClassifier;
        List<XLog> xLogs=new ArrayList<>();
        XLog mLog= xlp1.getFirstXLog();
        mLog.addAll(xlp2.getFirstXLog());
        xLogs.add(mLog);

        return new XesLogParser(
                taChaEncoDeco,
                taskCharArchive,
                traceParsers,
                startingTrace,
                subLogLength,
                parser,
                xesEventClassifier,
                xLogs);
    }
}