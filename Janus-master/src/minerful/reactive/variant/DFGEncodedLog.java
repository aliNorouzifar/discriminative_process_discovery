package minerful.reactive.variant;

import minerful.logparser.LogTraceParser;
import minerful.logparser.XesEventParser;
import minerful.logparser.XesLogParser;
import minerful.logparser.XesTraceParser;
import org.deckfour.xes.model.XAttributeTimestamp;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class DFGEncodedLog {
    public List<List<DFGEncodedEvent>> traces;

    XesLogParser eventLogParser;

    /**
     * Encode the given XES event log
     *
     * @param eventLogParser
     */
    public DFGEncodedLog(XesLogParser eventLogParser) {
        this.eventLogParser = eventLogParser;
        traces = new ArrayList<>(eventLogParser.length());
        for (Iterator<LogTraceParser> logIterator = eventLogParser.traceIterator(); logIterator.hasNext(); ) {
            XesTraceParser traceParser = (XesTraceParser) logIterator.next();
            traceParser.init();  // otherwise, if the trace was already read, the iterator is pointing to the end of the trace
            List<DFGEncodedEvent> currentTrace = new ArrayList<>(traceParser.length());
            while (!traceParser.isParsingOver()) {
                XesEventParser current = (XesEventParser) traceParser.parseSubsequent();
//                    Keep only the seconds, not the milliseconds
                currentTrace.add(new DFGEncodedEvent(
                        current.getEvent().getTaskClass(),
                        TimeUnit.SECONDS.convert(((XAttributeTimestamp) current.xesEvent.getAttributes().get("time:timestamp")).getValue().getTime(), TimeUnit.MILLISECONDS)
                ));
            }
            traces.add(currentTrace);
        }
    }

    /**
     * Shuffles randomly the order of the traces
     */
    public void shuffleTraces() {
        Collections.shuffle(traces);
    }

    /**
     * returns an encoded log merging this and the input encoded logs
     *
     * @param otherLog
     * @return
     */
    public DFGEncodedLog merge(DFGEncodedLog otherLog) {
        DFGEncodedLog result = new DFGEncodedLog(eventLogParser);
        result.traces.addAll(otherLog.traces);
        return result;
    }

    /**
     * returns the number of traces in this encoded log
     *
     * @return
     */
    public int length() {
        return traces.size();
    }
}
