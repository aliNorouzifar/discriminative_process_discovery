package minerful.reactive.dfg;

import minerful.concept.AbstractTaskClass;
import minerful.concept.TaskClass;
import minerful.logparser.LogTraceParser;
import minerful.logparser.XesEventParser;
import minerful.logparser.XesLogParser;
import minerful.logparser.XesTraceParser;
import minerful.reactive.variant.DFGEncodedLog;
import minerful.reactive.variant.DFGEncodedEvent;
import org.deckfour.xes.model.XAttributeTimestamp;

import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Class representing a DFG (Directly Follow Graph), i.e., storing al the direct transitions in an event log along with their times and counters
 */
public class DFG {
    Map<TaskClass, DFGNode> tasks;


    /**
     * Constructor. Build an empty DFG
     */
    public DFG() {
        this.tasks = new HashMap<>();
    }

    /**
     * Returns the map of tasks nodes contained in the DFG
     *
     * @return
     */
    public Map<TaskClass, DFGNode> getTasks() {
        return tasks;
    }

    /**
     * Computes for all the transitions in the DFG the average time required by the transition
     */
    public void computeAllAverages() {
        for (DFGNode node : tasks.values()) {
            for (DFGTransition tr : node.outgoingTransitions.values()) {
                tr.computeAverage();
            }
        }
    }


    /**
     * Create a transition between two nodes or Update it if already existing
     *
     * @param previousNode
     * @param destinationNode
     * @param timeDiff
     */
    public void addTransition(DFGNode previousNode, DFGNode destinationNode, long timeDiff) {
        this.tasks.putIfAbsent(previousNode.task, previousNode);
        this.tasks.putIfAbsent(destinationNode.task, destinationNode);
        previousNode.addTransition(destinationNode, timeDiff);
    }

    /**
     * Retrun the transition between two nodes if existing in the DFG, NULL otherwise.
     *
     * @param source
     * @param destination
     * @return
     */
    public DFGTransition getTransition(TaskClass source, TaskClass destination) {
        try {
            return tasks.get(source).outgoingTransitions.get(new DFGNode(destination));
        } catch (NullPointerException E) {
            return null;
        }
    }

    public static DFG buildDFGFromEncodedLog(DFGEncodedLog eLog) {
        return buildDFGFromEncodedLog(eLog.traces);
    }

    public static DFG buildDFGFromEncodedLog(List<List<DFGEncodedEvent>> traces) {
        DFG result = new DFG();

        for (List<DFGEncodedEvent> trace : traces) {
            DFGEncodedEvent previous = trace.get(0);
            DFGNode previousNode = result.tasks.getOrDefault(previous.eventsSequence, new DFGNode(previous.eventsSequence));

            DFGNode currentNode;
            boolean flag = true;
            for (DFGEncodedEvent current : trace) {
                if (flag) {
//                    skip first event
                    flag = false;
                    continue;
                }
                currentNode = result.tasks.getOrDefault(current.eventsSequence, new DFGNode(current.eventsSequence));
                long timeDiff = Math.abs(current.timesSequence - previous.timesSequence);

                result.addTransition(previousNode, currentNode, timeDiff);

                previous = current;
                previousNode = currentNode;
            }
        }
        result.computeAllAverages();
        return result;
    }


    /**
     * Return the DFG of a given XES event log. It is expected that the timestamp is stored in the attribute "time:timestamp"
     *
     * @param eventLogParser
     * @return
     */
    public static DFG buildDFGFromXesLogParser(XesLogParser eventLogParser) {
        DFG result = new DFG();

        for (Iterator<LogTraceParser> logIterator = eventLogParser.traceIterator(); logIterator.hasNext(); ) {
            XesTraceParser traceParser = (XesTraceParser) logIterator.next();
            traceParser.init();  // otherwise, if the trace was already read, the iterator is pointing to the end of the trace
            XesEventParser previous = (XesEventParser) traceParser.parseSubsequent();
            AbstractTaskClass previousTaskClass = previous.getEvent().getTaskClass();
            DFGNode previousNode = result.tasks.getOrDefault(previousTaskClass, new DFGNode(previousTaskClass));
            DFGNode currentNode;

            AbstractTaskClass currentTaskClass;

            XesEventParser current;
            Date prevDate;
            Date currDate;
            long timeDiff;
            while (!traceParser.isParsingOver()) {
                current = (XesEventParser) traceParser.parseSubsequent();
                currentTaskClass = current.getEvent().getTaskClass();
                currentNode = result.tasks.getOrDefault(currentTaskClass, new DFGNode(currentTaskClass));
                prevDate = ((XAttributeTimestamp) previous.xesEvent.getAttributes().get("time:timestamp")).getValue();
                currDate = ((XAttributeTimestamp) current.xesEvent.getAttributes().get("time:timestamp")).getValue();
                timeDiff = TimeUnit.SECONDS.convert(Math.abs(currDate.getTime() - prevDate.getTime()), TimeUnit.MILLISECONDS);

                result.addTransition(previousNode, currentNode, timeDiff);

                previous = current;
                previousNode = currentNode;
            }
        }
        result.computeAllAverages();
        return result;
    }


    /**
     * Build a Graphviz graphical representation of the DFG and save it.
     *
     * @param outputPath
     */
    public void toDot(String outputPath) {
        System.out.println("Not yet Implemented");
    }


    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("DFG:\n");
        for (DFGNode task : tasks.values()) {
            result.append("\t " + task.task + "\n");
            for (DFGTransition tran : task.outgoingTransitions.values()) {
                result.append("\t\t" + tran + "\n");
            }
        }
        return result.toString();
    }
}