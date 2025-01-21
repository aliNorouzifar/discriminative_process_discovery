package minerful.reactive.dfg;

/**
 * Transition of a DFG from a node to another with its time stats.
 * It stores:
 *      - the number of time the transition has been traversed in the event log
 *      - the average time required for the transition
 *      - the maximum/minimum time taken by this transition in the event log
 */
public class DFGTransition {
    DFGNode source;
    DFGNode destination;

    public float getTimeAvg() {
        return timeAvg;
    }

    public long getTimeSum() {
        return timeSum;
    }

    public long getTimeMin() {
        return timeMin;
    }

    public long getTimeMax() {
        return timeMax;
    }

    public int getCounter() {
        return counter;
    }

    float timeAvg; // average of the time taken by this transition in the event log
    long timeSum; // sum of the total time taken by this transition in the event log
    long timeMin; // minimum time taken by this transition in the event log
    long timeMax; // maximum time taken by this transition in the event log
    int counter; // number of occurrences of this transition in the event log

    /**
     * Constructor given the source and destination nodes and their time difference
     *
     * @param source
     * @param destination
     * @param timeDiff
     */
    public DFGTransition(DFGNode source, DFGNode destination, long timeDiff) {
        this.source = source;
        this.destination = destination;
        this.counter = 1;
        this.timeSum = timeDiff;
        this.timeMin = timeDiff;
        this.timeMax = timeDiff;
    }


    /**
     * Compute the current average and return it in output
     *
     * @return
     */
    public float computeAverage() {
        this.timeAvg = (float) this.timeSum / this.counter;
        return this.timeAvg;
    }

    /**
     * update the current counters given a new transition occurrence
     *
     * @param newTimeDifference
     */
    public void update(long newTimeDifference) {
        this.timeSum += newTimeDifference;
        this.timeMin = Math.min(newTimeDifference, this.timeMin);
        this.timeMax = Math.max(newTimeDifference, this.timeMax);
        counter++;
    }


    /**
     * update the current counters given a new transition occurrence and compute immediately the new average
     *
     * @param newTimeDifference
     */
    public void updateWithAverage(long newTimeDifference) {
        update(newTimeDifference);
        computeAverage();
    }

    @Override
    public String toString() {
        return "[" + source + "]-->[" + destination + "]{" +
                "avg=" + timeAvg +
                ", sum=" + timeSum +
                ", min=" + timeMin +
                ", max=" + timeMax +
                ", counter=" + counter +
                '}';
    }
}
