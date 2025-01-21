package minerful.reactive.variant;

import minerful.concept.AbstractTaskClass;

/**
 * Event object containing only the task class and the timestamp in seconds
 */
public class DFGEncodedEvent {
    public AbstractTaskClass eventsSequence;
    public long timesSequence;

    /**
     * @param eventsSequence
     * @param timesSequence
     */
    public DFGEncodedEvent(AbstractTaskClass eventsSequence, long timesSequence) {
        this.eventsSequence = eventsSequence;
        this.timesSequence = timesSequence;
    }
}
