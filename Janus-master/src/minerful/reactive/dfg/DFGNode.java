package minerful.reactive.dfg;

import minerful.concept.TaskClass;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Node of a DFG. It contains a task from the event log and all the transition from this task
 */
public class DFGNode {
    TaskClass task;
    Map<DFGNode, DFGTransition> outgoingTransitions; // destination node-> transition from this node to the destination

    /**
     * Constructor. Build a node for the given task with no transitions.
     *
     * @param task
     */
    public DFGNode(TaskClass task) {
        this.task = task;
        this.outgoingTransitions = new HashMap<>();
    }

    /**
     * Create or update a transition from this node to the destination one with a certain time.
     *
     * @param destination
     * @param timeDifference
     */
    public void addTransition(DFGNode destination, long timeDifference) {
//        IF destination already exists, update existing transition
        if (outgoingTransitions.containsKey(destination)) {
            outgoingTransitions.get(destination).update(timeDifference);
        }
//        ELSE create a new transition
        else {
            DFGTransition t = new DFGTransition(this, destination, timeDifference);
            this.outgoingTransitions.put(destination, t);
        }
    }

    @Override
    public String toString() {
        return task.getName();
    }

    //    Equivalence performed only over the task, not the transitions
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DFGNode dfgNode = (DFGNode) o;
        return Objects.equals(task, dfgNode.task);
    }

    @Override
    public int hashCode() {
        return Objects.hash(task);
    }
}