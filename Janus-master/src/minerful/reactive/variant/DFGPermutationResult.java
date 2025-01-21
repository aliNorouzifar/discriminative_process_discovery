package minerful.reactive.variant;

public class DFGPermutationResult {
    public String sourceNode; // source node of the transition
    public String destinationNode; // destination node of the transition
    public String kind; //AVG/MIN/MAX
    public double pValue;
    public double diff;
    public double log1Value;
    public double log2Value;

    /**
     * @param sourceNode
     * @param destinationNode
     * @param kind
     * @param pValue
     * @param diff
     * @param log1Value
     * @param log2Value
     */
    public DFGPermutationResult(String sourceNode, String destinationNode, String kind, double pValue, double diff, double log1Value, double log2Value) {
        this.sourceNode = sourceNode;
        this.destinationNode = destinationNode;
        this.kind = kind;
        this.pValue = pValue;
        this.diff = diff;
        this.log1Value = log1Value;
        this.log2Value = log2Value;
    }

    @Override
    public String toString() {
        return sourceNode + "-->" + destinationNode + ' ' +
                kind + ' ' +
                ", pValue=" + pValue +
                ", diff=" + diff +
                ", log1Value=" + log1Value +
                ", log2Value=" + log2Value;
    }
}
