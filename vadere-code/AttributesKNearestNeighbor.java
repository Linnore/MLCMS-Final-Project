
package org.vadere.state.attributes.processor;

/**
 * @author: Oliver Beck
 * Attribute processor needed for k-nearest Neighbors. It contains Measurement ID and value k for the nearest neighbors.
 *
 */
public class AttributesKNearestNeighbor extends AttributesProcessor {

    // Variables
    private int measurementAreaId = -1;
    private int kNearestNeighbors = -1;

    // Getter
    public int getMeasurementAreaId() {
        return this.measurementAreaId;
    }

    public int kNearestNeighbors() {
        return this.kNearestNeighbors;
    }

    // Setter
    public void setMeasurementAreaId(int measurementAreaId) {
        checkSealed();
        this.measurementAreaId = measurementAreaId;
    }

    public void setkNearestNeigbors(int kNearestNeighbors) {
        //checkSealed();
        this.kNearestNeighbors = kNearestNeighbors;
    }

}

