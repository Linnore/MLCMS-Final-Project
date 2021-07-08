
package org.vadere.state.attributes.processor;

public class AttributesKNearestNeighbor extends AttributesProcessor {

    // Variables
    private int measurementAreaId = -1;
    private int kNearestNeigbors = -1;

    // Getter
    public int getMeasurementAreaId() {
        return this.measurementAreaId;
    }

    public int kNearestNeighbors() {
        return this.kNearestNeigbors;
    }

    // Setter
    public void setMeasurementAreaId(int measurementAreaId) {
        checkSealed();
        this.measurementAreaId = measurementAreaId;
    }

    public void setkNearestNeigbors(int kNearestNeighbors) {
        //checkSealed();
        this.kNearestNeigbors = kNearestNeighbors;
    }

}

