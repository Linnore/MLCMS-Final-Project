package org.vadere.simulator.projects.dataprocessing.processor;

import org.vadere.annotation.factories.dataprocessors.DataProcessorClass;
import org.vadere.simulator.control.simulation.SimulationState;
import org.vadere.simulator.projects.dataprocessing.ProcessorManager;
import org.vadere.simulator.projects.dataprocessing.datakey.PedestriansKNearestNeighborData;
import org.vadere.simulator.projects.dataprocessing.datakey.TimestepPedestrianIdKey;
import org.vadere.state.attributes.processor.AttributesKNearestNeighbor;
import org.vadere.state.attributes.processor.AttributesProcessor;

import org.vadere.state.scenario.MeasurementArea;
import org.vadere.state.scenario.Pedestrian;
import org.vadere.util.geometry.shapes.VPoint;

import java.util.*;

@DataProcessorClass()
public class KNearestNeighborProcessor extends DataProcessor<TimestepPedestrianIdKey, PedestriansKNearestNeighborData> {

    public KNearestNeighborProcessor() {
        super("sk", "kNearestNeighbors");
    }

    private MeasurementArea measurementArea;

    private int kNearestNeighbors;

    @Override
    public void init(final ProcessorManager manager) {
        super.init(manager);
        AttributesKNearestNeighbor processorAttributes = (AttributesKNearestNeighbor) this.getAttributes();

        // manager.getMeasurementArea() throws an exception if area is "null" or not rectangular. Though, no checks required here.
        boolean rectangularAreaRequired = true;
        measurementArea = manager.getMeasurementArea(processorAttributes.getMeasurementAreaId(), rectangularAreaRequired);
        kNearestNeighbors = processorAttributes.kNearestNeighbors();

    }

    @Override
    public AttributesProcessor getAttributes() {
        if (super.getAttributes() == null) {
            setAttributes(new AttributesKNearestNeighbor());
        }

        return super.getAttributes();
    }

    @Override
    protected void doUpdate(SimulationState state) {

        //only for pedestrians in measurement area
        Collection<Pedestrian> pedestriansInMeasurementArea = new ArrayList<>();
        for (Pedestrian p : state.getTopography().getPedestrianDynamicElements().getElements()) {
            if (measurementArea.getShape().contains(p.getPosition())) {
                pedestriansInMeasurementArea.add(p);
            }
        }


        for (Pedestrian p : pedestriansInMeasurementArea) {

            PriorityQueue<DistCoords> pq = new PriorityQueue(kNearestNeighbors, new DistCoordsComparator());

            for (Pedestrian neighbor : state.getTopography().getPedestrianDynamicElements().getElements()) {
                if (p.equals(neighbor)) {
                    continue;
                }

                double distance = getDistance(p.getPosition(), neighbor.getPosition());
                VPoint relativeCoords = new VPoint(neighbor.getPosition().x - p.getPosition().x, neighbor.getPosition().y - p.getPosition().y);
                pq.add(new DistCoords(distance, relativeCoords));

                if (pq.size() > kNearestNeighbors) {
                    pq.poll();
                }

            }
            //double velocity=3.14;
            double sk = calculateMeanSpacingDistance(p, pq);
            ArrayList<VPoint> kneighbors = priorityQueueToArrayList(pq);
            PedestriansKNearestNeighborData kNN = new PedestriansKNearestNeighborData(sk, kneighbors);

            putValue(new TimestepPedestrianIdKey(state.getStep(), p.getId()), kNN);
        }

    }

    //calculating sk
    private double calculateMeanSpacingDistance(Pedestrian p, PriorityQueue<DistCoords> pq) {
        double ret = 0;
        for (DistCoords dc : pq) {
            ret += dc.distance;
        }
        ret /= pq.size();
        return ret;
    }


    public String[] toStrings(final TimestepPedestrianIdKey key) {
        return this.hasValue(key) ? this.getValue(key).toStrings() : new String[]{"N/A", "N/A"};
    }

    private ArrayList<VPoint> priorityQueueToArrayList(PriorityQueue<DistCoords> pq) {
        ArrayList<VPoint> list = new ArrayList<>();
        for (DistCoords dc : pq) {
            //list.add(dc.coords);
            list.add(pq.poll().coords);
        }
        return list;
    }


    private double getDistance(VPoint pos1, VPoint pos2) {

        return Math.sqrt(Math.pow((pos1.x - pos2.x), 2) + Math.pow((pos1.y - pos2.y), 2));
    }


    @Override
    public PedestriansKNearestNeighborData getValue(TimestepPedestrianIdKey key) {
        PedestriansKNearestNeighborData velocity = super.getValue(key);
        return velocity;
    }

    private class DistCoords {
        public double distance;
        public VPoint coords;

        public DistCoords(double distance, VPoint coords) {
            this.distance = distance;
            this.coords = coords;
        }
    }

    private class DistCoordsComparator implements Comparator<DistCoords> {
        @Override
        public int compare(DistCoords a, DistCoords b) {
            return a.distance < b.distance ? 1 : a.distance == b.distance ? 0 : -1;
            //ahdkjashdkasdhaskdhask
        }

    }
}

