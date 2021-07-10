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

/**
 * This processor computes
 *
 * @author: Oliver Beck, adjustments: Li Junle
 *
 * This processor calculates the k-nearest neighbors with a Priority Queue data structure (underlying heap).
 * It uses the TimestepPedestrianIdKey as Key and a self-defined Class PedestriansKNearestNeighborData as value.
 * It outputs a file with the headers timeStep pedestrianId sk-PID kNearestNeighbors-PID. Note that underneath the headline
 * kNearestNeighbors-PID, all the xi yi coordinates of the nearest k neighbors are contained. For the attributes we created
 * a class AttributesKNearestNeighbor(), so that the measurement area ID and the desired number of neighbors can be passed.
 */
@DataProcessorClass()
public class KNearestNeighborProcessor extends DataProcessor<TimestepPedestrianIdKey, PedestriansKNearestNeighborData> {

    public KNearestNeighborProcessor() {
        super("sk", "kNearestNeighbors");
    }

    private MeasurementArea measurementArea;

    private int kNearestNeighbors;

    /** note that the k-nearest neighbors is only calculated for pedestrians in a measurement area (but with neighbors that might be outside)**/
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

    /**
     *
     * @param state: calculated every state
     *
     * the doUpdate mantains the priority queue and calculates the different calues and passes them on as needed
     */
    @Override
    protected void doUpdate(SimulationState state) {

        //filters for pedestrians in measurement area
        Collection<Pedestrian> pedestriansInMeasurementArea = new ArrayList<>();
        for (Pedestrian p : state.getTopography().getPedestrianDynamicElements().getElements()) {
            if (measurementArea.getShape().contains(p.getPosition())) {
                pedestriansInMeasurementArea.add(p);
            }
        }

        //loop over all pedestrians, so that each of them gets their kneighbors etc calculated
        for (Pedestrian p : pedestriansInMeasurementArea) {

            PriorityQueue<DistCoords> pq = new PriorityQueue(kNearestNeighbors, new DistCoordsComparator());

            //loop over the neighborhood for every pedestrian
            for (Pedestrian neighbor : state.getTopography().getPedestrianDynamicElements().getElements()) {
                if (p.equals(neighbor)) {
                    continue;
                }

                double distance = getDistance(p.getPosition(), neighbor.getPosition());
                VPoint relativeCoords = new VPoint(neighbor.getPosition().x - p.getPosition().x, neighbor.getPosition().y - p.getPosition().y);

                //relative coords and distance to the pedestrian form a data we mantain in the priority queue
                DistCoords tmp = new DistCoords(distance, relativeCoords);

                //building priority queue until it reahes the desired size of the neighborhood
                if (pq.size() < kNearestNeighbors){
                    pq.add(tmp);
                    continue;
                }

                //if the neighbor is further away than the momentarily k-nearest neighbors, we skip this one
                if (tmp.distance >= pq.peek().distance){
                    continue;
                }

                //discard the k-nearest neighbor for a neighbor that is nearer, in the priority queue
                pq.poll();
                pq.add(tmp);
            }
            //calculate the value for the meanSpacingDistanceColumn with a dedicated function
            double sk = calculateMeanSpacingDistance(p, pq);

            //puting the generated data into the desired dataformat
            ArrayList<VPoint> kneighbors = priorityQueueToArrayList(pq);
            PedestriansKNearestNeighborData kNN = new PedestriansKNearestNeighborData(sk, kneighbors);

            //putting the generated data for this pedestrian into the output file (key,value)
            putValue(new TimestepPedestrianIdKey(state.getStep(), p.getId()), kNN);
        }


    }
    /** calculating "sk" the MeanSpacingDistance **/
    private double calculateMeanSpacingDistance(Pedestrian p, PriorityQueue<DistCoords> pq) {
        double ret = 0;
        for (DistCoords dc : pq) {
            ret += dc.distance;
        }
        ret /= pq.size();
        return ret;
    }

    /** needed for actual output **/
    public String[] toStrings(final TimestepPedestrianIdKey key) {
        return this.hasValue(key) ? this.getValue(key).toStrings() : new String[]{"N/A", "N/A"};
    }

    /** helperfunction to pass the priority queue, resulting in a arraylist from smallest to largest distance **/
    private ArrayList<VPoint> priorityQueueToArrayList(PriorityQueue<DistCoords> pq) {
        ArrayList<VPoint> list = new ArrayList<>();
        for (int i = 0; i < this.kNearestNeighbors; i++) {
            DistCoords elem = pq.poll();
            if (elem!=null) {
                list.add(elem.coords);
            }
        }
        Collections.reverse(list);
        return list;

    }

    /** simple euclidian distance **/
    private double getDistance(VPoint pos1, VPoint pos2) {

        return Math.sqrt(Math.pow((pos1.x - pos2.x), 2) + Math.pow((pos1.y - pos2.y), 2));
    }

    /** standard getValue **/
    @Override
    public PedestriansKNearestNeighborData getValue(TimestepPedestrianIdKey key) {
        PedestriansKNearestNeighborData val = super.getValue(key);
        return val;
    }

    /**helper datastructure used in the priority queue **/
    private class DistCoords {
        public double distance;
        public VPoint coords;

        public DistCoords(double distance, VPoint coords) {
            this.distance = distance;
            this.coords = coords;
        }
    }

    /** used for the priority queue, turns it into a max heap regarding the distance just as we need **/
    private class DistCoordsComparator implements Comparator<DistCoords> {
        @Override
        public int compare(DistCoords a, DistCoords b) {
            return a.distance < b.distance ? 1 : a.distance == b.distance ? 0 : -1;
        }

    }
}

