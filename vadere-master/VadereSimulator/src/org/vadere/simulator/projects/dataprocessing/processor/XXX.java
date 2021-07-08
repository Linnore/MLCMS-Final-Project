package org.vadere.simulator.projects.dataprocessing.processor;

import org.vadere.annotation.factories.dataprocessors.DataProcessorClass;
import org.vadere.simulator.control.simulation.SimulationState;
import org.vadere.simulator.projects.dataprocessing.ProcessorManager;
import org.vadere.simulator.projects.dataprocessing.datakey.TimestepPedestrianIdKey;
import org.vadere.state.attributes.processor.AttributesKNearestNeighbor;
import org.vadere.state.attributes.processor.AttributesProcessor;

import org.vadere.state.scenario.MeasurementArea;
import org.vadere.state.scenario.Pedestrian;
import org.vadere.util.geometry.shapes.VPoint;

import java.util.ArrayList;
import java.util.Collection;

@DataProcessorClass()
    public class XXX extends DataProcessor<TimestepPedestrianIdKey, ArrayList<VPoint>> {

        private MeasurementArea measurementArea;
        private int kDistance;
        //private PedestrianVelocityDefaultProcessor pedestrianVelocityDefaultProcessor;

        @Override
        public void init(final ProcessorManager manager) {
            super.init(manager);

            AttributesKNearestNeighbor processorAttributes = (AttributesKNearestNeighbor)this.getAttributes();

            // manager.getMeasurementArea() throws an exception if area is "null" or not rectangular. Though, no checks required here.
            boolean rectangularAreaRequired = true;
            measurementArea = manager.getMeasurementArea(processorAttributes.getMeasurementAreaId(), rectangularAreaRequired);
            kDistance = processorAttributes.kNearestNeighbors();

            /**
            pedestrianVelocityDefaultProcessor = (PedestrianVelocityDefaultProcessor) manager.getProcessor(processorAttributes.getPedestrianVelocityDefaultProcessorId());

            if (pedestrianVelocityDefaultProcessor == null) {
                throw new RuntimeException(String.format("PedestrianVelocityDefaultProcessor with index %d does not exist.", processorAttributes.getPedestrianVelocityDefaultProcessorId()));
            }
             **/

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
            /*******/

            //only for pedestrians in measurement area
            Collection<Pedestrian> pedestriansInMeasurementArea=new ArrayList<>();
            for (Pedestrian p : state.getTopography().getPedestrianDynamicElements().getElements()) {
                if(measurementArea.getShape().contains(p.getPosition())){
                    pedestriansInMeasurementArea.add(p);
                }
            }


            for (Pedestrian p : pedestriansInMeasurementArea) {
                ArrayList<VPoint> kneighbor_coords = new ArrayList<>();
                double maxdistance = Double.MIN_VALUE;
                VPoint maxcoords = null;
                for (Pedestrian neighbor : state.getTopography().getPedestrianDynamicElements().getElements()) {
                    if(p.equals(neighbor)){
                        continue;
                    }
                    double distance=getDistance(p.getPosition(),neighbor.getPosition());
                    int k=kDistance;
                    //maybe heapstructure for k nearest element

                    if(kneighbor_coords.size()<k){
                        kneighbor_coords.add(neighbor.getPosition());
                        //biggest neighbor in first k neighbors
                        if(distance>maxdistance){
                            maxdistance=distance;
                            maxcoords=neighbor.getPosition();
                        }
                    } else {
                        //replace biggist k neighbor
                        if(distance<maxdistance){
                            kneighbor_coords.remove(maxcoords);
                            kneighbor_coords.add(neighbor.getPosition());
                            //find new biggist kneighbor
                            maxdistance = Double.MIN_VALUE;
                            maxcoords = null;
                            for (VPoint c: kneighbor_coords){
                                if(getDistance(p.getPosition(),c)>maxdistance){
                                    maxdistance=getDistance(p.getPosition(),c);
                                    maxcoords=c;

                                }
                            }
                        }
                    }
                }
                putValue(new TimestepPedestrianIdKey(state.getStep(), p.getId()), kneighbor_coords);

            }

        }

        private double getDistance(VPoint pos1, VPoint pos2) {

            return Math.sqrt(Math.pow((pos1.x-pos2.x),2)+Math.pow((pos1.y-pos2.y),2));
        }


        @Override
        public ArrayList<VPoint> getValue(TimestepPedestrianIdKey key) {
            ArrayList<VPoint> velocity = super.getValue(key);
            if(velocity == null) {
                velocity = new ArrayList<>();
                //.add(new VPoint(5,3));
            }
            return velocity;
        }
    }

