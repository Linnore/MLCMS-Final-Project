package org.vadere.simulator.projects.dataprocessing.datakey;
import org.vadere.util.geometry.shapes.VPoint;
import java.util.List;

public class PedestriansKNearestNeighborData {

    //private double velocity;
    private double sk;
    private List<VPoint> kNN;


    public PedestriansKNearestNeighborData(double sk, List<VPoint> kNN) {
        this.sk = sk;
        this.kNN = kNN;
    }


    public double getSk() {
        return sk;
    }

    public List<VPoint> kNN() {
        return kNN;
    }


    public String[] toStrings() {
        String[] ret = new String[kNN.size() + 1];
        //ret[]=velocity+"";
        ret[0] = sk + "";
        int i = 1;
        for (VPoint v : kNN) {
            ret[i] = v.x + " " + v.y;
            i++;
        }
        return ret;
    }


}
