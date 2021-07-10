package org.vadere.simulator.projects.dataprocessing.datakey;
import org.vadere.util.geometry.shapes.VPoint;
import java.util.List;

/**
 * @autohor Oliver Beck
 *
 * Data structure for the Value needed to fit the <Key,Value> format of the Dataprocessor class
 * containing sk=MeanSpacingDistance from the k nearest neighbors
 */
public class PedestriansKNearestNeighborData {

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

    /**
     *
     * @return is a String array of the calculated columns, note that the length of the xi yi might be variable
     */
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
