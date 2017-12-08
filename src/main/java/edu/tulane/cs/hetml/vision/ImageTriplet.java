package edu.tulane.cs.hetml.vision;

import java.awt.geom.Rectangle2D;

public class ImageTriplet {
    private String imageId;
    private int firstSegId;
    private int secondSegId;
    private String trajector;
    private String landmark;
    private String sp;
    private Rectangle2D trBox;
    private Rectangle2D lmBox;
    private double imageWidth;
    private double imageHeight;
    private double trVectorX; // F1
    private double trVectorY; // F1
    private double trAreawrtLM; // F2
    private double trAspectRatio; // F3
    private double lmAspectRatio; // F3
    private double trAreaBbox; // F4
    private double lmAreaBbox; //F4
    private double iou; // F5
    private double euclideanDistance; // F6
    private double trAreaImage; // F7
    private double lmAreaImage; // F7
    private double above;
    private double below;
    private double left;
    private double right;
    private double intersectionArea;
    private double unionArea;

    public void setTrBox(Rectangle2D trBox) {
        this.trBox = trBox;
    }

    public void setLmBox(Rectangle2D lmBox) {
        this.lmBox = lmBox;
    }

    public void setImageWidth(double imageWidth) {
        this.imageWidth = imageWidth;
    }

    public void setImageHeight(double imageHeight) {
        this.imageHeight = imageHeight;
    }

    public void setTrVectorX(double trVectorX) {
        this.trVectorX = trVectorX;
    }

    public void setTrVectorY(double trVectorY) {
        this.trVectorY = trVectorY;
    }

    public void setTrAreawrtLM(double trAreawrtLM) {
        this.trAreawrtLM = trAreawrtLM;
    }

    public void setTrAspectRatio(double trAspectRatio) {
        this.trAspectRatio = trAspectRatio;
    }

    public void setLmAspectRatio(double lmAspectRatio) {
        this.lmAspectRatio = lmAspectRatio;
    }

    public void setTrAreaBbox(double trAreaBbox) {
        this.trAreaBbox = trAreaBbox;
    }

    public void setLmAreaBbox(double lmAreaBbox) {
        this.lmAreaBbox = lmAreaBbox;
    }

    public void setIou(double iou) {
        this.iou = iou;
    }

    public void setTrAreaImage(double trAreaImage) {
        this.trAreaImage = trAreaImage;
    }

    public void setLmAreaImage(double lmAreaImage) {
        this.lmAreaImage = lmAreaImage;
    }

    public ImageTriplet(String sp, String trajector, String landmark, Rectangle2D trBox, Rectangle2D lmBox, double imageWidth,
                        double imageHeight, double trVectorX, double trVectorY, double trAreawrtLM, double trAspectRatio,
                        double lmAspectRatio, double trAreaBbox, double lmAreaBbox, double iou, double euclideanDistance,
                        double trAreaImage, double lmAreaImage, double above, double below, double left, double right,
                        double intersectionArea, double unionArea) {
        this.setSp(sp);
        this.setTrajector(trajector);
        this.setLandmark(landmark);
        this.trBox = trBox;
        this.lmBox = lmBox;
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.trVectorX = trVectorX;
        this.trVectorY = trVectorY;
        this.trAreawrtLM = trAreawrtLM;
        this.trAspectRatio = trAspectRatio;
        this.lmAspectRatio = lmAspectRatio;
        this.trAreaBbox = trAreaBbox;
        this.lmAreaBbox = lmAreaBbox;
        this.iou = iou;
        this.euclideanDistance = euclideanDistance;
        this.trAreaImage = trAreaImage;
        this.lmAreaImage = lmAreaImage;
        this.above = above;
        this.below = below;
        this.left = left;
        this.right = right;
        this.intersectionArea = intersectionArea;
        this.unionArea = unionArea;
    }

    public ImageTriplet(String imageId, int firstSegId, int secondSegId, Rectangle2D trBox, Rectangle2D lmBox,
                        double imageWidth, double imageHeight, double trVectorX, double trVectorY, double trAreawrtLM, double trAspectRatio,
                        double lmAspectRatio, double trAreaBbox, double lmAreaBbox, double iou, double euclideanDistance,
                        double trAreaImage, double lmAreaImage, double above, double below, double left, double right,
                        double intersectionArea, double unionArea) {
        this.setImageId(imageId);
        this.setFirstSegId(firstSegId);
        this.setSecondSegId(secondSegId);
        this.trBox = trBox;
        this.lmBox = lmBox;
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.trVectorX = trVectorX;
        this.trVectorY = trVectorY;
        this.trAreawrtLM = trAreawrtLM;
        this.trAspectRatio = trAspectRatio;
        this.lmAspectRatio = lmAspectRatio;
        this.trAreaBbox = trAreaBbox;
        this.lmAreaBbox = lmAreaBbox;
        this.iou = iou;
        this.euclideanDistance = euclideanDistance;
        this.trAreaImage = trAreaImage;
        this.lmAreaImage = lmAreaImage;
        this.above = above;
        this.below = below;
        this.left = left;
        this.right = right;
        this.intersectionArea = intersectionArea;
        this.unionArea = unionArea;
    }

    public String getSp() {
        return sp;
    }

    public double getImageWidth() {
        return imageWidth;
    }
    public double getImageHeight() {
        return imageHeight;
    }

    public String getLandmark() {
        return landmark;
    }

    public double getEuclideanDistance() {
        return euclideanDistance;
    }

    public double getIou() {
        return iou;
    }

    public double getLmAreaBbox() {
        return lmAreaBbox;
    }

    public double getLmAreaImage() {
        return lmAreaImage;
    }

    public double getLmAspectRatio() {
        return lmAspectRatio;
    }

    public double getTrAreaBbox() {
        return trAreaBbox;
    }

    public double getTrAreaImage() {
        return trAreaImage;
    }

    public double getTrAreawrtLM() {
        return trAreawrtLM;
    }

    public double getTrAspectRatio() {
        return trAspectRatio;
    }

    public Rectangle2D getLmBox() {
        return lmBox;
    }

    public String getTrajector() {
        return trajector;
    }

    public Rectangle2D getTrBox() {
        return trBox;
    }

    public double getTrVectorX() {
        return trVectorX;
    }

    public double getTrVectorY() {
        return trVectorY;
    }

    public String getImageId() {
        return imageId;
    }

    public void setImageId(String imageId) {
        this.imageId = imageId;
    }

    public int getFirstSegId() {
        return firstSegId;
    }

    public void setFirstSegId(int firstSegId) {
        this.firstSegId = firstSegId;
    }

    public int getSecondSegId() {
        return secondSegId;
    }

    public void setSecondSegId(int secondSegId) {
        this.secondSegId = secondSegId;
    }

    public void setSp(String sp) {
        this.sp = sp;
    }

    public void setTrajector(String trajector) {
        this.trajector = trajector;
    }

    public void setLandmark(String landmark) {
        this.landmark = landmark;
    }

    public double getAbove() {
        return above;
    }

    public void setAbove(double above) {
        this.above = above;
    }

    public double getBelow() {
        return below;
    }

    public void setBelow(double below) {
        this.below = below;
    }

    public double getLeft() {
        return left;
    }

    public void setLeft(double left) {
        this.left = left;
    }

    public double getRight() {
        return right;
    }

    public void setRight(double right) {
        this.right = right;
    }

    public double getIntersectionArea() {
        return intersectionArea;
    }

    public void setIntersectionArea(double intersectionArea) {
        this.intersectionArea = intersectionArea;
    }

    public void setEuclideanDistance(double euclideanDistance) {
        this.euclideanDistance = euclideanDistance;
    }

    public double getUnionArea() {
        return unionArea;
    }

    public void setUnionArea(double unionArea) {
        this.unionArea = unionArea;
    }

}