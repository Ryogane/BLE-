package com.ryoga.k17124kk.signalloger_multi.Util;

public class StabilityData {
    private int sumRssi = -100;
    private int count = 1;
    private int ave = -100;

    private int negaPosi = 0;//ネガティブ-0 ポジティブ-1

    public StabilityData() {
    }

    public StabilityData(int sumRssi, int count, int ave, int negaPosi) {
        this.sumRssi = sumRssi;
        this.count = count;
        this.ave = ave;
        this.negaPosi = negaPosi;
    }


    public int getSumRssi() {
        return sumRssi;
    }

    public void setSumRssi(int sumRssi) {
        this.sumRssi = sumRssi;
    }

    public int getCount() {
        return count;
    }

    public void setCount(int count) {
        this.count = count;
    }

    public void setAve(int ave) {
        this.ave = ave;
    }

    public void addRssi(int rssi) {
        sumRssi += rssi;
        count += 1;

        ave = sumRssi / count;
    }


    public int getAve() {
//        Log.d("MYE_St", ave + "");
        return ave;
    }

    public void setNegaPosi(int negaPosi) {
        this.negaPosi = negaPosi;
    }

    public int getNegaPosi() {
        return negaPosi;
    }

    public void switch_NegaPosi() {
        if (getNegaPosi() == 0) {
            setNegaPosi(1);
        } else {
            setNegaPosi(0);
        }
    }


    public void resetData() {
        sumRssi = 0;
        ave = 0;
        count = 0;
    }


    @Override
    public String toString() {
        return "StabilityData{" +
                "sumRssi=" + sumRssi +
                ", count=" + count +
                ", ave=" + ave +
                ", negaPosi=" + negaPosi +
                '}';
    }
}
