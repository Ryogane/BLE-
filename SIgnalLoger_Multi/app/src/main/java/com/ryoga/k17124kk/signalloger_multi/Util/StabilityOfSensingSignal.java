package com.ryoga.k17124kk.signalloger_multi.Util;


//安定センシング区間
//
//  εチューブの考え方による安定センシング区間の判定
//

import android.util.Log;

public class StabilityOfSensingSignal {

    //安定センシング区間かどうか判断用の誤差範囲 +-
    private final int range = 7;
    private final int TimeThreshold = 5;//s　閾時間
    //安定センシング区間同士の比較差
    private final int stabilityRange = 10;


    private StabilityData beforeStabilityData;//１つ前の安定区間データ
    private StabilityData currentStabilityData;//現在の安定区間データ

    private String baseTime = "0";
    private String startTime = "0";
    private String endTime = "0";


    private int baseRssi = 0;
    private int currentRssi = 0;

    private boolean isStability = false;//安定->true  変化->false


    private File_ReadWriter file_readWriter;
    private String memo;


    public StabilityOfSensingSignal() {
        file_readWriter = new File_ReadWriter();
        beforeStabilityData = new StabilityData();
        currentStabilityData = new StabilityData();
    }

    public StabilityOfSensingSignal(String memo) {
        this();

        this.memo = memo;
    }

    public StabilityOfSensingSignal(int rssi, String memo) {
        this(memo);
        this.baseRssi = rssi;
    }

    public String getBaseTime() {
        return baseTime;
    }

    public void setBaseTime(String baseTime) {
        this.baseTime = baseTime;
    }

    public String getStartTime() {
        return startTime;
    }

    public void setStartTime(String startTime) {
        this.startTime = startTime;
    }

    public String getEndTime() {
        return endTime;
    }

    public void setEndTime(String endTime) {
        this.endTime = endTime;
    }

    public int getBaseRssi() {
        return baseRssi;
    }

    public void setBaseRssi(int rssi) {
        this.baseRssi = rssi;
    }


    public int getCurrentRssi() {
        return currentRssi;
    }

    public StabilityData getBeforeStabilityData() {
        return beforeStabilityData;
    }

    public StabilityData getCurrentStabilityData() {
        return currentStabilityData;
    }

    public void setCurrentRssi(int currentRssi) {
        this.currentRssi = currentRssi;
    }

    public StabilityData getStabilityData() {
        return currentStabilityData;
    }

    public boolean isStability() {
        return isStability;
    }

    //安定センシング区間の探索
    public void find_Stability_of_Sensing_Signal(String time, int rssi) {

        Log.d("MYE_St", "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
        Log.d("MYE_St", "rssi -> " + rssi);

        //現在のRssi更新
        setCurrentRssi(rssi);

        //誤差の範囲内であれば安定
        if ((baseRssi - range < currentRssi) && (currentRssi < baseRssi + range)) {
            Log.d("MYE_St", "安定 ？？？");


            //一定時間以上安定していれば安定
            //一定時間安定していなければスキップされる
            if ((Long.valueOf(time) - Long.valueOf(startTime)) >= TimeThreshold * 1000) {
                isStability = true;
                Log.d("MYE_St", "安定 : " + time);
            }


            currentStabilityData.addRssi(rssi);


        } else {//範囲外なら不安定

            Log.d("MYE_St", "変化 : " + time);
//            Log.d("MYE_St", stabilityData.toString());


            endTime = getBaseTime();

            //一定時間以上安定していれば書き出し
            //一定時間安定していなければスキップされる
            if ((Long.valueOf(time) - Long.valueOf(startTime)) >= TimeThreshold * 1000) {
                Log.d("MYE_St", "*************安定区間確定***********");

//                beforeStabilityData.switch_NegaPosi();


                if (currentStabilityData.getAve() < beforeStabilityData.getAve() - stabilityRange) {//安定区間の次が前より低い
                    currentStabilityData.setNegaPosi(1);
//                    currentStabilityData.setNegaPosi(0);
                    Log.d("MYE_St", "ネガ->ポジ");

                } else if (currentStabilityData.getAve() > beforeStabilityData.getAve() + stabilityRange) {//安定区間の次が前より高い
                    currentStabilityData.setNegaPosi(0);
//                    currentStabilityData.setNegaPosi(1);
                    Log.d("MYE_St", "ポジ->ネガ");

                } else {
                    Log.d("MYE_St", "何かの間違い");
                }

                beforeStabilityData = new StabilityData(currentStabilityData.getSumRssi(), currentStabilityData.getCount(), currentStabilityData.getAve(), currentStabilityData.getNegaPosi());

                writeLabel();
            }


            startTime = time;
            isStability = false;

            currentStabilityData.resetData();
            currentStabilityData.addRssi(rssi);


        }

        Log.d("MYE_St", "===========================");
        Log.d("MYE_St", "before  :  " + beforeStabilityData.toString());
        Log.d("MYE_St", "current :  " + currentStabilityData.toString());
        Log.d("MYE_St", "===========================");

        //現在のRssiを一つ前のRssiとする
        setBaseRssi(rssi);
        setBaseTime(time);


    }


    private void writeLabel() {
        String line = startTime + "," + endTime + ",";
        Log.d("MYE_St", line);

        if (beforeStabilityData.getNegaPosi() == 0) {
            file_readWriter.writeFile_Label("自動生成-" + memo, line + "ネガティブ");
            Log.d("MYE_St", "ネガティブで書き込み");
        } else {
            file_readWriter.writeFile_Label("自動生成-" + memo, line + "ポジティブ");
            Log.d("MYE_St", "ポジティブで書き込み");
        }

        Log.d("MYE_St", "書き出し");

    }

}
