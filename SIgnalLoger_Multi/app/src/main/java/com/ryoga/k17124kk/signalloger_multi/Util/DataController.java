package com.ryoga.k17124kk.signalloger_multi.Util;

import java.util.ArrayList;

public class DataController {


    //サンプルがFILTER_SIZE分だけ入る
    private ArrayList<DataSet> dataSetArrayList;
    //サンプル数
    private int FILTER_SIZE = 5;

    private String filterMode = "";
    private final String FILTER_MODE[] = {"中央値", "移動平均"};


    //中央値のインデックスがどこか
    //FILTER_SIZEが各中央値のindexに対応   1   2  3  4  5  6  7 8  9  10 11  偶数は n/2 +1の場所
//    private final int FILTER_CENTER[] = {1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6};

    private ArrayList<Integer> CENTER;

    private int filterd_rssi = 0;
    private DataSet dataSet;

    private String time = "";
    private StabilityOfSensingSignal stabilityOfSensingSignal;


    public DataController() {
        dataSetArrayList = new ArrayList<>();
        CENTER = new ArrayList<>();
        stabilityOfSensingSignal = new StabilityOfSensingSignal();

    }

    public DataController(DataSet dataSet, String filterMode) {
        this();
        this.dataSet = dataSet;
        this.filterMode = filterMode;
        stabilityOfSensingSignal = new StabilityOfSensingSignal(dataSet.getMemo());
    }


    public String getFilterMode() {
        return filterMode;
    }

    public void setFilterMode(String filterMode) {
        this.filterMode = filterMode;
    }

    public int getFILTER_SIZE() {
        return FILTER_SIZE;
    }

    public void setFILTER_SIZE(int FILTER_SIZE) {
        this.FILTER_SIZE = FILTER_SIZE;

        for (int i = 1; i < FILTER_SIZE * 2; i++) {
            CENTER.add(Integer.valueOf((int) Math.floor(i / 2) + 1));
        }


    }

    public DataSet getDataSet() {
        return dataSet;
    }

    public void setDataSet(DataSet dataSet) {
        this.dataSet = dataSet;
    }


    public void setDataSetRssi(String time, int rssi) {
        this.dataSet.setRssi(rssi);


        if (getFilterMode().equals(FILTER_MODE[0])) {
            lowpathFilter_Center(time, rssi);
//            Log.d("MYE_F_M", "中央値");
        } else if (getFilterMode().equals(FILTER_MODE[1])) {
            lowpathFilter_MoveAve(time, rssi);
//            Log.d("MYE_F_M", "移動平均");
        }

        seachStability();

    }

    public StabilityOfSensingSignal getStabilityOfSensingSignal() {
        return stabilityOfSensingSignal;
    }

    public int getFilterd_rssi() {
        return filterd_rssi;
    }

    public void setFilterd_rssi(int filterd_rssi) {
        this.filterd_rssi = filterd_rssi;
    }


    public void resetDatasetArrayList() {
        dataSetArrayList.clear();
    }


    //==============================================================================================
    //==============================================================================================
    //==============================================================================================

    //ローパスを中央値でかける
    public void lowpathFilter_Center(String time, int rssi) {

        this.time = time;

        dataSet.setRssi(rssi);

        //仮登録用にnewインスタンス
        DataSet ds = new DataSet(dataSet.getUuid(), dataSet.getMajor(), dataSet.getMinor(), dataSet.getMemo());
        ds.setRssi(rssi);
        dataSetArrayList.add(ds);


        //リストのサイズでソート用の配列確保
        int rssi_i[] = new int[dataSetArrayList.size()];

        //ソート用に数値取得
        for (int i = 0; i < dataSetArrayList.size(); i++) {
            rssi_i[i] = dataSetArrayList.get(i).getRssi();
        }


        //サイズ上限なら
        if (dataSetArrayList.size() >= FILTER_SIZE) {
            //数値ソート 昇順
            for (int i = 0; i < FILTER_SIZE; ++i) {
                for (int j = i + 1; j < FILTER_SIZE; ++j) {
                    if (rssi_i[i] < rssi_i[j]) {
                        rssi_i[i] = rssi_i[i] + rssi_i[j];
                        rssi_i[j] = rssi_i[i] - rssi_i[j];
                        rssi_i[i] = rssi_i[i] - rssi_i[j];

                    }
                }
            }

            //中央値取得してセット
            setFilterd_rssi(rssi_i[CENTER.get(dataSetArrayList.size() - 1)]);


            //リストの先頭を削除して詰める
            dataSetArrayList.remove(0);

        } else {

            if (dataSetArrayList.size() >= 2) {
                //数値ソート 昇順
                for (int i = 0; i < dataSetArrayList.size(); ++i) {
                    for (int j = i + 1; j < dataSetArrayList.size(); ++j) {
                        if (rssi_i[i] < rssi_i[j]) {
                            rssi_i[i] = rssi_i[i] + rssi_i[j];
                            rssi_i[j] = rssi_i[i] - rssi_i[j];
                            rssi_i[i] = rssi_i[i] - rssi_i[j];
                        }
                    }
                    //中央値取得してセット
                    setFilterd_rssi(rssi_i[CENTER.get(dataSetArrayList.size() - 1) - 1]);
                }
            } else {
                //そのままセット
                setFilterd_rssi(ds.getRssi());
            }


        }

    }
    //==============================================================================================
    //==============================================================================================
    //==============================================================================================


    //ローパスフィルタを移動平均でかける
    public void lowpathFilter_MoveAve(String time, int rssi) {
        this.time = time;
        dataSet.setRssi(rssi);

        //仮登録用にnewインスタンス
        DataSet ds = new DataSet(dataSet.getUuid(), dataSet.getMajor(), dataSet.getMinor(), dataSet.getMemo());
        ds.setRssi(rssi);
        dataSetArrayList.add(ds);


        int sum = 0;
        int ave = 0;


        //サイズ上限なら
        if (dataSetArrayList.size() >= FILTER_SIZE) {


            for (DataSet dataSet : dataSetArrayList) {
                sum += Math.abs(dataSet.getRssi());
            }
            ave = Math.round(sum / dataSetArrayList.size()) * -1;

            //平均値をセット
            setFilterd_rssi(ave);


            //リストの先頭を削除して詰める
            dataSetArrayList.remove(0);

        } else {

            if (dataSetArrayList.size() >= 2) {
                for (DataSet dataSet : dataSetArrayList) {
                    sum += Math.abs(dataSet.getRssi());
                }
                ave = Math.round(sum / dataSetArrayList.size()) * -1;

                //平均値をセット
                setFilterd_rssi(ave);


            } else {
                //そのままセット
                setFilterd_rssi(ds.getRssi());
            }

        }


    }


    //==============================================================================================
    //==============================================================================================
    //==============================================================================================

    //安定センシング区間の判定とネガポジ判定
    public void seachStability() {
        stabilityOfSensingSignal.find_Stability_of_Sensing_Signal(time, filterd_rssi);
    }


    @Override
    public String toString() {
        return "DataContoroller{" +
                "dataSet=" + dataSet +
                '}';
    }
}
