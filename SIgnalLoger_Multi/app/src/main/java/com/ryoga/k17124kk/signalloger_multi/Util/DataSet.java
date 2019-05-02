package com.ryoga.k17124kk.signalloger_multi.Util;

import java.io.Serializable;

public class DataSet implements Serializable {


    private final String KEY = "DATASET_KEY";

    private String uuid;
    private String major;
    private String minor;
    private String memo;
    private int rssi = 0;

    private boolean exist = false;

    public DataSet() {

    }

    public DataSet(String uuid, String major, String minor) {
        this.uuid = uuid;
        this.major = major;
        this.minor = minor;
    }

    public DataSet(String uuid, String major, String minor, String memo) {
        this.uuid = uuid;
        this.major = major;
        this.minor = minor;
        this.memo = memo;
    }

    public String getUuid() {
        return uuid;
    }

    public void setUuid(String uuid) {
        this.uuid = uuid;
    }

    public String getMajor() {
        return major;
    }

    public void setMajor(String major) {
        this.major = major;
    }

    public String getMinor() {
        return minor;
    }

    public void setMinor(String minor) {
        this.minor = minor;
    }

    public String getMemo() {
        return memo;
    }

    public void setMemo(String memo) {
        this.memo = memo;
    }

    public int getRssi() {
        return rssi;
    }

    public void setRssi(int rssi) {
        this.rssi = rssi;
    }

    public boolean isExist() {
        return exist;
    }

    public void setExist(boolean exist) {
        this.exist = exist;
    }

    @Override
    public String toString() {
        return "DataSet{" +
                "KEY='" + KEY + '\'' +
                ", uuid='" + uuid + '\'' +
                ", major='" + major + '\'' +
                ", minor='" + minor + '\'' +
                ", memo='" + memo + '\'' +
                ", rssi=" + rssi +
                ", exist=" + exist +
                '}';
    }
}
