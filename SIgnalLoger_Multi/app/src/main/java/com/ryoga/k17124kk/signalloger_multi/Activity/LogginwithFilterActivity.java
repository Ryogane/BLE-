package com.ryoga.k17124kk.signalloger_multi.Activity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.RemoteException;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.Spinner;
import android.widget.TextView;

import com.ryoga.k17124kk.signalloger_multi.R;
import com.ryoga.k17124kk.signalloger_multi.Util.DataController;
import com.ryoga.k17124kk.signalloger_multi.Util.DataSet;
import com.ryoga.k17124kk.signalloger_multi.Util.File_ReadWriter;

import org.altbeacon.beacon.Beacon;
import org.altbeacon.beacon.BeaconConsumer;
import org.altbeacon.beacon.BeaconManager;
import org.altbeacon.beacon.BeaconParser;
import org.altbeacon.beacon.MonitorNotifier;
import org.altbeacon.beacon.RangeNotifier;
import org.altbeacon.beacon.Region;

import java.util.ArrayList;
import java.util.Collection;

public class LogginwithFilterActivity extends AppCompatActivity implements BeaconConsumer {


    private static final String IBEACON_FORMAT = "m:2-3=0215,i:4-19,i:20-21,i:22-23,p:24-24";


    //    private Identifier _uuid;
//    private Identifier _major;
//    private Identifier _minor;
//ListViewのAdapter クラス内クラスで定義
    private ListAdapter_main listAdapter_main;

    private ListView listView;
    private ArrayList<DataController> dataControllers;
    private BeaconManager beaconManager;
    private Region mRegion;
    private int rssi;

    private File_ReadWriter file_readWriter;


    private long startTime;
    private long nowTime;

    private String fileName = "BleStrengthData";
    private String fileName_Filtered = "BleStrengthData_Lowpath";

    private String filter_Mode;
    private final String FILTER_MODE[] = {"中央値", "移動平均"};


    private int sample = 0;


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_filtering);

        Intent intent = getIntent();


        Log.d("MYE", "FilterActivity");

        // Keep screen on
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);


        file_readWriter = new File_ReadWriter();


        dataControllers = new ArrayList<>();
        for (DataSet d : file_readWriter.readFile_config()) {
            dataControllers.add(new DataController(d, "中央値"));
        }


        listAdapter_main = new ListAdapter_main(getApplicationContext(), dataControllers);

        listView = findViewById(R.id.listview_main_F);
        listView.setAdapter(listAdapter_main);


        Log.d("MYE", "onCreate");


        mRegion = new Region("iBeacon", null, null, null);

        beaconManager = BeaconManager.getInstanceForApplication(this);
        beaconManager.getBeaconParsers().add(new BeaconParser().setBeaconLayout(IBEACON_FORMAT));


        Button button_start = findViewById(R.id.button_start_F);
        button_start.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                startTime = getFirstDate();

                EditText editText_sample = findViewById(R.id.editText_sample_F);
                sample = Integer.valueOf(editText_sample.getText().toString());


                Spinner spinner_FilterMode = findViewById(R.id.spinner_filterMode);
                filter_Mode = (String) spinner_FilterMode.getSelectedItem();
                for (DataController dc : dataControllers) {
                    dc.setFilterMode(filter_Mode);
                }

                Log.d("MYE_F_M", filter_Mode + "------");
//
                //レンジングの開始
                try {

                    for (DataController dc : dataControllers) {
                        dc.setFILTER_SIZE(sample);
                    }
                    EditText editText_scan = findViewById(R.id.editText_scan_F);
                    beaconManager.setForegroundScanPeriod(Integer.parseInt(editText_scan.getText().toString()));
                    beaconManager.startRangingBeaconsInRegion(mRegion);
                    Log.d("MYE_F", "Ranging開始  ControlloerSize : " + dataControllers.size());
                } catch (RemoteException re) {
                    Log.d("MYE", re.getMessage());
                }

            }
        });


        Button button_stop = findViewById(R.id.button_stop_F);
        button_stop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                try {
                    //レンジングの停止
                    beaconManager.stopRangingBeaconsInRegion(mRegion);
                    Log.d("MYE", "Ranging停止");
                } catch (RemoteException re) {

                }

                String memos = "ms";
                for (DataController d : dataControllers) {
                    Log.d("MYE", d.toString());
                    memos += "," + d.getDataSet().getMemo();

                }
                file_readWriter.writeFile_LoggingData(fileName, memos);
                file_readWriter.writeFile_LoggingData(fileName_Filtered, memos);


                for (DataController DataController : dataControllers) {
                    DataController.getDataSet().setExist(false);
                    //DataCOntroller内の配列リセット
                    DataController.resetDatasetArrayList();
                }
            }
        });

    }

    @Override
    protected void onResume() {
        super.onResume();
        // サービスの開始
        beaconManager.bind(this);
    }

    @Override
    protected void onPause() {
        super.onPause();
        // サービスの停止
        beaconManager.unbind(this);
    }


    @Override
    public void onBeaconServiceConnect() {


        beaconManager.addMonitorNotifier(new MonitorNotifier() {
            @Override
            public void didEnterRegion(Region region) {
                // 領域侵入

                Log.d("MYE", "Enter");
            }

            @Override
            public void didExitRegion(Region region) {
                // 領域退出

            }

            @Override
            public void didDetermineStateForRegion(int i, Region region) {
                // 領域に対する状態が変化
            }


        });
        beaconManager.addRangeNotifier(new RangeNotifier() {
            @Override
            public void didRangeBeaconsInRegion(Collection<Beacon> beacons, Region region) {

                boolean flag = false;

                String rssi = "";
                String rssi_Filtered = "";

                nowTime = getNowDate();

                String time = Long.toString(nowTime - startTime);


                rssi += time;
                rssi_Filtered += time;


                for (DataController dataController : dataControllers) {
                    Log.d("MYE_F", dataController.toString() + "======================");
                    for (Beacon beacon : beacons) {
                        if (dataController.getDataSet().getUuid().equals(beacon.getId1().toString()) && dataController.getDataSet().getMajor().equals(beacon.getId2().toString()) && dataController.getDataSet().getMinor().equals(beacon.getId3().toString())) {

                            dataController.setDataSetRssi(time, beacon.getRssi());
                            dataController.getDataSet().setExist(true);

                            rssi += "," + dataController.getDataSet().getRssi();

                            rssi_Filtered += "," + dataController.getFilterd_rssi();

                            flag = true;

                            break;
                        } else {

                        }
                    }

                    if (!dataController.getDataSet().isExist()) {
                        rssi += ",0";
                        rssi_Filtered += ",0";
                    }

                    if (flag) {
                        file_readWriter.writeFile_LoggingData(dataController.getDataSet().getMemo(), time + "," + dataController.getDataSet().getRssi());

                        if (filter_Mode.equals(FILTER_MODE[0])) {
                            file_readWriter.writeFile_LoggingData(dataController.getDataSet().getMemo() + "_Lowpath_Median", time + "," + dataController.getFilterd_rssi());
                        } else if (filter_Mode.equals(FILTER_MODE[1])) {
                            file_readWriter.writeFile_LoggingData(dataController.getDataSet().getMemo() + "_Lowpath_MoveAverage", time + "," + dataController.getFilterd_rssi());

                        }

                    }
                }


                updateList(dataControllers);


                Log.d("MYE", rssi + "");

                if (flag) {
                    file_readWriter.writeFile_LoggingData(fileName, rssi);
                    if (filter_Mode.equals(FILTER_MODE[0])) {
                        file_readWriter.writeFile_LoggingData(fileName_Filtered + "_Median", rssi_Filtered);
                    } else if (filter_Mode.equals(FILTER_MODE[1])) {
                        file_readWriter.writeFile_LoggingData(fileName_Filtered + "_MoveAverage", rssi_Filtered);

                    }
                }


            }
        });
    }


    public static long getFirstDate() {
        return System.currentTimeMillis();
    }

    public static long getNowDate() {
        return System.currentTimeMillis();
    }


    //ListをListAdapterにセットし更新を通知する
    private void updateList(ArrayList<DataController> DataControllerList) {
        listAdapter_main.setDataSetList(DataControllerList);
        listAdapter_main.notifyDataSetChanged();
    }

    //ListViewにセットするためのAdapter
    private class ListAdapter_main extends BaseAdapter {

        Context context;
        LayoutInflater layoutInflater;
        ArrayList<DataController> dataControllerList;


        public ListAdapter_main(Context context, ArrayList<DataController> DataControllerList) {
            this.context = context;
            this.layoutInflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
            this.dataControllerList = DataControllerList;
        }


        public void setDataSetList(ArrayList<DataController> DataControllerList) {
            this.dataControllerList = DataControllerList;
        }


        public ArrayList<DataController> getDataControllerList() {
            return this.dataControllerList;
        }


        @Override
        public int getCount() {
            return dataControllerList.size();
        }

        @Override
        public DataController getItem(int position) {
            return dataControllerList.get(position);
        }


        @Override
        public long getItemId(int position) {
            return dataControllerList.indexOf(position);
        }

        @Override
        public View getView(final int position, View convertView, ViewGroup parent) {


            //ListViewで表示されるレイアウト
            convertView = layoutInflater.inflate(R.layout.loging_data, parent, false);

            TextView textView_memo = convertView.findViewById(R.id.textView_memo);
            TextView textView_rssi = convertView.findViewById(R.id.textView_rssi);
            TextView textView_stability = convertView.findViewById(R.id.textView_Stability);
            TextView textView_negaposi = convertView.findViewById(R.id.textView_state);

            textView_memo.setText(getItem(position).getDataSet().getMemo() + "");
            textView_rssi.setText(getItem(position).getDataSet().getRssi() + "");

            if (getItem(position).getStabilityOfSensingSignal().isStability()) {
                textView_stability.setText("安定");
            } else {
                textView_stability.setText("不安定");
            }

            //一つ前の状態を見て反転し現在の状態にする
            if (getItem(position).getStabilityOfSensingSignal().getCurrentStabilityData().getNegaPosi() == 0) {
                textView_negaposi.setText("ポジティブ");
            } else {
                textView_negaposi.setText("ネガティブ");
            }


            return convertView;
        }


    }


}
