package com.ryoga.k17124kk.signalloger_multi.Activity;

import android.content.Context;
import android.os.Bundle;
import android.os.RemoteException;
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
import android.widget.TextView;

import com.ryoga.k17124kk.signalloger_multi.R;
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

public class MainActivity extends AppCompatActivity implements BeaconConsumer {

    private static final String IBEACON_FORMAT = "m:2-3=0215,i:4-19,i:20-21,i:22-23,p:24-24";


    //    private Identifier _uuid;
//    private Identifier _major;
//    private Identifier _minor;
//ListViewのAdapter クラス内クラスで定義
    private ListAdapter_main listAdapter_main;

    private ListView listView;
    private ArrayList<DataSet> datas;
    private BeaconManager beaconManager;
    private Region mRegion;
    private int rssi;

    private File_ReadWriter file_readWriter;


    private long startTime;
    private long nowTime;

    private String fileName = "BleStrengthData";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        // Keep screen on
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);


        file_readWriter = new File_ReadWriter();


        datas = file_readWriter.readFile_config();
        listAdapter_main = new ListAdapter_main(getApplicationContext(), datas);

        listView = findViewById(R.id.listview_main);
        listView.setAdapter(listAdapter_main);


        Log.d("MYE", "onCreate");


        mRegion = new Region("iBeacon", null, null, null);

        beaconManager = BeaconManager.getInstanceForApplication(this);
        beaconManager.getBeaconParsers().add(new BeaconParser().setBeaconLayout(IBEACON_FORMAT));


        Button button_start = findViewById(R.id.button_start);
        button_start.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                startTime = getFirstDate();

//
                //レンジングの開始
                try {

                    EditText editText_scan = findViewById(R.id.editText_scan);
                    beaconManager.setForegroundScanPeriod(Integer.parseInt(editText_scan.getText().toString()));
                    beaconManager.startRangingBeaconsInRegion(mRegion);
                    Log.d("MYE", "Ranging開始");
                } catch (RemoteException re) {
                    Log.d("MYE", re.getMessage());
                }

            }
        });


        Button button_stop = findViewById(R.id.button_stop);
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
                for (DataSet d : datas) {
                    Log.d("MYE", d.toString());
                    memos += "," + d.getMemo();

                }
                file_readWriter.writeFile_LoggingData(fileName, memos);


                for (DataSet dataSet : datas) {
                    dataSet.setExist(false);
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


                String rssi = "";
                nowTime = getNowDate();
                rssi += nowTime - startTime;


                for (DataSet dataSet : datas) {
                    for (Beacon beacon : beacons) {
                        if (dataSet.getUuid().equals(beacon.getId1().toString()) && dataSet.getMajor().equals(beacon.getId2().toString()) && dataSet.getMinor().equals(beacon.getId3().toString())) {
                            dataSet.setRssi(beacon.getRssi());
                            dataSet.setExist(true);
                            rssi += "," + dataSet.getRssi();
                            break;
                        } else {

                        }
                    }

                    if (!dataSet.isExist()) {
                        rssi += ",0";
                    }
                    file_readWriter.writeFile_LoggingData(dataSet.getMemo(), nowTime - startTime + "," + dataSet.getRssi());
                }


                updateList(datas);


                Log.d("MYE", rssi + "");
                file_readWriter.writeFile_LoggingData(fileName, rssi);


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
    private void updateList(ArrayList<DataSet> datasetList) {
        listAdapter_main.setDataSetList(datasetList);
        listAdapter_main.notifyDataSetChanged();
    }

    //ListViewにセットするためのAdapter
    private class ListAdapter_main extends BaseAdapter {

        Context context;
        LayoutInflater layoutInflater;
        ArrayList<DataSet> datasetList;


        public ListAdapter_main(Context context, ArrayList<DataSet> datasetList) {
            this.context = context;
            this.layoutInflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
            this.datasetList = datasetList;
        }


        public void setDataSetList(ArrayList<DataSet> datasetList) {
            this.datasetList = datasetList;
        }


        public ArrayList<DataSet> getDatasetList() {
            return this.datasetList;
        }


        @Override
        public int getCount() {
            return datasetList.size();
        }

        @Override
        public DataSet getItem(int position) {
            return datasetList.get(position);
        }


        @Override
        public long getItemId(int position) {
            return datasetList.indexOf(position);
        }

        @Override
        public View getView(final int position, View convertView, ViewGroup parent) {


            //ListViewで表示されるレイアウト
            convertView = layoutInflater.inflate(R.layout.loging_data, parent, false);

            TextView textView_memo = convertView.findViewById(R.id.textView_memo);
            TextView textView_rssi = convertView.findViewById(R.id.textView_rssi);

            textView_memo.setText(getItem(position).getMemo() + "");
            textView_rssi.setText(getItem(position).getRssi() + "");


            return convertView;
        }


    }


}
