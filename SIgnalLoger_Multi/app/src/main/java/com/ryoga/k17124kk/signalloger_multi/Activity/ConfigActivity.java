package com.ryoga.k17124kk.signalloger_multi.Activity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.Toast;

import com.ryoga.k17124kk.signalloger_multi.R;
import com.ryoga.k17124kk.signalloger_multi.Util.DataSet;
import com.ryoga.k17124kk.signalloger_multi.Util.File_ReadWriter;

import java.util.ArrayList;
import java.util.List;


public class ConfigActivity extends AppCompatActivity {

    //ListViewのAdapter クラス内クラスで定義
    private ListAdaper listAdaper;

    //書くTweetを格納するList
    private ArrayList<DataSet> datasetList;

    private ListView listView;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_setting);


        listView = findViewById(R.id.listview);
        datasetList = new ArrayList<>();


        listAdaper = new ListAdaper(getApplicationContext(), datasetList);
        listView = findViewById(R.id.listview);
        listView.setAdapter(listAdaper);


        File_ReadWriter file_readWriter = new File_ReadWriter();
        final List<DataSet> list = file_readWriter.readFile_config();
        updateList(list);


        Button button_add = findViewById(R.id.button_add);
        button_add.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                Intent intent = new Intent(getApplication(), InputNewDeviceInformationActivity.class);

                list.get(0).getUuid().isEmpty();

                Log.d("MYE", "======");
                if (list.get(0).getUuid().equals("0")) {
                    intent.putExtra("EXIST", false);
                    Log.d("MYE", "intent");
                } else {
                    intent.putExtra("EXIST", true);
                }
                Log.d("MYE", "intent");
                startActivity(intent);


            }
        });

        Button button_start = findViewById(R.id.button_start);
        button_start.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(getApplication(), MainActivity.class);
                startActivity(intent);
            }
        });

        button_start.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View view) {
                Intent intent = new Intent(getApplication(), LogginwithFilterActivity.class);
                startActivity(intent);
                return true;
            }
        });


    }

    @Override
    protected void onStart() {
        super.onStart();

        Toast.makeText(this, "スタート長押しでローパスフィルター有りのログ取り", Toast.LENGTH_LONG).show();
    }

    //ListをListAdapterにセットし更新を通知する
    private void updateList(List<DataSet> datasetList) {
        listAdaper.setDataSetList(datasetList);
        listAdaper.notifyDataSetChanged();
    }

    //ListViewにセットするためのAdapter
    private class ListAdaper extends BaseAdapter {

        Context context;
        LayoutInflater layoutInflater;
        List<DataSet> datasetList;


        public ListAdaper(Context context, List<DataSet> datasetList) {
            this.context = context;
            this.layoutInflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
            this.datasetList = datasetList;
        }


        public void setDataSetList(List<DataSet> datasetList) {
            this.datasetList = datasetList;
        }


        public List<DataSet> getDatasetList() {
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
            convertView = layoutInflater.inflate(R.layout.daviceinfo_list, parent, false);

            EditText editText_uuid = convertView.findViewById(R.id.editText_uuid);
            EditText editText_major = convertView.findViewById(R.id.editText_major);
            EditText editText_minor = convertView.findViewById(R.id.editText_minor);
            EditText editText_memo = convertView.findViewById(R.id.editText_memo);

            DataSet dataSet = getItem(position);

            editText_uuid.setText(dataSet.getUuid());
            editText_major.setText(dataSet.getMajor());
            editText_minor.setText(dataSet.getMinor());
            editText_memo.setText(dataSet.getMemo());


            Button button_delete = convertView.findViewById(R.id.button_delete);
            button_delete.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    Log.d("MYE", "delete:" + position);
                    File_ReadWriter file_readWriter = new File_ReadWriter();

                    ArrayList<DataSet> list = file_readWriter.readFile_config();
                    list.remove(position);

                    file_readWriter.rewriteFile_config(list);

                    updateList(file_readWriter.readFile_config());

                }
            });


            Button button_edit = convertView.findViewById(R.id.button_edit);
            button_edit.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    Log.d("MYE", "edit:" + position);

                    Intent intent_edit = new Intent(getApplication(), EditConfigActivity.class);
                    intent_edit.putExtra("DATA", getItem(position));
                    intent_edit.putExtra("POSITION", position);
                    startActivity(intent_edit);
                }
            });

            return convertView;
        }


    }


}
