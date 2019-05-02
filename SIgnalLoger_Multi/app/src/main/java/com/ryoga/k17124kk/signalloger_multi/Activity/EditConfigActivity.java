package com.ryoga.k17124kk.signalloger_multi.Activity;

import android.content.Intent;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import com.ryoga.k17124kk.signalloger_multi.R;
import com.ryoga.k17124kk.signalloger_multi.Util.DataSet;
import com.ryoga.k17124kk.signalloger_multi.Util.File_ReadWriter;

import java.util.ArrayList;

public class EditConfigActivity extends AppCompatActivity {

    private EditText editText_uuid;
    private EditText editText_major;
    private EditText editText_minor;
    private EditText editText_memo;

    private DataSet dataSet;
    private int position = 0;


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_editconfig);


        Log.d("MYE", "EditConfig");

        Intent intent = getIntent();
        dataSet = (DataSet) intent.getSerializableExtra("DATA");
        position = intent.getIntExtra("POSITION", 0);

        Log.d("MYE", "position:" + position + ",,," + dataSet.toString());


        editText_uuid = findViewById(R.id.editText_uuid_editconfig);
        editText_major = findViewById(R.id.editText_major_editconfig);
        editText_minor = findViewById(R.id.editText_minor_editconfig);
        editText_memo = findViewById(R.id.editText_memo_editconfig);

        editText_uuid.setText(dataSet.getUuid());
        editText_major.setText(dataSet.getMajor());
        editText_minor.setText(dataSet.getMinor());
        editText_memo.setText(dataSet.getMemo());


        Button button_back = findViewById(R.id.button_back_config);
        button_back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
            }
        });


        Button button_edit = findViewById(R.id.button_edit_config);
        button_edit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                File_ReadWriter file_readWriter = new File_ReadWriter();
                ArrayList<DataSet> datalist = file_readWriter.readFile_config();

                datalist.get(position).setUuid(editText_uuid.getText().toString());
                datalist.get(position).setMajor(editText_major.getText().toString());
                datalist.get(position).setMinor(editText_minor.getText().toString());
                datalist.get(position).setMemo(editText_memo.getText().toString());

                file_readWriter.rewriteFile_config(datalist);

                Intent intent = new Intent(getApplication(), ConfigActivity.class);
                startActivity(intent);

            }
        });


    }


}
