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

public class InputNewDeviceInformationActivity extends AppCompatActivity {

    private EditText editText_uuid;
    private EditText editText_major;
    private EditText editText_minor;
    private EditText editText_memo;


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.inputdeviceinformation);

        Intent intent = getIntent();
        boolean flag = intent.getBooleanExtra("EXIST", true);

        Log.d("MYE", flag + "");


        editText_uuid = findViewById(R.id.editText_uuid);
        editText_major = findViewById(R.id.editText_major);
        editText_minor = findViewById(R.id.editText_minor);
        editText_memo = findViewById(R.id.editText_memo);


        Button button_input = findViewById(R.id.button_input);
        button_input.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                DataSet dataSet = new DataSet(editText_uuid.getText().toString(), editText_major.getText().toString(), editText_minor.getText().toString(), editText_memo.getText().toString());


                File_ReadWriter file_readWriter = new File_ReadWriter();

                ArrayList<DataSet> list = file_readWriter.readFile_config();
                list.add(dataSet);
                file_readWriter.rewriteFile_config(list);

                Log.d("MYE", "wrote");


                Intent intent = new Intent(getApplication(), ConfigActivity.class);
                startActivity(intent);


            }
        });

        Button button_back = findViewById(R.id.button_back);
        button_back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
            }
        });

    }
}
