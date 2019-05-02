package com.ryoga.k17124kk.signalloger_multi.Util;

import android.os.Environment;
import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class File_ReadWriter {
    private String fileName = "BLE_InformationList.txt";


    public File_ReadWriter() {

    }

    public void writeFile_config(DataSet data) {
        Log.d("MYE", "Writing File");

        String path = Environment.getExternalStorageDirectory().getPath() + "/" + fileName;
        try {
            // FileWriterクラスのオブジェクトを生成する
            FileWriter file = new FileWriter(path);
            // PrintWriterクラスのオブジェクトを生成する
            PrintWriter pw = new PrintWriter(new BufferedWriter(file));

            File_ReadWriter file_readWriter = new File_ReadWriter();
            Gson gson = new Gson();


            List<DataSet> d = file_readWriter.readFile_config();
            d.add(data);
            //ファイルに書き込む
            pw.println(gson.toJson(d));

            Log.d("MYE", path + "::に書き込みました.");


            //ファイルを閉じる
            pw.close();
        } catch (IOException e) {
            Log.d("MYE", "write------");
            Log.d("MYE", e.getMessage());
            e.getMessage();
        }

    }

    public void rewriteFile_config(ArrayList<DataSet> data) {
        Log.d("MYE", "Writing File");

        String path = Environment.getExternalStorageDirectory().getPath() + "/" + fileName;
        try {
            // FileWriterクラスのオブジェクトを生成する
            FileWriter file = new FileWriter(path);
            // PrintWriterクラスのオブジェクトを生成する
            PrintWriter pw = new PrintWriter(new BufferedWriter(file));

            File_ReadWriter file_readWriter = new File_ReadWriter();
            Gson gson = new Gson();


            //ファイルに書き込む
            pw.println(gson.toJson(data));

            Log.d("MYE", path + "::に書き込みました.");


            //ファイルを閉じる
            pw.close();
        } catch (IOException e) {
            Log.d("MYE", "write------");
            Log.d("MYE", e.getMessage());
            e.getMessage();
        }

    }


    public ArrayList<DataSet> readFile_config() {
        String path = Environment.getExternalStorageDirectory().getPath() + "/" + fileName;
        String data = "";
        String line = "";
        List<DataSet> datas = new ArrayList<>();


        try {
            FileInputStream fis = new FileInputStream(path);
            InputStreamReader isr = new InputStreamReader(fis, "UTF-8");

            BufferedReader br = new BufferedReader(isr);

            while ((line = br.readLine()) != null) {
                data += line;
            }


            Log.d("MYE", path + " を読み込みました.");
            Log.d("MYE", data);

            br.close();

            if (!data.isEmpty()) {
                Gson gson = new Gson();
                Log.d("MYE", "-----");


                datas = gson.fromJson(data, new TypeToken<Collection<DataSet>>() {
                }.getType());

            } else {
                datas.add(new DataSet("0", "0", "0", "ああああ"));
            }


        } catch (Exception e) {
            Log.d("MYE", e.getMessage());
            datas.add(new DataSet("0", "0", "0", "あああ"));
        }


        Log.d("MYE", datas.get(0).toString());

        Log.d("MYE", "Loaded File");
        return (ArrayList<DataSet>) datas;
    }


    public void writeFile_string(String str) {
        Log.d("MYE", "Writing File");

        String path = Environment.getExternalStorageDirectory().getPath() + "/" + fileName;
        try {
            // FileWriterクラスのオブジェクトを生成する
            FileWriter file = new FileWriter(path);
            // PrintWriterクラスのオブジェクトを生成する
            PrintWriter pw = new PrintWriter(new BufferedWriter(file));

            File_ReadWriter file_readWriter = new File_ReadWriter();

            //ファイルに書き込む
            pw.println(str);

            Log.d("MYE", path + "::に書き込みました.");


            //ファイルを閉じる
            pw.close();
        } catch (IOException e) {
            Log.d("MYE", "write------");
            Log.d("MYE", e.getMessage());
            e.getMessage();
        }

    }


    public String readFile_String() {
        String path = Environment.getExternalStorageDirectory().getPath() + "/" + fileName;
        String data = "";
        String line = "";
        List<DataSet> datas = new ArrayList<>();


        try {
            FileInputStream fis = new FileInputStream(path);
            InputStreamReader isr = new InputStreamReader(fis, "UTF-8");

            BufferedReader br = new BufferedReader(isr);

            while ((line = br.readLine()) != null) {
                data += line;
            }


            Log.d("MYE", path + " を読み込みました.");
            Log.d("MYE", data);

            br.close();


        } catch (Exception e) {
            Log.d("MYE", e.getMessage());
            datas.add(new DataSet("0", "", "", ""));
        }


        Log.d("MYE", "Loaded File");

        return data;
    }


    public void writeFile_LoggingData(String fileName, String str) {
        Log.d("MYE", "Writing File");

        String path = Environment.getExternalStorageDirectory().getPath() + "/" + fileName + ".csv";
        try {
            // FileWriterクラスのオブジェクトを生成する
            FileWriter file = new FileWriter(path, true);
            // PrintWriterクラスのオブジェクトを生成する
            PrintWriter pw = new PrintWriter(new BufferedWriter(file));

            File_ReadWriter file_readWriter = new File_ReadWriter();

            //ファイルに書き込む\
            pw.println(str);

            Log.d("MYE", path + "::に書き込みました.");


            //ファイルを閉じる
            pw.close();
        } catch (IOException e) {
            Log.d("MYE", "write------");
            Log.d("MYE", e.getMessage());
            e.getMessage();
        }

    }


    public void writeFile_Label(String fileName, String str) {
        Log.d("MYE_S", "Writing File");

        String path = Environment.getExternalStorageDirectory().getPath() + "/" + fileName + ".label";
        try {
            // FileWriterクラスのオブジェクトを生成する
            FileWriter file = new FileWriter(path, true);
            // PrintWriterクラスのオブジェクトを生成する
            PrintWriter pw = new PrintWriter(new BufferedWriter(file));


            //ファイルに書き込む\
            pw.println(str);

            Log.d("MYE_S", path + "::に書き込みました.");


            //ファイルを閉じる
            pw.close();
        } catch (IOException e) {
            Log.d("MYE_S", "write------");
            Log.d("MYE_S", e.getMessage());
            e.getMessage();
        }
    }


}
