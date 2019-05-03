# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics


# csvファイルから値を読み込む
def read_csv_value(file_path):
    """
    :param file_path:   # str型：開くcsvファイルのファイルパス
    :return: time, rssi # ndarray型：時間, ndarray型：rssi値
    """
    # csvファイルからの読み込み
    csv = pd.read_csv(file_path, names=('Time', 'RSSI'))
    # 読み込んだ時間文字列を時間形式に変換
    # time = pd.to_datetime(csv.Time, format='%H:%M:%S')
    # 読み込んだTimeの値をnumpyの形式に変換
    time = csv.Time.values / 1000
    # 読み込んだrssiの値をnumpyの形式に変換
    rssi = csv.RSSI.values
    # 時間が0秒から始まるように調整
    for i in range(len(time) - 1):
        time[i] = time[i + 1] - time[0]
    return time, rssi


# 値を正規化する関数
def min_max_normalization(x):
    """
    :param x:   # ndarray型：BELビーコンの信号強度データ
    :return:    # 0~1の間の値に正規化したデータ
    """
    # リスト内から最小値を取得
    x_min = x.min()
    # リスト内から最大値を取得n
    x_max = x.max()
    # 正規化する
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


# 移動平均をかける
def move_ave(rssi, num):
    """
    :param rssi:    # ndarray型：BELビーコンの信号強度データ
    :param num:     # int型：フィルタサンプル数
    :return:        # ndarray型：移動平均をかけたデータ
    """
    b = np.ones(num) / num
    result = np.convolve(rssi, b, mode='valid')  # 移動平均をかける
    return result


# フィルターをかけた後のデータをファイルに書き込む
def write_csv(file_path, time, rssi):
    """
    :param file_path:   # str型：保存するcsvファイルの名前
    :param time:        # ndarray型：経過時間(BLEビーコンの信号強度グラフのx軸データ)
    :param time:        # ndarray型：BELビーコンの信号強度データ
    """
    with open(file_path, mode='w') as f:
        for i in range(len(time)-1):
            f.write(str(time[i]*1000) + ',' + str(rssi[i]) + '\n')


# 安定センシング区間を探す関数
def find_stable_sensing_zone(t, rssi, under, upper):
    """
    :param t:                             # list型：経過時間(BLEビーコンの信号強度グラフのx軸データ)
    :param rssi:                          # ndarray型：BELビーコンの信号強度データ
    :param under:                         # int型：安定センシング区間だと判断する閾値の下限
    :param upper:                         # int型：安定センシング区間だと判断する閾値の上限
    :return: stable_time, stable_rssi     # list型：安定センシング区間の時間データ, list型：安定センシング区間の信号強度データ
    """
    tmp = []
    tmp_rssi = []
    stable_rssi = []
    stable_time = []

    # 安定センシング区間の範囲の設定
    num_min = rssi[0] - under
    num_max = rssi[0] + upper

    for i in range(len(rssi)-1):
        # もし設定値を超えていなかったらその時間をリストに追加する
        if num_min < rssi[i+1] < num_max:
            tmp.append(t[i])
            tmp_rssi.append(rssi[i])

            # もし今見ているデータが最後だったら一時リスト(tmp)からstable_timeに移す
            if i == len(rssi)-1:
                stable_time.append(tmp)
                stable_rssi.append(tmp_rssi)
        else:
            # もし一時リストが空じゃなかったらstable_timeに移して初期化する
            if len(tmp) > 0:
                stable_time.append(tmp)
                stable_rssi.append(tmp_rssi)
                tmp = []
                tmp_rssi = []

            # もし今見ているデータが最後だったら一時リスト(tmp)からstable_timeに移す
            elif i == len(rssi) - 1:
                stable_time.append(tmp)
                stable_rssi.append(tmp_rssi)

            # もし設定値を超えていたら設定値を再設定
            num_min = rssi[i] - under
            num_max = rssi[i] + upper

    # もし一時リスト(tmp)にデータが残っていたらstable_timeに移す
    if len(tmp) > 0:
        stable_time.append(tmp)
        stable_rssi.append(tmp_rssi)

    return stable_time, stable_rssi


# 安定センシング区間をプロットする関数
def plot_stable_zone(t, rssi, stable_time, interval, save_name):
    """
    :param t:               # list型：経過時間(BLEビーコンの信号強度グラフのx軸データ)
    :param rssi:            # ndarray型：BELビーコンの信号強度データ
    :param stable_time:     # list型：安定センシング区間の時間データ
    :param interval:        # int型：安定センシング区間を判断する時間間隔
    :param save_name:       # str型：保存グラフ名
    """
    # グラフの色
    # color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    # 結果を元に推定箇所に色を塗る
    count = 0
    for j in range(len(stable_time)):
        if stable_time[j][-1] - stable_time[j][0] > interval:
            plt.fill_between([stable_time[j][0], stable_time[j][-1]], [0, 0], [1, 1], facecolor='r', alpha=0.2)
            count += 1

        if count == 2:
            count = 0

    plt.title("安定センシング区間", fontsize=18)  # グラフタイトル
    plt.xlabel('time(sec)', fontsize=18)
    plt.ylabel('signal', fontsize=18)
    # plt.xticks(np.arange(0, 120, 10))  # グラフのメモリ間隔の設定
    #ax.xaxis.grid(linestyle='--', lw=1, color='black')
    plt.plot(t, rssi, color="blue")
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_name, format='png', dpi=300)


# 安定センシング区間のネガポジ判定を行う関数
def up_down_judge(stable_rssi):
    """
    :param stable_rssi:     # list型：安定センシング区間の信号強度データ
    :return:                # list型：ネガポジの判定結果
    """

    rssi_ave = [sum(i) / len(i) for i in stable_rssi]   # 各安定センシング区間のrssi値の平均値を算出
    up_down = []

    # 区間ごとの中央値を使う場合
    rssi_median = statistics.median([statistics.median(i) for i in stable_rssi])+0.1

    # 各安定センシング区間のrssi値の平均値の中央値を使う場合
    #rssi_median = statistics.median(rssi_ave)

    for value in rssi_ave:
        if value >= rssi_median:
            up_down.append(0)
        else:
            up_down.append(1)
    return up_down


# 安定センシング区間のネガポジをプロットする関数
def plot_negaposi(t, rssi, stable_time, interval, negaposi, save_name):
    """
    :param t:               # list型：経過時間(BLEビーコンの信号強度グラフのx軸データ)
    :param rssi:            # ndarray型：BELビーコンの信号強度データ
    :param stable_time:     # list型：安定センシング区間の時間データ
    :param negaposi:        # list型：ネガポジ判定結果のリスト
    :param interval:        # int型：安定センシング区間を判断する時間間隔
    :param save_name:       # str型：保存グラフ名
    """

    interval2 = interval+2

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    # ラベル用
    plt.fill_between([0,0], [0,0], [1,1], facecolor='r', alpha=0.2, label='ポジティブ')
    plt.fill_between([0,0], [0,0], [1,1], facecolor='g', alpha=0.2, label='ネガティブ')

    # 結果を元に推定箇所に色を塗る
    for i in range(len(stable_time)):
        if stable_time[i][-1] - stable_time[i][0] > interval and negaposi[i] == 0:
            #if interval2 > stable_time[i][-1] - stable_time[i][0] > interval and negaposi[i] == 0:
            plt.fill_between([stable_time[i][0], stable_time[i][-1]], [0,0], [1,1], facecolor='r', alpha=0.2)
        elif stable_time[i][-1] - stable_time[i][0] > interval and negaposi[i] == 1:
            #elif interval2 > stable_time[i][-1] - stable_time[i][0] > interval and negaposi[i] == 1:
            plt.fill_between([stable_time[i][0], stable_time[i][-1]], [0,0], [1,1], facecolor='g', alpha=0.2)

    plt.title('座椅子の状態推定結果', fontsize=18)  # グラフタイトル
    plt.xlabel('time(sec)', fontsize=18)
    plt.ylabel('signal', fontsize=18)
    #plt.xticks(np.arange(0, 120, 10))  # グラフのメモリ間隔の設定
    #ax.xaxis.grid(linestyle='--', lw=1, color='black')
    plt.plot(t, rssi, color="blue")
    plt.legend(loc='lower right', facecolor='w')    # lower⇆upper, left⇆ center ⇆right で場所を指定.
    plt.tight_layout()
    plt.savefig(save_name, format='png', dpi=300)


# オリジナルデータのグラフを作る関数
def create_original_data_graph(relative_time, rssi, save_name):
    """
    :param relative_time:  # list型：経過時間のデータ
    :param rssi:           # list型：rssi強度データ
    :param save_name:      # str型：保存ファイル名
    """
    x = np.array(relative_time)
    y = np.array(rssi)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    plt.plot(x, y, color="blue")                              # グラフ線の設定と描画    marker="o"
    plt.title("ローパスフィルタ適用前", fontsize=18)            # グラフタイトル
    plt.xlabel('time(sec)', fontsize=18)        # x軸のラベル
    plt.ylabel('signal', fontsize=18)           # x軸のラベル
    plt.tight_layout()
    plt.savefig(save_name, format='png', dpi=300)


def create_lowpath_data_graph(relative_time, rssi, save_name):
    """
    :param relative_time:  # list型：経過時間のデータ
    :param rssi:           # list型：rssi強度データ
    :param save_name:      # str型：保存ファイル名
    """
    x = np.array(relative_time)
    y = np.array(rssi)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    plt.plot(x, y, color="blue")                             # グラフ線の設定と描画
    plt.title("ローパスフィルタ適用後", fontsize=18)      # グラフタイトル
    plt.xlabel("time(sec)", fontsize=18)                      # x軸のラベル
    plt.ylabel("signal", fontsize=18)                        # y軸のラベル
    #plt.xticks(np.arange(0, 120, 10))  # グラフのメモリ間隔の設定
    #ax.xaxis.grid(linestyle='--', lw=1, color='black')
    plt.tight_layout()
    plt.savefig(save_name, format='png', dpi=300)


def main():

    # csvファイルを読み込む
    time, rssi_raw = read_csv_value('./背中4.csv')

    # データの正規化(0~1)を行う
    rssi = min_max_normalization(rssi_raw)

    # 各種パラメータ
    filter_size = 10           # ローパスフィルタのフィルターサイズ
    sz_interval = 5          # 安定センシング区間を判断する時間間隔
    sz_under = 0.235            # 安定センシング区間だと判断する閾値の下限
    sz_upper = 0.235            # 安定センシング区間だと判断する閾値の上限

    # 保存グラフ名
    raw_graph_name = 'raw_rssi.png'
    move_ave_graph_name = 'move_ave_rssi.png'
    StableZone_graph_name = 'stable_zone.png'
    negaposi_graph_name = 'stable_zone_negaposi.png'

    # オリジナルデータのグラフを作成
    print('オリジナルデータのグラフを書き出します', end='')
    create_original_data_graph(time, rssi, raw_graph_name)
    print(' 　 ・・・OK!')

    # 移動平均を用いたローパスフィルタをかける
    convo = move_ave(rssi, filter_size)

    # mode=’valid’(先頭と末尾の平均を取れない部分を省くモード)で計算したため
    # 結果は時系列よりも前1つ、後ろ1つ分少ない要素数で出力されるので時間の要素数もそれに合わせる
    for i in range(filter_size // 2):
        time = np.delete(time, 0)
        time = np.delete(time, len(time)-1)
    if filter_size % 2 == 0:
        convo = np.delete(convo, len(convo)-1)

    # 時間が0秒から始まるように調整する
    for i in range(len(time) - 1):
        time[i] = time[i + 1] - time[0]

    # ローパスフィルタをかけたデータを書き出す
    # write_csv('./move_ave_rssi.csv', time, rssi)

    # ローパスフィルタをかけたグラフを保存
    create_lowpath_data_graph(time, convo, move_ave_graph_name)

    # 安定センシング区間か判別する
    sys.stdout.write('安定センシング区間を探します')
    sys.stdout.flush()
    stable_time, stable_rssi = find_stable_sensing_zone(time, convo, sz_under, sz_upper)

    # 安定センシング区間をプロットする
    plot_stable_zone(time, convo, stable_time, sz_interval, StableZone_graph_name)
    print(' 　　　　　　 ・・・OK!')

    # 安定センシング区間のネガポジを判定する
    sys.stdout.write('安定センシング区間のネガポジを判定します')
    sys.stdout.flush()
    up_down = up_down_judge(stable_rssi)
    plot_negaposi(time, convo, stable_time, sz_interval, up_down, negaposi_graph_name)
    print('  ・・・OK!')
    print('\nFinish!!')
    sys.stdout.flush()


if __name__ == '__main__':
    main()