import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_widths_robust(raw_file, group_file, output_file):
    print("1. 正在读取数据...")
    df = pd.read_csv(raw_file)
    groups_df = pd.read_csv(group_file)

    # --- 步骤A: 提取原始波形 ---
    print("2. 正在提取原始波形数据...")
    is_start = (df['Channel 1'] == 0) & (df['Channel 2'] == 0)
    is_end = (df['Channel 1'] == 0) & (df['Channel 2'] == 61680)
    start_indices = df.index[is_start].tolist()
    end_indices = df.index[is_end].tolist()

    all_segments = []
    for s_idx in start_indices:
        possible_ends = [e for e in end_indices if e > s_idx]
        if possible_ends:
            e_idx = possible_ends[0]
            if e_idx - s_idx > 100:
                seg_data = df.iloc[s_idx + 1: e_idx].reset_index(drop=True)
                all_segments.append(seg_data)

    print(f"   提取到 {len(all_segments)} 幅图像的波形。")


    # --- 生成图像函数---
    def plot_width_details(csv_file, output_img):
        print(f"正在读取文件: {csv_file} ...")
        df = pd.read_csv(csv_file)

        # 创建一个大画布，包含 2 个子图 (上下排列)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # --- 画 Channel 1 ---
        ax1.set_title("Channel 1 波形半周期宽度 (正常值约 20-30)", fontsize=14, fontweight='bold')
        ax1.plot(df['Position_mm'], df['Ch1_Width_Mean'], '.-', color='#1f77b4', label='Ch1 Width', linewidth=1)

        # 标记异常点 (>100)
        abnormal_1 = df[df['Ch1_Width_Mean'] > 100]
        ax1.plot(abnormal_1['Position_mm'], abnormal_1['Ch1_Width_Mean'], 'rx', markersize=8, label='异常突变 (>100)')

        # 在图上标出具体的数值 (只标前几个，避免太乱)
        for i, row in abnormal_1.head(5).iterrows():
            ax1.annotate(f"{row['Ch1_Width_Mean']:.0f}",
            (row['Position_mm'], row['Ch1_Width_Mean']),
            textcoords="offset points", xytext=(0, 10), ha='center', color='red', fontsize=9)

        ax1.set_ylabel("宽度 (数据点)", fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # --- 画 Channel 2 ---
        ax2.set_title("Channel 2 波形半周期宽度 (大量失效数据 ≈ 850)", fontsize=14, fontweight='bold')
        ax2.plot(df['Position_mm'], df['Ch2_Width_Mean'], '.-', color='#ff7f0e', label='Ch2 Width', linewidth=1)

        # 标记异常点
        abnormal_2 = df[df['Ch2_Width_Mean'] > 100]
        ax2.plot(abnormal_2['Position_mm'], abnormal_2['Ch2_Width_Mean'], 'rx', markersize=6, alpha=0.5)

        # 标出一个典型的“平台值”
        if not abnormal_2.empty:
            peak_val = abnormal_2['Ch2_Width_Mean'].median()
            ax2.text(df['Position_mm'].min(), peak_val + 50,
            f"注意：此处大量数据处于 {peak_val:.0f} 附近 (信号丢失/饱和)",
            color='red', fontsize=11, fontweight='bold')

        ax2.set_xlabel("物理位置 (mm)", fontsize=12)
        ax2.set_ylabel("宽度 (数据点)", fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_img, dpi=150)
        print(f"图表已生成！请查看: {output_img}")
        plt.show()


    # =========================================================
    # ⭐ 新增：去噪核心函数 (Outlier Removal)
    # =========================================================
    def get_robust_mean(data_list, tolerance=0.5):
        """
        计算健壮的平均值，自动剔除异常大或异常小的数据
        :param data_list: 数据列表
        :param tolerance: 容忍度 (0.5 表示只允许在中位数上下 50% 浮动)
        """
        data = np.array(data_list)
        data = data[~np.isnan(data)]  # 去除空值

        if len(data) == 0: return np.nan
        if len(data) <= 2: return np.mean(data)  # 数据太少，没法判断谁是噪声，直接平均

        # 1. 计算中位数 (Median) - 它是最抗干扰的统计量
        median_val = np.median(data)

        # 2. 设定合理范围 (例如: 0.5倍 ~ 1.5倍中位数)
        # 这样像 2000 这种巨大的噪声（远超 1.5倍）就会被过滤
        lower_bound = median_val * (1 - tolerance)
        upper_bound = median_val * (1 + tolerance)

        # 3. 筛选有效数据
        valid_data = data[(data >= lower_bound) & (data <= upper_bound)]

        if len(valid_data) == 0: return np.nan

        # 4. 对剩下的干净数据求平均
        return np.mean(valid_data)

    # --- 步骤B: 单列波形分析 ---
    def analyze_channel_width(y_data):
        y = y_data.values
        window = 10
        y_s = np.convolve(y, np.ones(window) / window, mode='same')

        # 局部中位线
        mid = (np.max(y_s) + np.min(y_s)) / 2

        # 寻找交叉点
        crossings = []
        diff = y_s - mid
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        for i in sign_changes:
            x1, y1 = i, y_s[i]
            x2, y2 = i + 1, y_s[i + 1]
            if y2 != y1:
                frac = (mid - y1) / (y2 - y1)
                x_exact = x1 + frac
                crossings.append(x_exact)

        crossings = np.array(crossings)
        if len(crossings) < 2: return np.nan

        # 计算所有相邻宽度
        widths = np.diff(crossings)

        # ⭐ 在这里调用去噪函数：剔除单个波形里的畸变
        # 容忍度设为 0.8，意味着允许宽度在 0.2x ~ 1.8x 之间波动，
        # 那些“巨大矩形”带来的超宽数据会被无情抛弃。
        return get_robust_mean(widths, tolerance=0.8)

    # --- 步骤C: 按组计算 ---
    print("3. 正在按组计算 (已启用双重去噪)...")
    results = []
    current_seg_idx = 0

    for i, row in groups_df.iterrows():
        pos = row['Position_mm']
        count = int(row['Image_Count'])

        ch1_widths = []
        ch2_widths = []

        if current_seg_idx + count <= len(all_segments):
            for k in range(count):
                seg = all_segments[current_seg_idx + k]

                w1 = analyze_channel_width(seg['Channel 1'])
                w2 = analyze_channel_width(seg['Channel 2'])

                if not np.isnan(w1): ch1_widths.append(w1)
                if not np.isnan(w2): ch2_widths.append(w2)

            current_seg_idx += count
        else:
            break

        # ⭐ 在这里再次调用去噪函数：剔除整幅图都是噪声的情况
        # 这里的容忍度设小一点(0.5)，因为同一组内的图像应该非常一致
        avg_w1 = get_robust_mean(ch1_widths, tolerance=0.5)
        avg_w2 = get_robust_mean(ch2_widths, tolerance=0.5)

        results.append({
            'Position_mm': pos,
            'Ch1_Width_Mean': avg_w1,
            'Ch2_Width_Mean': avg_w2,
            'Raw_Images_Ch1': ch1_widths,  # 记录原始数据方便调试
            'Raw_Images_Ch2': ch2_widths
        })

    # --- 步骤D: 保存 ---
    # 这里需要做一点处理，因为 DataFrame 不能直接存列表，我们只存 Mean
    output_data = [{k: v for k, v in r.items() if 'Raw' not in k} for r in results]
    result_df = pd.DataFrame(output_data)

    result_df.to_csv(output_file, index=False)
    print(f"4. 计算完成！已自动剔除噪声干扰。结果保存至: {output_file}")

    # 简单的文本报告：看看剔除了多少数据
    print("\n--- 噪声处理报告 ---")
    for i, r in enumerate(results[:10]):  # 只打印前10个看看
        raw = r['Raw_Images_Ch1']
        clean = r['Ch1_Width_Mean']
        print(f"Pos {r['Position_mm']}: 原始数据 {np.round(raw, 1)} -> 去噪后 {clean:.1f}")
    plot_width_details(r'D:\桌面\OPC\position_widths_robust.csv', r'D:\桌面\OPC\width_detail_view.png')

# 运行
calculate_widths_robust(
    r'D:\桌面\OPC\2.csv',
    r'D:\桌面\OPC\processed_positions_optimized.csv',
    r'D:\桌面\OPC\position_widths_robust.csv'
)
