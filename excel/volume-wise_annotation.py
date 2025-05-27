import os
import pandas as pd
from datetime import datetime

# 1. 讀取整本 Excel（所有 sheet），並取出 'annotations' 這張表
# file_path = r"D:\cleaning_GUI_annotated_Data\new_data_tab_data.xlsx"
file_path = r"D:\cleaning_GUI_annotated_Data\tab_data_annotated_pats.xlsx"
dfs_dict  = pd.read_excel(file_path, sheet_name=None)
df        = dfs_dict['annotations']

# 2. 指定 OCT 影像根資料夾
# base_dir = r"D:\cleaning_GUI_annotated_Data\New_Data"
base_dir = r"D:\cleaning_GUI_annotated_Data\Cirrus_OCT_Imaging_Data"

# 3. 明列要當作 stage 依據的欄位
stage_cols = ['Early AMD','Int AMD','GA','Wet','Scar','Not AMD']

# 4. 將這些日期欄位轉成 pandas datetime，並確保有 baseline_stage 欄位
for c in stage_cols:
    df[c] = pd.to_datetime(df[c], errors='coerce')
# 假設 annotations sheet 有一欄 'baseline_stage'，直接保留即可

# 5. 定義：給定 annotation row + visit_dt，回傳該 scan 應該的 stage
def get_stage_for_date(row, visit_dt):
    # 收集所有事件 (dt, label)
    raw = [(row[c], c) for c in stage_cols if pd.notnull(row[c])]
    # 按日期分組：date -> set(labels)
    by_date = {}
    for dt, lbl in raw:
        by_date.setdefault(dt, set()).add(lbl)
    # 只保留 <= visit_dt 的，並依規則解決同日衝突
    events = []
    for dt, labels in by_date.items():
        if dt <= visit_dt:
            if labels == {'Wet', 'Scar'}:
                # 若同天只有 Wet & Scar 衝突，整個 stage 視為 Scar
                resolved = 'Scar'
            elif len(labels) == 1:
                resolved = next(iter(labels))
            else:
                # 其他多重衝突，隨機取一（或自訂邏輯）
                resolved = sorted(labels)[0]
            events.append((dt, resolved))
    if not events:
        return None
    # 最新事件決定當前 stage
    latest_dt = max(dt for dt, _ in events)
    for dt, lbl in events:
        if dt == latest_dt:
            return lbl

# 6. 遍歷每個 patient/laterality/visit_folder，標記 stage
records = []
for _, row in df.iterrows():
    pid = str(row['research_id']).zfill(9)
    lat = row['laterality']  # 'L' or 'R'
    eye_dir = os.path.join(base_dir, pid, lat)
    if not os.path.isdir(eye_dir):
        continue

    for visit in os.listdir(eye_dir):
        path = os.path.join(eye_dir, visit)
        if not os.path.isdir(path):
            continue
        # 只處理 YYYYMMDD 命名的資料夾
        try:
            vdt = datetime.strptime(visit, '%Y%m%d')
        except ValueError:
            continue

        stage = get_stage_for_date(row, vdt)
        records.append({
            'research_id':    pid,
            'laterality':     lat,
            'baseline_stage': row.get('baseline_stage', None),
            'visit_date':     visit,      # YYYYMMDD
            'visit_date_dt':  vdt,        # datetime
            'stage':          stage
        })

# 7. 輸出到新的 Excel
out_df = pd.DataFrame(records)
out_df.sort_values(['research_id','laterality','visit_date_dt'], inplace=True)
out_df.to_excel(
    r"D:\cleaning_GUI_annotated_Data\volume_labels.xlsx",
    index=False
)

print("Done! 標記結果已寫入：")
print(r"D:\cleaning_GUI_annotated_Data\volume_labels.xlsx")
