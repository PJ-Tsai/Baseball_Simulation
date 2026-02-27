import csv
import json
from collections import defaultdict

def convert_angle_format(angle):
    """
    將原始角度 (-45~45) 轉換為 90~0 度
    90度 = 右外野 (-45度原始)
    0度 = 左外野 (45度原始)
    """
    # 修正：將映射邏輯改為 45 - angle
    return 45 - angle

def create_latest_park_profiles(csv_file_path, output_json_path):
    """
    創建只包含最新數據的球場輪廓
    
    輸出格式：
    {
        "ballparks": [
            {
                "park_id": 1,
                "name": "Angel Stadium",
                "angles_plot": [0, 1, 2, ...],
                "distances": [329.63, 339.01, ...],
                "wall_heights": [3.18, 3.21, ...]
            }
        ]
    }
    """
    
    # 使用字典收集每個球場的最新數據
    ballparks_dict = {}
    
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader:
            year = int(row[1])
            park_id = int(row[2])
            park_name = row[3]
            original_angle = int(row[5])
            fence_height = float(row[6])
            fence_distance = float(row[7])
            
            # 轉換角度
            plot_angle = convert_angle_format(original_angle)
            
            # 如果球場還不存在，初始化
            if park_name not in ballparks_dict:
                ballparks_dict[park_name] = {
                    "park_id": park_id,
                    "name": park_name,
                    "latest_year": year,
                    "data": {}  # 用角度作為key儲存最新的數據
                }
            else:
                # 如果這筆數據年份更新，更新latest_year
                if year > ballparks_dict[park_name]["latest_year"]:
                    ballparks_dict[park_name]["latest_year"] = year
            
            # 儲存或更新該角度的數據（保留最新年份的）
            current_data = ballparks_dict[park_name]["data"]
            if plot_angle not in current_data or year > current_data[plot_angle]["year"]:
                current_data[plot_angle] = {
                    "year": year,
                    "distance": round(fence_distance, 2),
                    "wall_height": round(fence_height, 2)
                }
    
    # 整理最終輸出
    ballparks_list = []
    for park_name, park_data in ballparks_dict.items():
        # 將數據轉換為三個平行列表
        angles = []
        distances = []
        wall_heights = []
        
        for angle in sorted(park_data["data"].keys()):
            angles.append(angle)
            distances.append(park_data["data"][angle]["distance"])
            wall_heights.append(park_data["data"][angle]["wall_height"])
        
        ballparks_list.append({
            "park_id": park_data["park_id"],
            "name": park_name,
            "latest_year": park_data["latest_year"],
            "angles_plot": angles,
            "distances": distances,
            "wall_heights": wall_heights
        })
    
    # 按球場名稱排序
    ballparks_list = sorted(ballparks_list, key=lambda x: x["name"])
    
    result = {
        "ballparks": ballparks_list
    }
    
    # 寫入JSON檔案
    with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(result, jsonfile, ensure_ascii=False, indent=2)
    
    print(f"成功轉換 {len(ballparks_list)} 個球場的數據到 {output_json_path}")
    
    # 輸出統計資訊
    print("\n=== 球場統計資訊 (最新年份數據) ===")
    for park in ballparks_list:
        print(f"ID: {park['park_id']:3d}, {park['name']:30s} | 最新年份: {park['latest_year']} | 數據點: {len(park['angles_plot'])}")
    
    return result

# 使用範例 - 選擇您需要的版本
if __name__ == "__main__":
    input_file = "ballpark_data.csv"
    
    # 版本1: 只保留最新數據
    output_file_latest = "ballpark_data.json"
    latest_profiles = create_latest_park_profiles(input_file, output_file_latest)
    
    print(f"\n轉換完成！")
    print(f"輸出文件: {output_file_latest}")
    
    # 顯示範例數據
    if latest_profiles["ballparks"]:
        example = latest_profiles["ballparks"][0]
        print(f"\n=== 範例數據: ID: {example['park_id']}, {example['name']} (最新年份: {example['latest_year']}) ===")
        print(f"  角度數量: {len(example['angles_plot'])}")
        print(f"  前5個角度: {example['angles_plot'][:5]}")
        print(f"  前5個距離: {example['distances'][:5]}")
        print(f"  前5個牆高: {example['wall_heights'][:5]}")