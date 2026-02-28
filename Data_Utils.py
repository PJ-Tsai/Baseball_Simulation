import pandas as pd
import numpy as np
import warnings
import pybaseball
from pybaseball import statcast
import argparse
import os
from datetime import datetime
from Config_Loader import config  # 直接導入已初始化的 config 實例
from Logger_Setup import setup_logger

# 設定日誌
logger = setup_logger(__name__)

# 從配置讀取設定 - 修正讀取方式
DATA_DIR = config.get('data', 'data_dir')  # 原本是 config.get('data', 'data_dir')
CACHE_ENABLED = config.get('data', 'cache_enabled')  # 原本是 config.get('data', 'cache_enabled')

# 修正：使用正確的 config.get 方式
park_config = config.get('park')
if park_config is None:
    park_config = {}

PARK_ID_MAPPING = park_config.get('park_id_mapping', {})
DEFAULT_PARK_ID = 0  # 預設球場 ID（如果找不到對應的話）

if CACHE_ENABLED:
    pybaseball.cache.enable()
    logger.info("已啟用 pybaseball 快取功能")

warnings.filterwarnings("ignore")

# 反轉 park_id_mapping 以便透過球場名稱查詢 ID
# 增加空值檢查
if PARK_ID_MAPPING:
    PARK_NAME_TO_ID = {v: k for k, v in PARK_ID_MAPPING.items()}
    logger.info(f"成功載入 {len(PARK_NAME_TO_ID)} 個球場映射")
else:
    PARK_NAME_TO_ID = {}
    logger.warning("未找到球場映射表，將使用預設值")

team_primary_stadium = {
    'ARI': 'Chase Field',
    'ATL': 'Truist Park',
    'ATH': 'Oakland Coliseum',  # 2024年前使用
    'AZ': 'Chase Field',
    'BAL': 'Oriole Park at Camden Yards',
    'BOS': 'Fenway Park',
    'CHC': 'Wrigley Field',
    'CHW': 'Rate Field',
    'CIN': 'Great American Ball Park',
    'CLE': 'Progressive Field',
    'COL': 'Coors Field',
    'DET': 'Comerica Park',
    'HOU': 'Daikin Park',
    'KCR': 'Kauffman Stadium',
    'LAA': 'Angel Stadium',
    'LAD': 'Dodger Stadium',
    'MIA': 'loanDepot park',
    'MIL': 'American Family Field',
    'MIN': 'Target Field',
    'NYM': 'Citi Field',
    'NYY': 'Yankee Stadium',
    'PHI': 'Citizens Bank Park',
    'PIT': 'PNC Park',
    'SDP': 'Petco Park',
    'SEA': 'T-Mobile Park',
    'SFG': 'Oracle Park',
    'STL': 'Busch Stadium',
    'TBR': 'Tropicana Field',
    'TEX': 'Globe Life Field',
    'TOR': 'Rogers Centre',
    'WSN': 'Nationals Park'
}

def get_park_id_from_name(park_name):
    """
    根據球場名稱取得對應的 park_id
    
    Args:
        park_name: 球場名稱
    
    Returns:
        int: 對應的 park_id，若找不到則回傳 default_park_id
    """
    if pd.isna(park_name):
        return DEFAULT_PARK_ID
    
    # 直接從映射表查詢
    park_id = PARK_NAME_TO_ID.get(park_name)
    
    if park_id is not None:
        return park_id
    
    # 處理一些常見的名稱差異
    common_variations = {
        'Minute Maid Park': 'Daikin Park',
        'Guaranteed Rate Field': 'Rate Field',
        'Minute Maid Field': 'Daikin Park',
        'Enron Field': 'Daikin Park',
        'Astros Field': 'Daikin Park',
        'Miller Park': 'American Family Field',
        'Safeco Field': 'T-Mobile Park',
        'AT&T Park': 'Oracle Park',
        'Pacific Bell Park': 'Oracle Park',
        'SBC Park': 'Oracle Park',
        'Candlestick Park': 'Oracle Park',
        'Jack Murphy Stadium': 'Petco Park',
        'Qualcomm Stadium': 'Petco Park',
        'Sun Life Stadium': 'loanDepot park',
        'Dolphin Stadium': 'loanDepot park',
        'Marlins Park': 'loanDepot park',
        'Joe Robbie Stadium': 'loanDepot park',
        'Turner Field': 'Truist Park',
        'The Ballpark in Arlington': 'Globe Life Field',
        'Rangers Ballpark': 'Globe Life Field',
        'Ameriquest Field': 'Globe Life Field',
    }
    
    if park_name in common_variations:
        return PARK_NAME_TO_ID.get(common_variations[park_name], DEFAULT_PARK_ID)
    
    logger.debug(f"找不到球場映射: {park_name}，使用預設ID {DEFAULT_PARK_ID}")
    return DEFAULT_PARK_ID

def match_game_to_stadium(home_team, game_date=None, game_pk=None):
    """
    根據主隊和比賽日期判斷實際比賽球場，並回傳 park_id
    
    Args:
        home_team: 主隊代碼
        game_date: 比賽日期（用於判斷特殊賽事和球隊搬遷）
        game_pk: 比賽ID（用於日誌）
    
    Returns:
        int: 對應的 park_id
    """
    # 先記錄實際收到的 home_team 值
    logger.debug(f"收到的主隊代碼: '{home_team}' (類型: {type(home_team)})")
    
    # 球隊搬遷/臨時球場處理
    team_relocation = {
        'OAK': [ 
            {
                'start': '2025-01-01', 
                'end': '2027-12-31', 
                'stadium': 'Sutter Health Park',
                'note': '暫時搬遷至沙加緬度'
            },
            # 2028年後會搬到拉斯維加斯，到時需要更新
        ],
        # 可以加入其他球隊的搬遷記錄
    }
    
    # 先保留原始球隊代碼，用於特殊賽事和搬遷判斷
    original_team = home_team
    
    # 標準化球隊代碼（用於一般球場查詢）
    team_code_map = {
        'CWS': 'CHW', 'KC': 'KCR', 'SD': 'SDP', 
        'SF': 'SFG', 'TB': 'TBR', 'WSH': 'WSN'
    }
    
    if home_team in team_code_map:
        home_team = team_code_map[home_team]
        logger.debug(f"標準化球隊代碼: {original_team} -> {home_team}")
    
    # 預設使用主隊的主要球場
    stadium_name = team_primary_stadium.get(home_team)
    
    # 檢查是否為球隊搬遷/臨時球場
    if game_date is not None:
        game_date = pd.to_datetime(game_date)
        
        # 檢查球隊搬遷
        if home_team in team_relocation:
            for period in team_relocation[home_team]:
                start_dt = pd.to_datetime(period['start'])
                end_dt = pd.to_datetime(period['end'])
                
                if start_dt <= game_date <= end_dt:
                    stadium_name = period['stadium']
                    logger.info(f"球隊搬遷 detected: {home_team} 使用 {stadium_name} ({period['note']}) (game_pk={game_pk}, 日期={game_date.date()})")
                    break
    
    # 特殊賽事處理 - 使用原始球隊代碼進行判斷
    if game_date is not None and game_pk is not None and stadium_name == team_primary_stadium.get(home_team):
        game_date = pd.to_datetime(game_date)
        game_year = game_date.year
        game_month = game_date.month
        game_day = game_date.day
        
        # 定義特殊賽事的具體日期範圍（根據實際比賽日期）
        special_events = {
            'London Stadium': [
                # 2019年倫敦賽：紅襪vs洋基
                {'start': '2019-06-29', 'end': '2019-06-30', 'teams': ['BOS', 'NYY']},
                # 2023年倫敦賽：小熊vs紅雀
                {'start': '2023-06-24', 'end': '2023-06-25', 'teams': ['CHC', 'STL']},
                # 2024年倫敦賽：費城人vs大都會
                {'start': '2024-06-08', 'end': '2024-06-09', 'teams': ['PHI', 'NYM']},
            ],
            'Field of Dreams': [
                # 2021年：白襪vs洋基
                {'start': '2021-08-12', 'end': '2021-08-12', 'teams': ['CHW', 'NYY']},
                # 2022年：小熊vs紅人
                {'start': '2022-08-11', 'end': '2022-08-11', 'teams': ['CHC', 'CIN']},
            ],
            'Sahlen Field': [
                # 2020-2021年藍鳥因疫情使用
                {'start': '2020-08-11', 'end': '2020-09-30', 'teams': ['TOR']},
                {'start': '2021-05-25', 'end': '2021-07-21', 'teams': ['TOR']},
            ],
            'TD Ballpark': [
                # 2020-2021年藍鳥春訓基地
                {'start': '2020-07-29', 'end': '2020-08-09', 'teams': ['TOR']},
                {'start': '2021-04-05', 'end': '2021-05-24', 'teams': ['TOR']},
            ],
            'Rickwood Field': [
                # 2024年巨人vs紅雀
                {'start': '2024-06-20', 'end': '2024-06-20', 'teams': ['SFG', 'STL']},
            ],
            'Speedway': [
                # 2024年小熊vs紅人（Bristol Motor Speedway）
                {'start': '2024-08-02', 'end': '2024-08-04', 'teams': ['CHC', 'CIN']},
            ],
            'Omaha': [
                # 大學世界大賽特別賽（如果有的話）
                {'start': '2024-06-18', 'end': '2024-06-20', 'teams': ['KCR', 'DET']},
            ]
        }
        
        # 檢查是否為特殊賽事
        for special_stadium, date_ranges in special_events.items():
            for date_range in date_ranges:
                start_dt = pd.to_datetime(date_range['start'])
                end_dt = pd.to_datetime(date_range['end'])
                
                # 檢查日期是否在範圍內
                if start_dt <= game_date <= end_dt:
                    # 檢查球隊是否匹配（使用原始球隊代碼）
                    if original_team in date_range['teams'] or home_team in date_range['teams']:
                        stadium_name = special_stadium
                        logger.info(f"特殊賽事 detected: {special_stadium} (game_pk={game_pk}, 日期={game_date.date()}, 球隊={original_team})")
                        break
            if stadium_name != team_primary_stadium.get(home_team):
                break
    
    # 如果找不到對應球場，使用預設名稱
    if stadium_name is None:
        logger.warning(f"找不到球隊 {original_team}/{home_team} 的對應球場")
        stadium_name = f"{original_team} Home Stadium"
    
    # 轉換為 park_id
    park_id = get_park_id_from_name(stadium_name)
    
    if game_pk:
        logger.info(f"比賽 {game_pk}: {original_team} -> {stadium_name} (ID: {park_id})")
    
    return park_id

def fetch_and_refine_data(start_date, end_date, data_dir=None):
    """
    抓取 MLB Statcast 數據並進行初步清洗與特徵工程，最後存成 CSV。
    加入 park_id 欄位，方便後續繪圖使用。
    """
    if data_dir is None:
        data_dir = DATA_DIR
        
    logger.info(f"啟動場內球數據抓取 ({start_date} 至 {end_date})")
    
    # 建立資料夾
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"已建立資料夾: {data_dir}")
    
    try:
        # 抓取原始資料
        logger.debug("正在呼叫 Statcast API...")
        raw_data = statcast(start_dt=start_date, end_dt=end_date)
        logger.info(f"成功抓取原始數據，筆數: {len(raw_data)}")
    except Exception as e:
        logger.error(f"抓取中斷！錯誤訊息: {e}", exc_info=True)
        return None
    
    if raw_data.empty:
        logger.warning("該時間範圍內沒有數據")
        return None

    # 偵錯：查看所有獨特的 home_team 值
    unique_home_teams = raw_data['home_team'].unique()
    logger.info(f"資料中所有獨特的主隊代碼: {unique_home_teams}")
    
    # 偵錯：查看特殊賽事期間的資料
    special_dates = [
        ('2019-06-29', '2019-06-30', '倫敦賽'),
        ('2023-06-24', '2023-06-25', '倫敦賽'),
        ('2021-08-12', '2021-08-12', '夢田之戰'),
        ('2024-06-20', '2024-06-20', 'Rickwood Field'),
    ]
    
    for start_d, end_d, event_name in special_dates:
        mask = (raw_data['game_date'] >= start_d) & (raw_data['game_date'] <= end_d)
        event_data = raw_data[mask]
        if not event_data.empty:
            logger.info(f"{event_name} 期間 ({start_d} 至 {end_d}) 有 {len(event_data)} 筆資料")
            teams_in_event = event_data[['home_team', 'away_team']].drop_duplicates()
            logger.info(f"  參與球隊: {teams_in_event.values.tolist()}")

    # 1. 篩選場內球 (hit_into_play)
    df = raw_data[raw_data['description'] == 'hit_into_play'].copy()
    logger.info(f"篩選後場內球筆數: {len(df)}")
    
    # 2. 為每筆資料匹配 park_id
    logger.info("正在為每筆資料匹配 park_id...")
    
    # 確保 game_date 為 datetime 類型
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    
    # 為每個獨特的 (game_pk, home_team) 組合獲取 park_id
    unique_games = df[['game_pk', 'home_team']].drop_duplicates()
    if 'game_date' in df.columns:
        unique_games = df[['game_pk', 'home_team', 'game_date']].drop_duplicates()
    
    # 建立 game_pk 到 park_id 的映射
    game_park_map = {}
    special_events_count = 0
    failed_matches = []
    
    for _, game in unique_games.iterrows():
        park_id = match_game_to_stadium(
            home_team=game['home_team'],
            game_date=game.get('game_date') if 'game_date' in game.index else None,
            game_pk=game['game_pk']
        )
        game_park_map[game['game_pk']] = park_id
        
        # 記錄匹配失敗的情況（如果 park_id 是預設值）
        if park_id == DEFAULT_PARK_ID:
            failed_matches.append((game['game_pk'], game['home_team'], game.get('game_date')))
    
    if failed_matches:
        logger.warning(f"有 {len(failed_matches)} 場比賽使用了預設球場 ID")
        for pk, team, date in failed_matches[:5]:  # 只顯示前5個
            logger.warning(f"  比賽 {pk}: {team} @ {date} -> 使用預設 ID {DEFAULT_PARK_ID}")
    
    logger.info(f"成功為 {len(game_park_map)} 場比賽匹配 park_id")
    
    # 將 park_id 合併回原始數據
    df['park_id'] = df['game_pk'].map(game_park_map)
    
    # 檢查是否有找不到 park_id 的資料
    missing_park = df['park_id'].isna().sum()
    if missing_park > 0:
        logger.warning(f"有 {missing_park} 筆資料無法匹配到 park_id，使用預設值 {DEFAULT_PARK_ID}")
        df.loc[df['park_id'].isna(), 'park_id'] = DEFAULT_PARK_ID
    
    # 3. 移除關鍵參數缺失的資料
    initial_count = len(df)
    df = df.dropna(subset=['launch_speed', 'launch_angle', 'hc_x', 'hc_y']).copy()
    logger.info(f"移除缺失值後筆數: {len(df)} (移除 {initial_count - len(df)} 筆)")
    
    # 4. 座標轉換為 Spray Angle
    df['spray_angle'] = np.degrees(np.arctan((df['hc_x'] - 125.42) / (198.27 - df['hc_y'])))
    logger.debug("已完成 spray angle 計算")

    # 5. 標籤映射
    label_map = config.get('labels')
    
    def label_result(row):
        ev = row['events']
        if ev == 'home_run': return 'HR'
        if ev == 'single': return 'SINGLE'
        if ev == 'double': return 'DOUBLE'
        if ev == 'triple': return 'TRIPLE'
        if ev in ['field_out', 'force_out', 'sac_fly', 'field_error', 
                  'grounded_into_double_play', 'fielders_choice']: 
            return 'OUT'
        return None

    df['result_label'] = df.apply(label_result, axis=1)
    df = df.dropna(subset=['result_label']).copy()
    
    # 統計各類別數量
    result_counts = df['result_label'].value_counts()
    logger.info(f"結果類別分布:\n{result_counts}")

    # 6. 選取最終特徵欄位
    cols = ['result_label', 'launch_speed', 'launch_angle', 'spray_angle', 
            'hit_distance_sc', 'bb_type', 'events',
            'park_id', 'home_team', 'game_pk', 'hc_x', 'hc_y']
    
    if 'game_date' in df.columns:
        cols.append('game_date')
    
    # 檢查所有欄位是否存在
    available_cols = [col for col in cols if col in df.columns]
    final_df = df[available_cols].copy()
    
    # 將 result_label 轉換為數值標籤（用於模型訓練）
    if label_map:
        final_df['label'] = final_df['result_label'].map(label_map)
    else:
        logger.warning("找不到標籤映射，跳過 label 轉換")
    
    # 檢查標籤轉換是否成功
    if 'label' in final_df.columns:
        label_missing = final_df['label'].isna().sum()
        if label_missing > 0:
            logger.warning(f"有 {label_missing} 筆資料無法轉換標籤")
            final_df = final_df.dropna(subset=['label'])
    
    # 儲存檔案
    file_name = f"ml_data_{start_date}_{end_date}.csv"
    save_path = os.path.join(data_dir, file_name)
    final_df.to_csv(save_path, index=False)
    
    logger.info(f"成功儲存至: {save_path} (樣本數: {len(final_df)})")
    
    # 統計 park_id 分布
    if 'park_id' in final_df.columns:
        park_distribution = final_df['park_id'].value_counts().head(10)
        logger.info(f"park_id 分布 (前10):\n{park_distribution}")
    
    return final_df

def date_check(selected_year, selected_month):
    """檢查輸入的年月是否在 MLB 賽季合理範圍內，並處理未來日期"""
    current_date = datetime.now()
    start_year = config.get('data', 'start_year')
    
    # 基本年份檢查
    if not (start_year <= selected_year <= current_date.year):
        error_msg = f"年份必須在 {start_year} 到 {current_date.year} 之間"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 賽季月份檢查 (3月春訓開始至11月世界大賽結束)
    if not (3 <= selected_month <= 11):
        error_msg = "MLB 賽季數據通常只存在於 3 月至 11 月之間"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    start_date = f"{selected_year}-{selected_month:02d}-01"
    end_dt_obj = pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)
    
    # 若查詢的是本月，結束日期不能超過今天
    if selected_year == current_date.year and selected_month == current_date.month:
        if end_dt_obj > current_date:
            end_dt_obj = current_date
            logger.info(f"查詢當月數據，結束日期調整為今天: {current_date}")
            
    end_date = end_dt_obj.strftime("%Y-%m-%d")
    
    logger.info(f"日期範圍: {start_date} 至 {end_date}")
    return start_date, end_date

def combine_datasets(csv_list, data_dir=None):
    """
    保留函式：合併多個 CSV 檔案。
    """
    if data_dir is None:
        data_dir = DATA_DIR
        
    logger.info(f"開始合併資料集 (共 {len(csv_list)} 個檔案)")
    df_list = []
    for f in csv_list:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            df_list.append(pd.read_csv(path))
            logger.debug(f"已載入: {f}")
        else:
            logger.warning(f"找不到檔案 {path}")
            
    if df_list:
        combined = pd.concat(df_list, axis=0, ignore_index=True)
        logger.info(f"合併完成，總筆數: {len(combined)}")
        return combined
    else:
        logger.error("沒有找到任何可合併的檔案")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Static Data Fetcher")
    parser.add_argument("--year", type=str, required=True, help="Year of Data (2015-2026)")
    parser.add_argument("--month", type=str, required=True, help="Month of Data (3-11)")
    parser.add_argument("--dir", type=str, default=None, help="Data Directory")
    
    args = parser.parse_args()
    
    try:
        start_date, end_date = date_check(int(args.year), int(args.month))
        fetch_and_refine_data(start_date, end_date, args.dir)
    except Exception as e:
        logger.error(f"程式執行失敗: {e}")
        raise