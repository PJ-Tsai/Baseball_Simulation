# Baseball_Predictor_GUI.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import datetime
import random
import sys
import io
import queue
import traceback
from contextlib import redirect_stdout, redirect_stderr

# 導入原有模組
from ML_Physics_Hybrid_Predictor import BaseballPredictorEngine
from Draw_Utils import list_available_parks, get_park_name_by_id, get_park_config
from Config_Loader import config

class TextRedirector(io.StringIO):
    """將 stdout/stderr 重定向到 tkinter 文字元件"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""
        self.queue = queue.Queue()
        
    def write(self, string):
        self.buffer += string
        # 使用 queue 來安全地更新 UI
        self.queue.put(string)
        # 排程在主執行緒中更新
        if hasattr(self.text_widget, 'master') and self.text_widget.master:
            self.text_widget.master.after(10, self._update_text)
        
    def _update_text(self):
        """在主執行緒中更新文字"""
        try:
            while not self.queue.empty():
                text = self.queue.get_nowait()
                self.text_widget.insert(tk.END, text)
                self.text_widget.see(tk.END)
                self.text_widget.update_idletasks()
        except queue.Empty:
            pass
        
    def flush(self):
        pass

class BaseballPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MLB 擊球分析混合模型系統")
        self.root.geometry("1400x900")
        
        # 設定圖示和主題
        style = ttk.Style()
        style.theme_use('clam')
        
        # 初始化引擎
        self.engine = None
        self.model_path = config.get('model', 'name', default='baseball_dual_model.pkl')
        
        # 可用球場列表
        self.park_mapping = config.get('park', 'park_id_mapping', default={0: "generic"})
        self.park_ids = sorted(self.park_mapping.keys())
        self.current_park_id = config.get('park', 'default_id', default=0)
        
        # 追蹤 after 任務的 ID
        self._after_tasks = []
        
        # 預測歷史記錄
        self.prediction_history = []
        
        # 輸出目錄設定
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 綁定自定義事件
        self.root.bind('<<ProcessPendingPlots>>', self._on_process_pending_plots)
        self.root.bind('<<ProcessPendingAnimations>>', self._on_process_pending_animations)
        self.root.bind('<<ProcessPendingVideos>>', self._on_process_pending_videos)
        
        # 創建主框架
        self.create_menu()
        self.create_main_layout()
        
        # 啟動時初始化引擎
        self.init_engine()

        # 初始化引擎後，設定 tk root
        if self.engine:
            self.engine.set_tk_root(root)
            # 設定 video_recorder 的 tk root
            if hasattr(self.engine, 'video_recorder'):
                self.engine.video_recorder.set_tk_root(root)
        
        # 設定定期檢查任務
        self._setup_periodic_tasks()

    def _setup_periodic_tasks(self):
        """設定定期執行的任務"""
        self._check_engine_tasks()
    
    def _schedule_after(self, ms, callback):
        """安全地排程 after 任務，並追蹤任務 ID"""
        try:
            after_id = self.root.after(ms, callback)
            self._after_tasks.append(after_id)
            return after_id
        except Exception:
            return None
    
    def _cancel_all_after_tasks(self):
        """取消所有排程的 after 任務"""
        for after_id in self._after_tasks:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        self._after_tasks.clear()
        
    def _check_engine_tasks(self):
        """定期檢查引擎的待處理任務"""
        # 檢查 root 是否還存在
        try:
            self.root.winfo_exists()
        except Exception:
            return  # root 已銷毀，停止排程
        
        if self.engine:
            try:
                # 檢查是否有待處理的圖表
                if hasattr(self.engine, '_pending_plots') and self.engine._pending_plots:
                    self._schedule_after(100, self._process_pending_plots)
                
                # 檢查是否有待處理的動畫
                if hasattr(self.engine, '_pending_animations') and self.engine._pending_animations:
                    self._schedule_after(100, self._process_pending_animations)
                
                # 檢查是否有待處理的影片
                if (hasattr(self.engine, 'video_recorder') and 
                    hasattr(self.engine.video_recorder, '_pending_videos') and 
                    self.engine.video_recorder._pending_videos):
                    self._schedule_after(100, self._process_pending_videos)
                        
            except Exception as e:
                print(f"檢查引擎任務時出錯: {e}")
        
        # 每 500ms 檢查一次，但只在 root 存在時
        try:
            if self.root.winfo_exists():
                after_id = self.root.after(500, self._check_engine_tasks)
                self._after_tasks.append(after_id)
        except Exception:
            pass  # root 已銷毀，忽略錯誤

    def _process_pending_plots(self):
        """處理待顯示的圖表"""
        if not self.engine or not hasattr(self.engine, '_pending_plots'):
            return
        
        try:
            # 檢查 root 是否存在
            if not self.root.winfo_exists():
                return
            
            # 直接呼叫事件處理方法
            self._on_process_pending_plots()
            
        except Exception as e:
            self.log_message(f"顯示圖表失敗: {e}", 'error')

    def _process_pending_animations(self):
        """處理待顯示的動畫"""
        if not self.engine or not hasattr(self.engine, '_pending_animations'):
            return
        
        try:
            # 檢查 root 是否存在
            if not self.root.winfo_exists():
                return
            
            # 直接呼叫事件處理方法
            self._on_process_pending_animations()
            
        except Exception as e:
            self.log_message(f"顯示動畫失敗: {e}", 'error')
    
    def _process_pending_videos(self):
        """處理待處理的影片"""
        if not self.engine or not hasattr(self.engine, 'video_recorder'):
            return
        
        try:
            # 檢查 root 是否存在
            if not self.root.winfo_exists():
                return
            
            # 直接呼叫事件處理方法
            self._on_process_pending_videos()
            
        except Exception as e:
            self.log_message(f"處理影片任務失敗: {e}", 'error')
  
    def _show_single_plot(self, result):
        """顯示單個圖表在嵌入視窗中"""
        try:
            # 獲取 figure
            fig = self.engine.visualize_result(
                result['input_speed'],
                result['input_angle'],
                result['input_spray'],
                result
            )
            
            # 清除舊的 canvas 內容
            for widget in self.static_plot_frame.winfo_children():
                widget.destroy()
            
            # 創建新的 canvas
            self.static_fig = fig
            self.static_canvas = FigureCanvasTkAgg(fig, master=self.static_plot_frame)
            self.static_canvas.draw()
            self.static_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 確保 figure 不會彈出新視窗
            plt.close(fig)  # 關閉可能存在的 figure 視窗
            
            # 切換到靜態圖表頁面
            self.plot_notebook.select(0)
            
            self.log_message("圖表已顯示在嵌入視窗中")
            
        except Exception as e:
            self.log_message(f"顯示圖表時出錯: {e}", 'error')
            traceback.print_exc()
    
    def _show_single_animation(self, traj, result):
        """顯示單個動畫在嵌入視窗中"""
        try:
            if self.show_animation_var.get():
                # 清除舊的動畫
                if hasattr(self, 'current_animation') and self.current_animation:
                    try:
                        self.current_animation.event_source.stop()
                    except:
                        pass
                
                # 清除舊的 canvas 內容
                for widget in self.animation_frame.winfo_children():
                    widget.destroy()
                
                # 重新初始化動畫圖表
                self._init_animation_plot()
                
                # 在嵌入的 axes 中顯示動畫
                self.current_animation = self.engine.video_recorder.show_trajectory_animation(
                    traj,
                    title=f"{result['result_class']}: {result['input_speed']:.0f}mph, {result['input_angle']:.0f}°",
                    park_id=result['park_id'],
                    embed_fig=self.animation_fig,
                    embed_ax=self.animation_ax
                )
                
                # 更新 canvas
                self.animation_canvas.draw()
                
                # 切換到動畫頁面
                self.plot_notebook.select(1)
                
                self.log_message("動畫已顯示在嵌入視窗中")
                
        except Exception as e:
            self.log_message(f"顯示動畫時出錯: {e}", 'error')
            traceback.print_exc()

    def _show_result_plots(self, result):
        """在主執行緒中顯示結果圖表"""
        try:
            # 顯示靜態圖表
            self.engine.visualize_result(
                result['input_speed'],
                result['input_angle'],
                result['input_spray'],
                result
            )
            
            # 顯示動畫
            if self.show_animation_var.get():
                self.engine.video_recorder.show_trajectory_animation(
                    result['trajectory'],
                    title=f"{result['result_class']}: {result['input_speed']:.0f}mph, {result['input_angle']:.0f}°",
                    park_id=result['park_id']  # 使用結果中的 park_id
                )
        except Exception as e:
            self.log_message(f"顯示圖表失敗: {e}", 'error')

    def init_engine(self):
        """初始化預測引擎"""
        try:
            self.log_message(f"正在初始化引擎 (模型: {self.model_path})...")
            
            # 重定向輸出
            self.old_stdout = sys.stdout
            self.old_stderr = sys.stderr
            self.text_redirector = TextRedirector(self.log_text)
            sys.stdout = self.text_redirector
            sys.stderr = self.text_redirector
            
            self.engine = BaseballPredictorEngine(model_path=self.model_path)
            self.engine.set_park(self.current_park_id)
            
            # 設定 tk root
            self.engine.set_tk_root(self.root)
            
            # 設定影片錄製器的動畫顯示選項
            self.engine.video_recorder.show_animation = self.show_animation_var.get()
            
            self.engine_status.config(text=f"引擎狀態: 已初始化 (球場: {get_park_name_by_id(self.current_park_id)})")
            self.log_message("引擎初始化成功")
            
        except Exception as e:
            self.log_message(f"引擎初始化失敗: {e}", 'error')
            traceback.print_exc(file=sys.stderr)
            self.engine = None
            self.engine_status.config(text="引擎狀態: 初始化失敗")
        finally:
            # 恢復輸出
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr
    
    def create_menu(self):
        """創建選單列"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 檔案選單
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="檔案", menu=file_menu)
        file_menu.add_command(label="載入模型...", command=self.load_model_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="儲存預測結果", command=self.manual_save_results)
        file_menu.add_command(label="匯出設定", command=self.export_config)
        file_menu.add_command(label="匯入設定", command=self.import_config)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing)
        
        # 工具選單
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="列出可用球場", command=self.show_available_parks)
        tools_menu.add_command(label="開啟輸出資料夾", command=self.open_output_folder)
        tools_menu.add_command(label="開啟影片資料夾", command=self.open_video_folder)
        
        # 說明選單
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="說明", menu=help_menu)
        help_menu.add_command(label="使用說明", command=self.show_help)
        help_menu.add_command(label="關於", command=self.show_about)
    
    def create_main_layout(self):
        """創建主界面布局"""
        # 使用 PanedWindow 分割左右區域
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左側控制面板
        left_frame = ttk.Frame(main_paned, width=400)
        main_paned.add(left_frame, weight=1)
        
        # 右側顯示區域（包含 Notebook）
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        
        # 創建左側控制面板內容
        self.create_control_panel(left_frame)
        
        # 創建右側 Notebook
        self.create_notebook(right_frame)
    
    def create_control_panel(self, parent):
        """創建左側控制面板"""
        # 標題
        title_label = ttk.Label(parent, text="⚾ 擊球分析控制面板", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # 模型狀態框架
        model_frame = ttk.LabelFrame(parent, text="模型狀態", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.model_status = ttk.Label(model_frame, text=f"模型: {self.model_path}")
        self.model_status.pack(anchor=tk.W)
        
        self.engine_status = ttk.Label(model_frame, text="引擎狀態: 未初始化")
        self.engine_status.pack(anchor=tk.W)
        
        # 球場選擇框架
        park_frame = ttk.LabelFrame(parent, text="球場選擇", padding=10)
        park_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 球場 ID 選擇
        id_frame = ttk.Frame(park_frame)
        id_frame.pack(fill=tk.X, pady=2)
        ttk.Label(id_frame, text="球場 ID:").pack(side=tk.LEFT)
        
        self.park_id_var = tk.StringVar(value=str(self.current_park_id))
        park_id_combo = ttk.Combobox(id_frame, textvariable=self.park_id_var,
                                     values=[str(id) for id in self.park_ids],
                                     width=10, state='readonly')
        park_id_combo.pack(side=tk.LEFT, padx=5)
        park_id_combo.bind('<<ComboboxSelected>>', self.on_park_change)
        
        # 球場名稱顯示
        self.park_name_var = tk.StringVar(value=get_park_name_by_id(self.current_park_id))
        ttk.Label(park_frame, textvariable=self.park_name_var, 
                 font=('Arial', 10, 'italic')).pack(pady=2)
        
        # 重新整理球場列表按鈕
        ttk.Button(park_frame, text="重新整理球場列表", 
                  command=self.refresh_park_list).pack(pady=5)
        
        # 輸入參數框架
        input_frame = ttk.LabelFrame(parent, text="擊球參數輸入", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 初速輸入
        speed_frame = ttk.Frame(input_frame)
        speed_frame.pack(fill=tk.X, pady=2)
        ttk.Label(speed_frame, text="初速 (mph):", width=12).pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="95.0")
        speed_entry = ttk.Entry(speed_frame, textvariable=self.speed_var, width=15)
        speed_entry.pack(side=tk.LEFT, padx=5)
        
        # 仰角輸入
        angle_frame = ttk.Frame(input_frame)
        angle_frame.pack(fill=tk.X, pady=2)
        ttk.Label(angle_frame, text="仰角 (°):", width=12).pack(side=tk.LEFT)
        self.angle_var = tk.StringVar(value="25.0")
        angle_entry = ttk.Entry(angle_frame, textvariable=self.angle_var, width=15)
        angle_entry.pack(side=tk.LEFT, padx=5)
        
        # 噴射角輸入
        spray_frame = ttk.Frame(input_frame)
        spray_frame.pack(fill=tk.X, pady=2)
        ttk.Label(spray_frame, text="噴射角 (°):", width=12).pack(side=tk.LEFT)
        self.spray_var = tk.StringVar(value="0.0")
        spray_entry = ttk.Entry(spray_frame, textvariable=self.spray_var, width=15)
        spray_entry.pack(side=tk.LEFT, padx=5)
        
        # 作弊模式框架
        cheat_frame = ttk.LabelFrame(parent, text="補償增益設定 (作弊模式)", padding=10)
        cheat_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # EV 增益
        ev_frame = ttk.Frame(cheat_frame)
        ev_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ev_frame, text="EV 增益:", width=10).pack(side=tk.LEFT)
        self.ev_boost_var = tk.StringVar(value="1.0")
        ev_spinbox = ttk.Spinbox(ev_frame, from_=1.0, to=2.0, increment=0.1,
                                 textvariable=self.ev_boost_var, width=10)
        ev_spinbox.pack(side=tk.LEFT, padx=5)
        
        # 距離增益
        dist_frame = ttk.Frame(cheat_frame)
        dist_frame.pack(fill=tk.X, pady=2)
        ttk.Label(dist_frame, text="距離增益:", width=10).pack(side=tk.LEFT)
        self.dist_boost_var = tk.StringVar(value="1.0")
        dist_spinbox = ttk.Spinbox(dist_frame, from_=1.0, to=2.0, increment=0.1,
                                   textvariable=self.dist_boost_var, width=10)
        dist_spinbox.pack(side=tk.LEFT, padx=5)
        
        # 影片設定框架
        video_frame = ttk.LabelFrame(parent, text="影片設定", padding=10)
        video_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.save_video_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(video_frame, text="儲存軌跡影片", 
                       variable=self.save_video_var).pack(anchor=tk.W)
        
        self.show_animation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(video_frame, text="顯示動畫", 
                       variable=self.show_animation_var).pack(anchor=tk.W)
        
        # 操作按鈕框架
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 主要操作按鈕
        ttk.Button(button_frame, text="執行單次預測", 
                  command=self.run_single_prediction,
                  style='Accent.TButton').pack(fill=tk.X, pady=2)
        
        ttk.Button(button_frame, text="隨機測試", 
                  command=self.run_random_test).pack(fill=tk.X, pady=2)
        
        ttk.Button(button_frame, text="批量CSV處理", 
                  command=self.run_batch_process).pack(fill=tk.X, pady=2)
        
        ttk.Button(button_frame, text="即時輸入模式", 
                  command=self.run_realtime_mode).pack(fill=tk.X, pady=2)
        
        # 重新初始化引擎按鈕
        ttk.Button(button_frame, text="重新初始化引擎", 
                  command=self.init_engine).pack(fill=tk.X, pady=5)
    
    def create_notebook(self, parent):
        """創建右側的 Notebook 標籤頁"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 輸出日誌頁面
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text="輸出日誌")
        self.create_log_tab()
        
        # 結果圖表頁面
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="結果圖表")
        self.create_plot_tab()
        
        # 統計資訊頁面
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="統計資訊")
        self.create_stats_tab()
        
        # 球場資訊頁面
        self.park_info_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.park_info_frame, text="球場資訊")
        self.create_park_info_tab()
    
    def create_log_tab(self):
        """創建日誌顯示頁面"""
        # 創建文字顯示區域
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame, wrap=tk.WORD, 
            font=('Consolas', 10),
            background='black',
            foreground='lightgreen'
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 設定標籤樣式
        self.log_text.tag_config('error', foreground='red')
        self.log_text.tag_config('warning', foreground='yellow')
        self.log_text.tag_config('info', foreground='lightgreen')
        
        # 按鈕框架
        btn_frame = ttk.Frame(self.log_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="清除日誌", 
                  command=self.clear_log).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_frame, text="儲存日誌", 
                  command=self.save_log).pack(side=tk.LEFT, padx=2)
    
    def create_plot_tab(self):
        """創建圖表顯示頁面"""
        # 創建 matplotlib 圖形框架
        self.plot_canvas_frame = ttk.Frame(self.plot_frame)
        self.plot_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # 創建一個 Notebook 來切換靜態圖表和動畫
        self.plot_notebook = ttk.Notebook(self.plot_canvas_frame)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True)
        
        # 靜態圖表頁面
        self.static_plot_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.static_plot_frame, text="靜態軌跡圖")
        
        # 動畫頁面
        self.animation_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.animation_frame, text="動畫軌跡")
        
        # 初始化靜態圖表
        self._init_static_plot()
        
        # 初始化動畫
        self._init_animation_plot()
        
        # 圖表控制按鈕
        control_frame = ttk.Frame(self.plot_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="儲存圖表", 
                command=self.save_plot).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(control_frame, text="清除圖表", 
                command=self.clear_plot).pack(side=tk.LEFT, padx=2)
        
    def _init_static_plot(self):
        """初始化靜態圖表"""
        # 創建 matplotlib figure
        self.static_fig = plt.Figure(figsize=(10, 8))
        self.static_ax = self.static_fig.add_subplot(111, projection='3d')
        
        # 創建 canvas
        self.static_canvas = FigureCanvasTkAgg(self.static_fig, master=self.static_plot_frame)
        self.static_canvas.draw()
        self.static_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 顯示初始提示
        self.static_ax.text(0.5, 0.5, 0.5, "Execution results will be displayed here after the prediction is completed.", 
                            ha='center', va='center', transform=self.static_ax.transAxes)
        self.static_canvas.draw()

    def _init_animation_plot(self):
        """初始化動畫圖表"""
        # 創建 matplotlib figure
        self.animation_fig = plt.Figure(figsize=(10, 8))
        self.animation_ax = self.animation_fig.add_subplot(111, projection='3d')
        
        # 創建 canvas
        self.animation_canvas = FigureCanvasTkAgg(self.animation_fig, master=self.animation_frame)
        self.animation_canvas.draw()
        self.animation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 顯示初始提示
        self.animation_ax.text(0.5, 0.5, 0.5, "Animation will be displayed here after the prediction is executed.", 
                            ha='center', va='center', transform=self.animation_ax.transAxes)
        self.animation_canvas.draw()
        self.current_animation = None
    
    def create_stats_tab(self):
        """創建統計資訊頁面"""
        # 統計資訊文字區域
        self.stats_text = scrolledtext.ScrolledText(
            self.stats_frame, wrap=tk.WORD,
            font=('Arial', 11),
            background='white'
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 初始化統計資訊
        self.prediction_history = []
        self.update_stats_display()
    
    def create_park_info_tab(self):
        """創建球場資訊頁面"""
        # 球場資訊文字區域
        self.park_info_text = scrolledtext.ScrolledText(
            self.park_info_frame, wrap=tk.WORD,
            font=('Arial', 11),
            background='white'
        )
        self.park_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 更新球場資訊
        self.update_park_info()
    
    def init_engine(self):
        """初始化預測引擎"""
        try:
            self.log_message(f"正在初始化引擎 (模型: {self.model_path})...")
            
            # 重定向輸出
            self.old_stdout = sys.stdout
            self.old_stderr = sys.stderr
            sys.stdout = TextRedirector(self.log_text)
            sys.stderr = TextRedirector(self.log_text)
            
            self.engine = BaseballPredictorEngine(model_path=self.model_path)
            self.engine.set_park(self.current_park_id)
            
            # 設定 tk root
            self.engine.set_tk_root(self.root)
            
            # 設定影片錄製器的動畫顯示選項
            self.engine.video_recorder.show_animation = self.show_animation_var.get()
            
            self.engine_status.config(text=f"引擎狀態: 已初始化 (球場: {get_park_name_by_id(self.current_park_id)})")
            self.log_message("引擎初始化成功")
            
        except Exception as e:
            self.log_message(f"引擎初始化失敗: {e}", 'error')
            self.engine = None
            self.engine_status.config(text="引擎狀態: 初始化失敗")
        finally:
            # 恢復輸出
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr
    
    def on_park_change(self, event=None):
        """球場變更事件"""
        try:
            park_id = int(self.park_id_var.get())
            self.current_park_id = park_id
            park_name = get_park_name_by_id(park_id)
            self.park_name_var.set(park_name)
            
            if self.engine:
                self.engine.set_park(park_id)
                self.log_message(f"球場已切換為: ID={park_id}, 名稱={park_name}")
                
                # 清除待處理的動畫（避免使用舊球場）
                if hasattr(self.engine, '_pending_animations'):
                    self.engine._pending_animations.clear()
            
            # 更新球場資訊
            self.update_park_info()
            
        except Exception as e:
            self.log_message(f"切換球場失敗: {e}", 'error')
    
    def refresh_park_list(self):
        """重新整理球場列表"""
        try:
            from Draw_Utils import load_specific_park_data
            load_specific_park_data()
            
            # 更新下拉選單
            park_combo = self.root.nametowidget(self.park_id_var.get())
            # 這裡需要找到正確的 widget 引用，簡化處理
            
            self.log_message("球場列表已重新整理")
            self.update_park_info()
            
        except Exception as e:
            self.log_message(f"重新整理球場列表失敗: {e}", 'error')
    
    def run_single_prediction(self):
        """執行單次預測"""
        if not self.engine:
            messagebox.showerror("錯誤", "引擎尚未初始化")
            return
        
        try:
            # 取得輸入參數
            speed = float(self.speed_var.get())
            angle = float(self.angle_var.get())
            spray = float(self.spray_var.get())
            ev_boost = float(self.ev_boost_var.get())
            dist_boost = float(self.dist_boost_var.get())
            
            self.log_message(f"\n{'='*60}")
            self.log_message(f"執行單次預測:")
            self.log_message(f"  初速: {speed:.1f} mph")
            self.log_message(f"  仰角: {angle:.1f}°")
            self.log_message(f"  噴射角: {spray:.1f}°")
            self.log_message(f"  球場: {self.park_name_var.get()} (ID: {self.current_park_id})")
            self.log_message(f"{'='*60}")
            
            # 使用執行緒執行預測，避免凍結界面
            def predict():
                try:
                    # 執行預測
                    result = self.engine.run_inference(
                        speed, angle, spray,
                        Is_plot=True,  # 讓圖表進入待處理佇列
                        Video_save=self.save_video_var.get(),
                        ev_boost=ev_boost,
                        dist_boost=dist_boost
                    )
                    
                    # 更新統計（在主執行緒中）
                    self.root.after(0, lambda: self._update_after_prediction(result))
                    
                except Exception as e:
                    self.root.after(0, lambda: self.log_message(f"預測失敗: {e}", 'error'))
                    traceback.print_exc()
            
            thread = threading.Thread(target=predict)
            thread.daemon = True
            thread.start()
            
        except ValueError as e:
            self.log_message(f"輸入參數格式錯誤: {e}", 'error')

    def _update_after_prediction(self, result):
        """預測完成後的更新（在主執行緒中執行）"""
        try:
            # 檢查 root 是否存在
            if not self.root.winfo_exists():
                return
                
            self.prediction_history.append(result)
            self.update_stats_display()
            self.notebook.select(1)
            self.log_message(f"\n預測完成: {result['result_class']}")
            
            # 自動儲存單次預測結果
            if len(self.prediction_history) % 10 == 0:  # 每10筆自動儲存一次
                self.save_results_to_csv(self.prediction_history[-10:], "auto_save")
                
        except Exception as e:
            self.log_message(f"更新顯示時出錯: {e}", 'error')
    
    def run_random_test(self):
        """執行隨機測試"""
        if not self.engine:
            messagebox.showerror("錯誤", "引擎尚未初始化")
            return
        
        # 開啟對話框詢問測試數量
        dialog = tk.Toplevel(self.root)
        dialog.title("隨機測試設定")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="請輸入測試數量:").pack(pady=10)
        
        count_var = tk.StringVar(value="5")
        entry = ttk.Entry(dialog, textvariable=count_var, width=10)
        entry.pack(pady=5)
        
        def confirm():
            try:
                count = int(count_var.get())
                dialog.destroy()
                self._run_random_test_thread(count)
            except ValueError:
                messagebox.showerror("錯誤", "請輸入有效的數字")
        
        ttk.Button(dialog, text="開始測試", command=confirm).pack(pady=10)
    
    def _run_random_test_thread(self, count):
        """在執行緒中執行隨機測試"""
        def test():
            try:
                # 使用 queue 來傳遞日誌訊息
                log_queue = queue.Queue()
                
                def log_message(msg, tag='info'):
                    log_queue.put((msg, tag))
                
                ev_boost = float(self.ev_boost_var.get())
                dist_boost = float(self.dist_boost_var.get())
                
                log_message(f"\n開始執行 {count} 組隨機測試...")
                
                results = []
                for i in range(count):
                    speed = random.uniform(50, 110)
                    angle = random.uniform(-15, 60)
                    spray = random.uniform(-60, 60)
                    
                    log_message(f"\n--- 測試組 {i+1} ---")
                    log_message(f"隨機參數: {speed:.1f}mph, {angle:.1f}°, {spray:.1f}°")
                    
                    result = self.engine.run_inference(
                        speed, angle, spray,
                        Is_plot=True,  # 讓圖表進入待處理佇列
                        Video_save=self.save_video_var.get(),
                        ev_boost=ev_boost,
                        dist_boost=dist_boost
                    )
                    
                    results.append(result)
                
                # 在主執行緒中更新 UI
                self.root.after(0, lambda: self._update_after_random_test(results, log_queue))
                
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"隨機測試失敗: {e}", 'error'))
                traceback.print_exc()
        
        thread = threading.Thread(target=test)
        thread.daemon = True
        thread.start()
    
    def _update_after_random_test(self, results, log_queue):
        """隨機測試完成後的更新（在主執行緒中執行）"""
        try:
            # 處理佇列中的日誌訊息
            while not log_queue.empty():
                msg, tag = log_queue.get_nowait()
                self.log_message(msg, tag)
            
            # 更新統計
            for result in results:
                self.prediction_history.append(result)
            
            self.update_stats_display()
            self.log_message(f"\n隨機測試完成，共 {len(results)} 組")
            
            # 儲存隨機測試結果
            if results:
                filepath = self.save_results_to_csv(results, "random_test")
                if filepath:
                    self.log_message(f"隨機測試結果已儲存")
        
        except Exception as e:
            self.log_message(f"更新顯示時出錯: {e}", 'error')
    
    def run_batch_process(self):
        """執行批量 CSV 處理"""
        if not self.engine:
            messagebox.showerror("錯誤", "引擎尚未初始化")
            return
        
        # 選擇 CSV 檔案
        file_path = filedialog.askopenfilename(
            title="選擇 CSV 檔案",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        def process():
            try:
                log_queue = queue.Queue()
                
                def log_message(msg, tag='info'):
                    log_queue.put((msg, tag))
                
                ev_boost = float(self.ev_boost_var.get())
                dist_boost = float(self.dist_boost_var.get())
                
                log_message(f"\n開始批量處理: {file_path}")
                
                # 讀取 CSV
                df = pd.read_csv(file_path)
                
                required_cols = ['launch_speed', 'launch_angle', 'spray_angle']
                if not all(col in df.columns for col in required_cols):
                    log_message(f"錯誤: CSV 必須包含欄位 {required_cols}", 'error')
                    return
                
                results = []
                total = len(df)
                
                for idx, row in df.iterrows():
                    log_message(f"\n處理第 {idx+1}/{total} 筆...")
                    
                    result = self.engine.run_inference(
                        row['launch_speed'], 
                        row['launch_angle'], 
                        row['spray_angle'],
                        Is_plot=False,
                        Video_save=self.save_video_var.get(),
                        ev_boost=ev_boost,
                        dist_boost=dist_boost
                    )
                    
                    results.append(result)
                
                # 儲存結果
                output_df = pd.DataFrame(results)
                if 'trajectory' in output_df.columns:
                    output_df = output_df.drop(columns=['trajectory'])
                
                output_dir = "outputs"
                os.makedirs(output_dir, exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{output_dir}/batch_output_{timestamp}.csv"
                output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                
                # 在主執行緒中更新 UI
                self.root.after(0, lambda: self._update_after_batch(results, output_path, log_queue))
                
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"批量處理失敗: {e}", 'error'))
                traceback.print_exc()
        
        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()
    
    def _update_after_batch(self, results, output_path, log_queue):
        """批量處理完成後的更新（在主執行緒中執行）"""
        try:
            # 處理佇列中的日誌訊息
            while not log_queue.empty():
                msg, tag = log_queue.get_nowait()
                self.log_message(msg, tag)
            
            # 更新統計
            for result in results:
                self.prediction_history.append(result)
            
            self.update_stats_display()
            
            # 批量處理的結果已經在 process 函數中儲存了
            self.log_message(f"\n批量處理完成，結果已儲存至: {output_path}")
            
        except Exception as e:
            self.log_message(f"更新顯示時出錯: {e}", 'error')
    
    def run_realtime_mode(self):
        """執行即時輸入模式"""
        if not self.engine:
            messagebox.showerror("錯誤", "引擎尚未初始化")
            return
        
        # 創建即時輸入視窗
        realtime_win = tk.Toplevel(self.root)
        realtime_win.title("即時輸入模式")
        realtime_win.geometry("400x500")
        realtime_win.transient(self.root)
        
        # 儲存即時模式的結果
        realtime_results = []
        
        # 輸入框架
        input_frame = ttk.LabelFrame(realtime_win, text="輸入參數", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(input_frame, text="初速 (mph):").grid(row=0, column=0, sticky=tk.W, pady=5)
        speed_entry = ttk.Entry(input_frame, width=15)
        speed_entry.grid(row=0, column=1, pady=5, padx=5)
        speed_entry.insert(0, "95")
        
        ttk.Label(input_frame, text="仰角 (°):").grid(row=1, column=0, sticky=tk.W, pady=5)
        angle_entry = ttk.Entry(input_frame, width=15)
        angle_entry.grid(row=1, column=1, pady=5, padx=5)
        angle_entry.insert(0, "25")
        
        ttk.Label(input_frame, text="噴射角 (°):").grid(row=2, column=0, sticky=tk.W, pady=5)
        spray_entry = ttk.Entry(input_frame, width=15)
        spray_entry.grid(row=2, column=1, pady=5, padx=5)
        spray_entry.insert(0, "0")
        
        # 結果顯示區域
        result_frame = ttk.LabelFrame(realtime_win, text="預測結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        result_text = scrolledtext.ScrolledText(result_frame, height=10)
        result_text.pack(fill=tk.BOTH, expand=True)
        
        def submit_prediction():
            try:
                speed = float(speed_entry.get())
                angle = float(angle_entry.get())
                spray = float(spray_entry.get())
                ev_boost = float(self.ev_boost_var.get())
                dist_boost = float(self.dist_boost_var.get())
                
                result = self.engine.run_inference(
                    speed, angle, spray,
                    Is_plot=False,
                    Video_save=self.save_video_var.get(),
                    ev_boost=ev_boost,
                    dist_boost=dist_boost
                )
                
                # 顯示結果
                result_text.insert(tk.END, f"\n{'='*40}\n")
                result_text.insert(tk.END, f"時間: {datetime.datetime.now().strftime('%H:%M:%S')}\n")
                result_text.insert(tk.END, f"參數: {speed:.1f}mph, {angle:.1f}°, {spray:.1f}°\n")
                result_text.insert(tk.END, f"結果: {result['result_class']}\n")
                result_text.insert(tk.END, f"距離: {result['pred_dist_ft']:.1f} ft\n")
                result_text.insert(tk.END, f"機率: {result['hit_prob']:.1%}\n")
                result_text.see(tk.END)
                
                # 更新主視窗統計
                self.prediction_history.append(result)
                realtime_results.append(result)
                self.update_stats_display()
                
            except ValueError as e:
                messagebox.showerror("錯誤", f"輸入格式錯誤: {e}")
        
        # 新增儲存按鈕
        def save_realtime_results():
            if realtime_results:
                filepath = self.save_results_to_csv(realtime_results, "realtime_log")
                if filepath:
                    result_text.insert(tk.END, f"\n已儲存 {len(realtime_results)} 筆結果\n")
            else:
                messagebox.showinfo("提示", "尚無預測結果")
        
        # 按鈕框架
        btn_frame = ttk.Frame(realtime_win)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(btn_frame, text="預測", command=submit_prediction).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="清除結果", 
                command=lambda: result_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="儲存結果", 
                command=save_realtime_results).pack(side=tk.LEFT, padx=5)
    
    def update_stats_display(self):
        """更新統計資訊顯示"""
        self.stats_text.delete(1.0, tk.END)
        
        if not self.prediction_history:
            self.stats_text.insert(tk.END, "尚無預測記錄")
            return
        
        # 計算統計
        total = len(self.prediction_history)
        results = [r['result_class'] for r in self.prediction_history]
        
        # 各類別計數
        from collections import Counter
        counter = Counter(results)
        
        self.stats_text.insert(tk.END, f"📊 預測統計 (共 {total} 筆)\n")
        self.stats_text.insert(tk.END, f"{'='*40}\n\n")
        
        self.stats_text.insert(tk.END, "結果分布:\n")
        for result, count in counter.items():
            percentage = count / total * 100
            self.stats_text.insert(tk.END, f"  {result}: {count} 筆 ({percentage:.1f}%)\n")
        
        self.stats_text.insert(tk.END, "\n")
        
        # 平均數據
        avg_speed = np.mean([r['input_speed'] for r in self.prediction_history])
        avg_dist = np.mean([r['pred_dist_ft'] for r in self.prediction_history])
        
        self.stats_text.insert(tk.END, f"平均初速: {avg_speed:.1f} mph\n")
        self.stats_text.insert(tk.END, f"平均距離: {avg_dist:.1f} ft\n")
        
        # 最近 10 筆記錄
        self.stats_text.insert(tk.END, f"\n📝 最近預測記錄:\n")
        for r in self.prediction_history[-10:]:
            self.stats_text.insert(tk.END, 
                f"  {r['input_speed']:.0f}mph, {r['input_angle']:.0f}°, "
                f"{r['input_spray']:.0f}° -> {r['result_class']}\n"
            )
    
    def update_park_info(self):
        """更新球場資訊顯示"""
        self.park_info_text.delete(1.0, tk.END)
        
        try:
            park_config = get_park_config(self.current_park_id)
            
            self.park_info_text.insert(tk.END, f"🏟️ 球場資訊: {park_config['name']}\n")
            self.park_info_text.insert(tk.END, f"{'='*50}\n\n")
            
            self.park_info_text.insert(tk.END, f"球場 ID: {self.current_park_id}\n")
            self.park_info_text.insert(tk.END, f"球場類型: {park_config['type']}\n\n")
            
            if park_config['type'] == 'generic':
                self.park_info_text.insert(tk.END, "通用球場設定:\n")
                self.park_info_text.insert(tk.END, f"  邊線距離: {park_config['foul_line_m']:.1f} m\n")
                self.park_info_text.insert(tk.END, f"  中外野距離: {park_config['center_field_m']:.1f} m\n")
                self.park_info_text.insert(tk.END, f"  牆高: {park_config['wall_height']:.1f} m\n")
            else:
                self.park_info_text.insert(tk.END, "特定球場設定:\n")
                self.park_info_text.insert(tk.END, f"  資料點數: {len(park_config['angles'])}\n")
                self.park_info_text.insert(tk.END, f"  最大距離: {np.max(park_config['distances']):.1f} m\n")
                self.park_info_text.insert(tk.END, f"  最小距離: {np.min(park_config['distances']):.1f} m\n")
                self.park_info_text.insert(tk.END, f"  平均牆高: {np.mean(park_config['wall_heights']):.2f} m\n")
                
                # 顯示主要角度距離
                self.park_info_text.insert(tk.END, f"\n主要點位:\n")
                for i, (angle, dist, height) in enumerate(zip(
                    park_config['angles'], 
                    park_config['distances'],
                    park_config['wall_heights']
                )):
                    if i % 3 == 0:  # 每三個點顯示一個
                        self.park_info_text.insert(tk.END, 
                            f"  {angle:3.0f}°: {dist:5.1f} m, 牆高 {height:.2f} m\n")
            
            # 列出所有可用球場
            self.park_info_text.insert(tk.END, f"\n\n📋 所有可用球場:\n")
            for pid in sorted(self.park_mapping.keys()):
                name = self.park_mapping[pid]
                marker = "✓" if pid == self.current_park_id else " "
                self.park_info_text.insert(tk.END, f"  [{marker}] ID {pid:4d}: {name}\n")
                
        except Exception as e:
            self.park_info_text.insert(tk.END, f"無法載入球場資訊: {e}")
    
    def log_message(self, message, tag='info'):
        """寫入日誌訊息"""
        self.log_text.insert(tk.END, message + "\n", tag)
        self.log_text.see(tk.END)
        self.log_text.update_idletasks()
    
    def clear_log(self):
        """清除日誌"""
        self.log_text.delete(1.0, tk.END)
    
    def save_log(self):
        """儲存日誌"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get(1.0, tk.END))
            self.log_message(f"日誌已儲存至: {file_path}")
    
    def save_plot(self):
        """儲存當前圖表"""
        # 這裡需要獲取當前顯示的圖表
        messagebox.showinfo("提示", "請從圖表視窗手動儲存 (使用工具欄的儲存按鈕)")
    
    def save_results_to_csv(self, results, prefix="predictions"):
        """將預測結果儲存為 CSV 檔案"""
        if not results:
            self.log_message("沒有結果可儲存")
            return None
        
        try:
            # 轉換為 DataFrame
            df_list = []
            for r in results:
                # 複製字典，排除 trajectory 物件
                r_copy = r.copy()
                if 'trajectory' in r_copy:
                    del r_copy['trajectory']
                df_list.append(r_copy)
            
            df = pd.DataFrame(df_list)
            
            # 生成檔案名稱
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # 儲存 CSV
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            self.log_message(f"結果已儲存至: {filepath}")
            return filepath
            
        except Exception as e:
            self.log_message(f"儲存結果失敗: {e}", 'error')
            return None
    
    def manual_save_results(self):
        """手動儲存預測結果"""
        if not self.prediction_history:
            messagebox.showinfo("提示", "尚無預測記錄")
            return
        
        filepath = self.save_results_to_csv(self.prediction_history, "manual_save")
        if filepath:
            messagebox.showinfo("成功", f"已儲存 {len(self.prediction_history)} 筆結果")

    def clear_plot(self):
        """清除圖表"""
        # 重新初始化靜態圖表
        self._init_static_plot()
        
        # 重新初始化動畫圖表
        self._init_animation_plot()
        
        self.log_message("圖表已清除")
        
    def load_model_dialog(self):
        """載入模型對話框"""
        file_path = filedialog.askopenfilename(
            title="選擇模型檔案",
            filetypes=[("PKL files", "*.pkl"), ("All files", "*.*")]
        )
        if file_path:
            self.model_path = file_path
            self.model_status.config(text=f"模型: {os.path.basename(file_path)}")
            self.init_engine()
    
    def export_config(self):
        """匯出設定"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if file_path:
            # 讀取當前設定並儲存
            import shutil
            shutil.copy2("config.yaml", file_path)
            self.log_message(f"設定已匯出至: {file_path}")
    
    def import_config(self):
        """匯入設定"""
        file_path = filedialog.askopenfilename(
            title="選擇設定檔案",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if file_path:
            # 複製設定檔
            import shutil
            shutil.copy2(file_path, "config.yaml")
            # 重新載入設定
            from Config_Loader import reload_config
            reload_config()
            self.log_message("設定已匯入並重新載入")
    
    def show_available_parks(self):
        """顯示可用球場列表"""
        from Draw_Utils import list_available_parks
        
        # 擷取輸出
        old_stdout = sys.stdout
        string_io = io.StringIO()
        sys.stdout = string_io
        
        list_available_parks()
        
        sys.stdout = old_stdout
        
        # 顯示在日誌中
        self.log_message("\n" + string_io.getvalue())
    
    def open_output_folder(self):
        """開啟輸出資料夾"""
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        if sys.platform == 'win32':
            os.startfile(output_dir)
        elif sys.platform == 'darwin':
            os.system(f'open "{output_dir}"')
        else:
            os.system(f'xdg-open "{output_dir}"')
    
    def open_video_folder(self):
        """開啟影片資料夾"""
        video_dir = config.get('video', 'output_dir', default='videos')
        os.makedirs(video_dir, exist_ok=True)
        
        if sys.platform == 'win32':
            os.startfile(video_dir)
        elif sys.platform == 'darwin':
            os.system(f'open "{video_dir}"')
        else:
            os.system(f'xdg-open "{video_dir}"')
    
    def show_help(self):
        """顯示使用說明"""
        help_text = """
         MLB 擊球分析混合模型系統 - 使用說明
        
        1. 基本操作流程:
           - 確認引擎初始化成功 (狀態顯示)
           - 選擇球場 ID
           - 輸入擊球參數
           - 點擊執行單次預測
        
        2. 三種主要模式:
           - 單次預測: 手動輸入參數進行預測
           - 隨機測試: 生成隨機數據批量測試
           - 批量CSV處理: 讀取 CSV 檔案批量預測
           - 即時輸入: 連續輸入模式
        
        3. 作弊模式增益:
           - EV 增益: 提高初速 (1.0 = 無增益)
           - 距離增益: 提高預測距離
        
        4. 輸出功能:
           - 可儲存軌跡影片
           - 顯示 3D 軌跡圖
           - 統計資訊自動更新
        
        5. 球場設定:
           - ID 0: 通用球場
           - 其他 ID: 對應 MLB 實際球場
        """
        
        messagebox.showinfo("使用說明", help_text)
    
    def show_about(self):
        """顯示關於資訊"""
        about_text = """
        MLB 擊球分析混合模型系統 v1.0
        
        這是一個結合機器學習與物理模擬的
        棒球擊球軌跡預測系統。
        
        特點:
        - 混合 AI + 物理模型
        - 支援 MLB 30+ 座球場
        - 3D 視覺化與影片輸出
        - 作弊模式增益調整
        
        © 2025 MLB Analytics
        """
        
        messagebox.showinfo("關於", about_text)

     # 在關閉視窗時取消所有 after 任務
    
    def on_closing(self):
        """關閉視窗時的處理"""
        try:
            self.log_message("正在關閉系統，清理資源...")
            
            # 自動儲存所有預測結果
            if self.prediction_history:
                self.log_message(f"自動儲存 {len(self.prediction_history)} 筆預測結果...")
                self.save_results_to_csv(self.prediction_history, "session_complete")
            
            # 取消所有排程的 after 任務
            self._cancel_all_after_tasks()
            
            # 清理引擎資源
            if self.engine:
                self.engine.cleanup()
                self.engine = None
            
            # 關閉所有 matplotlib 圖形
            plt.close('all')
            
            # 關閉視窗
            if messagebox.askokcancel("退出", "確定要退出系統？"):
                self.root.destroy()
                
        except Exception as e:
            print(f"關閉時出錯: {e}")
            self.root.destroy()

    def _on_process_pending_plots(self, event=None):
        """處理待顯示的圖表（由事件觸發）"""
        if not self.engine or not hasattr(self.engine, '_pending_plots'):
            return
        
        try:
            plots = self.engine._pending_plots.copy()
            self.engine._pending_plots.clear()
            
            for result in plots:
                self._show_single_plot(result)
                
        except Exception as e:
            self.log_message(f"顯示圖表失敗: {e}", 'error')

    def _on_process_pending_animations(self, event=None):
        """處理待顯示的動畫（由事件觸發）"""
        if not self.engine or not hasattr(self.engine, '_pending_animations'):
            return
        
        try:
            animations = self.engine._pending_animations.copy()
            self.engine._pending_animations.clear()
            
            for traj, result in animations:
                self._show_single_animation(traj, result)
                
        except Exception as e:
            self.log_message(f"顯示動畫失敗: {e}", 'error')

    def _on_process_pending_videos(self, event=None):
        """處理待處理的影片任務"""
        if not self.engine or not hasattr(self.engine, 'video_recorder'):
            return
        
        try:
            self.engine.video_recorder.process_pending_videos()
        except Exception as e:
            self.log_message(f"處理影片任務失敗: {e}", 'error')

def main():
    """主程式入口"""
    root = tk.Tk()
    app = BaseballPredictorGUI(root)
    
    # 設定關閉事件
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # 啟動主循環
    root.mainloop()

if __name__ == "__main__":
    main()