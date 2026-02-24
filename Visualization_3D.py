# Visualization_3D.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime
from Draw_Utils import draw_field
from Config_Loader import config
from Logger_Setup import setup_logger

logger = setup_logger(__name__)

# 從配置讀取所有影片相關設定
VIDEO_CONFIG = {
    'output_dir': config.get('video', 'output_dir', default='videos'),
    'fps': config.get('video', 'fps', default=30),
    'dpi': config.get('video', 'dpi', default=200),
    'rotation': config.get('video', 'rotation', default=True),
    'bitrate': config.get('video', 'bitrate', default=1800),
    'codec': config.get('video', 'codec', default='h264'),
    'show_animation': config.get('video', 'show_animation', default=False),
    'default_interval': config.get('video', 'default_interval_ms', default=50)
}
PARK_CONFIG = {
    'foul_line_ft': config.get('park', 'foul_line_ft', default=328),
    'center_field_ft': config.get('park', 'center_field_ft', default=400)
}

class Baseball3DVisualizer:
    """棒球軌跡 3D 動畫視覺化器"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.trajectories = []
        self.labels = []
        self.colors = []
        
    def add_trajectory(self, traj_data, label="Trajectory", color='red'):
        """
        加入軌跡資料
        
        Args:
            traj_data: 包含 'x', 'y', 'z' 的字典
            label: 軌跡標籤
            color: 軌跡顏色
        """
        self.trajectories.append(traj_data)
        self.labels.append(label)
        self.colors.append(color)
        logger.debug(f"已加入軌跡: {label}, 點數: {len(traj_data['x'])}")
    
    def clear_trajectories(self):
        """清除所有軌跡"""
        self.trajectories = []
        self.labels = []
        self.colors = []
        logger.debug("已清除所有軌跡")
    
    def setup_3d_plot(self, title="Baseball Trajectory Simulation"):
        """設定 3D 圖表"""
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 繪製球場
        from Draw_Utils import draw_field
        draw_field(self.ax)
        
        # 設定視角
        self.ax.view_init(elev=20, azim=-45)
        
        # 設定座標範圍
        foul_line_m = PARK_CONFIG.get('foul_line_ft', 328) * 0.3048
        center_field_m = PARK_CONFIG.get('center_field_ft', 400) * 0.3048
        
        self.ax.set_xlim(-20, center_field_m + 20)
        self.ax.set_ylim(-20, center_field_m + 20)
        self.ax.set_zlim(0, 60)
        
        # 標籤
        self.ax.set_xlabel("X: 1st Base Line (m)")
        self.ax.set_ylabel("Y: 3rd Base Line (m)")
        self.ax.set_zlabel("Z: Height (m)")
        self.ax.set_title(title)
        
        return self.fig, self.ax
    
    def create_static_3d_plot(self, title="Baseball Trajectory Simulation"):
        """建立靜態 3D 圖（多條軌跡）"""
        self.setup_3d_plot(title)
        
        for traj, label, color in zip(self.trajectories, self.labels, self.colors):
            self.ax.plot(traj['x'], traj['y'], traj['z'], 
                        color=color, lw=2, label=label)
            
            # 標示起點和終點
            self.ax.scatter(traj['x'][0], traj['y'][0], traj['z'][0], 
                          color=color, s=100, marker='o', edgecolors='black')
            self.ax.scatter(traj['x'][-1], traj['y'][-1], 0, 
                          color=color, s=100, marker='x', edgecolors='black')
        
        self.ax.legend()
        plt.tight_layout()
        
        return self.fig


class TrajectoryVideoRecorder:
    """軌跡影片錄製器"""
    
    def __init__(self, output_dir=None):
        """
        初始化影片錄製器
        
        Args:
            output_dir: 影片輸出目錄（預設從 config 讀取）
        """
        if output_dir is None:
            output_dir = VIDEO_CONFIG.get('output_dir', 'videos')
        
        self.output_dir = output_dir
        self.fps = VIDEO_CONFIG.get('fps', 30)
        self.dpi = VIDEO_CONFIG.get('dpi', 200)
        self.rotation = VIDEO_CONFIG.get('rotation', True)  # 影片是否自動旋轉
        self.bitrate = VIDEO_CONFIG.get('bitrate', 1800)
        self.codec = VIDEO_CONFIG.get('codec', 'h264')
        self.show_animation = VIDEO_CONFIG.get('show_animation', False)
        
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"影片輸出目錄: {self.output_dir}")
        logger.debug(f"影片設定: fps={self.fps}, dpi={self.dpi}, rotation={self.rotation}")
    
    def record_single_trajectory(self, traj_data, filename=None, 
                                 title="Baseball Trajectory"):
        """
        錄製單一軌跡影片（儲存用，自動旋轉視角）
        
        Args:
            traj_data: 軌跡資料（包含 'x', 'y', 'z'）
            filename: 輸出檔名（預設自動產生）
            title: 影片標題
        
        Returns:
            str: 影片儲存路徑
        """
        # 自動產生檔名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.mp4"
        
        save_path = os.path.join(self.output_dir, filename)
        
        # 強制關閉互動模式（避免顯示）
        plt.ioff()
        
        # 創建圖形
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})
        
        # 繪製球場
        from Draw_Utils import draw_field
        draw_field(ax)
        
        # 設定初始視角
        ax.view_init(elev=20, azim=-45)
        
        # 設定座標範圍
        foul_line_m = PARK_CONFIG.get('foul_line_ft', 328) * 0.3048
        center_field_m = PARK_CONFIG.get('center_field_ft', 400) * 0.3048
        
        ax.set_xlim(-20, center_field_m + 20)
        ax.set_ylim(-20, center_field_m + 20)
        ax.set_zlim(0, 60)
        ax.set_xlabel("X: 1st Base Line (m)")
        ax.set_ylabel("Y: 3rd Base Line (m)")
        ax.set_zlabel("Z: Height (m)")
        
        # 初始化線條
        line, = ax.plot([], [], [], color='red', lw=2)
        point, = ax.plot([], [], [], color='red', marker='o', 
                        markersize=8, markeredgecolor='black')
        
        max_frame = len(traj_data['x'])
        
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point
        
        def update(frame):
            # 更新軌跡
            line.set_data(traj_data['x'][:frame+1], traj_data['y'][:frame+1])
            line.set_3d_properties(traj_data['z'][:frame+1])
            
            # 更新球的位置
            point.set_data([traj_data['x'][frame]], [traj_data['y'][frame]])
            point.set_3d_properties([traj_data['z'][frame]])
            
            # **影片專用：自動旋轉視角**
            if self.rotation:
                # 從 -45 度旋轉到 45 度
                azim = -45 + (frame / max_frame) * 90
                ax.view_init(elev=20, azim=azim)
            
            ax.set_title(f"{title} (Frame: {frame}/{max_frame-1})")
            return line, point
        
        # 建立動畫
        interval = VIDEO_CONFIG.get('default_interval', 50)
        anim = FuncAnimation(fig, update, frames=max_frame,
                           init_func=init, interval=interval, blit=True)
        
        # 儲存影片
        try:
            logger.info(f"正在渲染影片: {save_path}")
            writer = FFMpegWriter(fps=self.fps, metadata={'title': title}, 
                                 bitrate=self.bitrate, codec=self.codec)
            anim.save(save_path, writer=writer, dpi=self.dpi)
            logger.info(f"影片已儲存: {save_path}")
            
        except Exception as e:
            logger.error(f"儲存影片失敗: {e}", exc_info=True)
            logger.info("提示: 需要安裝 ffmpeg - 'conda install ffmpeg' 或 'apt-get install ffmpeg'")
            return None
        
        finally:
            # 儲存完成後關閉圖形
            plt.close(fig)
        
        return save_path
    
    def show_trajectory_animation(self, traj_data, title="Baseball Trajectory"):
        """
        顯示軌跡動畫（即時觀看用，可手動控制視角）
        
        Args:
            traj_data: 軌跡資料
            title: 動畫標題
        """
        # 啟用互動模式
        plt.ion()
        
        # 創建圖形
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})
        
        # 繪製球場
        from Draw_Utils import draw_field
        draw_field(ax)
        
        # 設定初始視角
        ax.view_init(elev=20, azim=-45)
        
        # 設定座標範圍
        foul_line_m = PARK_CONFIG.get('foul_line_ft', 328) * 0.3048
        center_field_m = PARK_CONFIG.get('center_field_ft', 400) * 0.3048
        
        ax.set_xlim(-20, center_field_m + 20)
        ax.set_ylim(-20, center_field_m + 20)
        ax.set_zlim(0, 60)
        ax.set_xlabel("X: 1st Base Line (m)")
        ax.set_ylabel("Y: 3rd Base Line (m)")
        ax.set_zlabel("Z: Height (m)")
        ax.set_title(title)
        
        # 初始化線條
        line, = ax.plot([], [], [], color='red', lw=2)
        point, = ax.plot([], [], [], color='red', marker='o', 
                        markersize=8, markeredgecolor='black')
        
        max_frame = len(traj_data['x'])
        
        def init():
            """初始化函數"""
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point
        
        def update(frame):
            """更新函數"""
            # 更新軌跡（從起點到目前位置）
            line.set_data(traj_data['x'][:frame+1], traj_data['y'][:frame+1])
            line.set_3d_properties(traj_data['z'][:frame+1])
            
            # 更新球的位置
            point.set_data([traj_data['x'][frame]], [traj_data['y'][frame]])
            point.set_3d_properties([traj_data['z'][frame]])
            
            # 更新標題
            ax.set_title(f"{title} (Frame: {frame}/{max_frame-1})")
            
            # 強制重繪整個圖形
            fig.canvas.draw_idle()
            
            return line, point
        
        # 建立動畫 - 關閉 blit 優化
        interval = VIDEO_CONFIG.get('default_interval', 50)
        anim = FuncAnimation(fig, update, frames=max_frame,
                            init_func=init, interval=interval, 
                            blit=False,  # <-- 關閉 blit，確保完整重繪
                            repeat=True)  # 讓動畫循環播放
        
        # 顯示動畫
        plt.show(block=True)
        
        # 清理資源
        plt.close(fig)
        
        return anim