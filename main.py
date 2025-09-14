import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import threading
import subprocess
import os
import sys
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pylab import mpl
import glob
import datetime  # 用于生成带时间戳的输出文件名
# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei', 'STZhongsong']
mpl.rcParams['axes.unicode_minus'] = False
# 支持的算法列表 (包括确定性算法)
SUPPORTED_ALGS = ['PPO', 'A2C', 'DQN']
DETERMINISTIC_ALG_NAME = 'Deterministic' # 定义确定性算法的名称
QMIX_ALG_NAME = 'QMIX' # 定义 QMIX 算法名称
QTRAN_ALG_NAME = 'QTRAN' # 定义 QTRAN 算法名称

class UAVVisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UAV 任务分配可视化系统 Made by: GX7")
        self.root.geometry("1400x1000")
        # 存储数据
        self.results = {}  # 存储每个算法的结果
        self.inputs = {}   # 存储每个算法的输入路径
        self.models = {}   # 存储每个算法的模型路径
        # 初始化模型路径
        self.models['DQN'] = 'save_models/DQN_best_model.zip'
        self.models['A2C'] = 'save_models/A2C_best_model.zip'
        self.models['PPO'] = 'save_models/PPO_best_model.zip'
        # 训练进程相关
        self.training_process = None
        self.is_training = False
        self.setup_ui()

    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # 每个算法的控制面板
        self.alg_frames = {}
        self.alg_input_labels = {}
        self.alg_model_labels = {}
        for i, alg in enumerate(SUPPORTED_ALGS):
            alg_frame = ttk.LabelFrame(control_frame, text=f"{alg} 控制", padding="5")
            alg_frame.grid(row=0, column=i, padx=10)
            ttk.Label(alg_frame, text=f"{alg} 模型:").grid(row=0, column=0, sticky=tk.W)
            # 输入选择
            ttk.Button(alg_frame, text="选择输入JSON", command=lambda a=alg: self.select_input_json(a)).grid(row=1, column=0, pady=5)
            self.inputs[alg] = None
            # 输入文件标签
            input_label = ttk.Label(alg_frame, text="未选择输入文件", foreground="gray")
            input_label.grid(row=2, column=0, columnspan=2)
            self.alg_input_labels[alg] = input_label
            # 推理按钮
            ttk.Button(alg_frame, text="运行推理", command=lambda a=alg: self.run_inference(a)).grid(row=3, column=0, pady=5)
            # 加载结果按钮
            ttk.Button(alg_frame, text="加载结果", command=lambda a=alg: self.load_result(a)).grid(row=3, column=1, pady=5, padx=5)
            # 模型选择
            ttk.Button(alg_frame, text="选择模型", command=lambda a=alg: self.select_model(a)).grid(row=4, column=0, pady=5)
            model_label = ttk.Label(alg_frame, text="未选择模型", foreground="gray")
            model_label.grid(row=5, column=0, columnspan=2)
            self.alg_model_labels[alg] = model_label
            # 训练按钮
            ttk.Button(alg_frame, text="训练模型", command=lambda a=alg: self.train_model(a)).grid(row=6, column=0, pady=5)
            ttk.Button(alg_frame, text="加载最新模型", command=lambda a=alg: self.load_latest_model(a)).grid(row=6, column=1, pady=5, padx=5)
            self.alg_frames[alg] = alg_frame

        # DT 控制
        dt_frame = ttk.LabelFrame(control_frame, text="DT 控制", padding="5")
        dt_frame.grid(row=0, column=len(SUPPORTED_ALGS), padx=10)
        ttk.Label(dt_frame, text="DT 模型:").grid(row=0, column=0, sticky=tk.W)
        ttk.Button(dt_frame, text="运行DT推理", command=self.run_dt_inference).grid(row=1, column=0, pady=5)
        ttk.Button(dt_frame, text="加载DT结果", command=self.load_dt_results).grid(row=1, column=1, pady=5, padx=5)
        ttk.Button(dt_frame, text="选择输入JSON", command=self.select_dt_input_json).grid(row=2, column=0, pady=5)
        ttk.Button(dt_frame, text="训练DT模型", command=self.train_dt_model).grid(row=3, column=0, pady=5)
        ttk.Button(dt_frame, text="停止训练", command=self.stop_training).grid(row=3, column=1, pady=5, padx=5)
        ttk.Button(dt_frame, text="选择DT模型", command=self.select_dt_model).grid(row=4, column=0, pady=5)
        ttk.Button(dt_frame, text="加载最新模型", command=self.load_latest_dt_model).grid(row=4, column=1, pady=5, padx=5)
        self.dt_input_label = ttk.Label(dt_frame, text="未选择输入文件", foreground="gray")
        self.dt_input_label.grid(row=5, column=0, columnspan=2)
        self.dt_model_label = ttk.Label(dt_frame, text="未选择模型", foreground="gray")
        self.dt_model_label.grid(row=6, column=0, columnspan=2)
        self.dt_input_json = None
        self.dt_model_path = None
        self.results['DT'] = None

        # QMIX 控制 (已修改)
        qmix_frame = ttk.LabelFrame(control_frame, text="QMIX 控制", padding="5")
        qmix_frame.grid(row=0, column=len(SUPPORTED_ALGS) + 1, padx=10)
        ttk.Label(qmix_frame, text="QMIX 模型:").grid(row=0, column=0, sticky=tk.W)
        ttk.Button(qmix_frame, text="运行QMIX推理", command=self.run_qmix_inference).grid(row=1, column=0, pady=5) # 修改函数名
        ttk.Button(qmix_frame, text="加载QMIX结果", command=self.load_qmix_results).grid(row=1, column=1, pady=5, padx=5) # 修改函数名
        ttk.Button(qmix_frame, text="选择输入JSON", command=self.select_qmix_input_json).grid(row=2, column=0, pady=5) # 新增选择输入
        self.qmix_input_label = ttk.Label(qmix_frame, text="未选择输入文件", foreground="gray") # 新增输入文件标签
        self.qmix_input_label.grid(row=3, column=0, columnspan=2)
        self.qmix_input_json = None # 新增输入文件路径存储
        self.results[QMIX_ALG_NAME] = None # 初始化 QMIX 算法结果存储

        # QTRAN 控制 (新增)
        qtran_frame = ttk.LabelFrame(control_frame, text="QTRAN 控制", padding="5")
        qtran_frame.grid(row=0, column=len(SUPPORTED_ALGS) + 2, padx=10)
        ttk.Label(qtran_frame, text="QTRAN 模型:").grid(row=0, column=0, sticky=tk.W)
        ttk.Button(qtran_frame, text="运行QTRAN推理", command=self.run_qtran_inference).grid(row=1, column=0, pady=5)
        ttk.Button(qtran_frame, text="加载QTRAN结果", command=self.load_qtran_results).grid(row=1, column=1, pady=5, padx=5)
        ttk.Button(qtran_frame, text="选择输入JSON", command=self.select_qtran_input_json).grid(row=2, column=0, pady=5)
        self.qtran_input_label = ttk.Label(qtran_frame, text="未选择输入文件", foreground="gray")
        self.qtran_input_label.grid(row=3, column=0, columnspan=2)
        self.qtran_input_json = None
        self.results[QTRAN_ALG_NAME] = None

        # 确定性算法控制 (新增)
        det_frame = ttk.LabelFrame(control_frame, text="确定性算法 控制", padding="5")
        det_frame.grid(row=0, column=len(SUPPORTED_ALGS) + 3, padx=10) # 列索引更新
        ttk.Label(det_frame, text="确定性算法:").grid(row=0, column=0, sticky=tk.W)
        ttk.Button(det_frame, text="运行确定性推理", command=self.run_deterministic_inference).grid(row=1, column=0, pady=5)
        ttk.Button(det_frame, text="加载确定性结果", command=self.load_deterministic_results).grid(row=1, column=1, pady=5, padx=5)
        ttk.Button(det_frame, text="选择输入JSON", command=self.select_deterministic_input_json).grid(row=2, column=0, pady=5) # 新增选择输入
        self.deterministic_input_label = ttk.Label(det_frame, text="未选择输入文件", foreground="gray") # 新增输入文件标签
        self.deterministic_input_label.grid(row=3, column=0, columnspan=2)
        self.deterministic_input_json = None # 新增输入文件路径存储
        self.results[DETERMINISTIC_ALG_NAME] = None # 初始化确定性算法结果存储

        # 标签页
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 每个算法标签页
        self.alg_tabs = {}
        self.alg_texts = {}
        self.alg_fig_frames = {}
        for alg in SUPPORTED_ALGS:
            tab = ttk.Frame(notebook, padding="10")
            notebook.add(tab, text=f"{alg} 结果")
            self.alg_tabs[alg] = tab
            self.setup_alg_tab(tab, alg)

        # DT标签页
        dt_tab = ttk.Frame(notebook, padding="10")
        notebook.add(dt_tab, text="DT 结果")
        self.setup_dt_tab(dt_tab)

        # QMIX标签页 (已修改)
        qmix_tab = ttk.Frame(notebook, padding="10")
        notebook.add(qmix_tab, text="QMIX 结果")
        self.setup_qmix_tab(qmix_tab) # 修改函数名

        # QTRAN标签页 (新增)
        qtran_tab = ttk.Frame(notebook, padding="10")
        notebook.add(qtran_tab, text="QTRAN 结果")
        self.setup_qtran_tab(qtran_tab) # 新增设置标签页函数

        # 确定性算法标签页 (新增)
        deterministic_tab = ttk.Frame(notebook, padding="10")
        notebook.add(deterministic_tab, text="确定性算法 结果")
        self.setup_deterministic_tab(deterministic_tab) # 新增设置标签页函数

        # 对比标签页
        compare_tab = ttk.Frame(notebook, padding="10")
        notebook.add(compare_tab, text="模型对比")
        self.setup_compare_tab(compare_tab)

        # 日志标签页
        log_tab = ttk.Frame(notebook, padding="10")
        notebook.add(log_tab, text="训练日志")
        self.setup_log_tab(log_tab)

        # 对比模型选择 (更新下拉框选项)
        compare_frame = ttk.Frame(compare_tab)
        compare_frame.pack(fill=tk.X, pady=5)
        # 包含确定性算法、QMIX、QTRAN 的选项
        all_model_options = SUPPORTED_ALGS + ['DT', QMIX_ALG_NAME, QTRAN_ALG_NAME, DETERMINISTIC_ALG_NAME]
        self.compare_model1 = tk.StringVar(value="PPO")
        self.compare_model2 = tk.StringVar(value="DT")
        ttk.Label(compare_frame, text="对比模型1:").pack(side=tk.LEFT)
        ttk.Combobox(compare_frame, textvariable=self.compare_model1, values=all_model_options, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Label(compare_frame, text="对比模型2:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Combobox(compare_frame, textvariable=self.compare_model2, values=all_model_options, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(compare_frame, text="刷新对比", command=self.update_comparison).pack(side=tk.LEFT, padx=10)
        # 添加加载特定结果文件的按钮
        ttk.Button(compare_frame, text="加载模型1结果", command=self.load_compare_model1).pack(side=tk.LEFT, padx=10)
        ttk.Button(compare_frame, text="加载模型2结果", command=self.load_compare_model2).pack(side=tk.LEFT, padx=10)

    # --- UI Setup Functions (Existing and New) ---
    def setup_alg_tab(self, parent, alg):
        paned_window = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        json_frame = ttk.LabelFrame(paned_window, text="JSON 数据")
        paned_window.add(json_frame, weight=1)
        text = tk.Text(json_frame, wrap=tk.WORD, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(json_frame, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.alg_texts[alg] = text
        viz_frame = ttk.LabelFrame(paned_window, text="可视化")
        paned_window.add(viz_frame, weight=1)
        fig_frame = ttk.Frame(viz_frame)
        fig_frame.pack(fill=tk.BOTH, expand=True)
        self.alg_fig_frames[alg] = fig_frame

    def setup_dt_tab(self, parent):
        paned_window = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        json_frame = ttk.LabelFrame(paned_window, text="JSON 数据")
        paned_window.add(json_frame, weight=1)
        self.dt_text = tk.Text(json_frame, wrap=tk.WORD, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(json_frame, orient=tk.VERTICAL, command=self.dt_text.yview)
        self.dt_text.configure(yscrollcommand=scrollbar.set)
        self.dt_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        viz_frame = ttk.LabelFrame(paned_window, text="可视化")
        paned_window.add(viz_frame, weight=1)
        self.dt_fig_frame = ttk.Frame(viz_frame)
        self.dt_fig_frame.pack(fill=tk.BOTH, expand=True)

    # --- QMIX UI Setup (已修改) ---
    def setup_qmix_tab(self, parent): # 修改函数名
        paned_window = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        json_frame = ttk.LabelFrame(paned_window, text="JSON 数据")
        paned_window.add(json_frame, weight=1)
        self.qmix_text = tk.Text(json_frame, wrap=tk.WORD, font=('Consolas', 10)) # 修改变量名
        scrollbar = ttk.Scrollbar(json_frame, orient=tk.VERTICAL, command=self.qmix_text.yview) # 修改变量名
        self.qmix_text.configure(yscrollcommand=scrollbar.set) # 修改变量名
        self.qmix_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True) # 修改变量名
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        viz_frame = ttk.LabelFrame(paned_window, text="可视化")
        paned_window.add(viz_frame, weight=1)
        self.qmix_fig_frame = ttk.Frame(viz_frame) # 修改变量名
        self.qmix_fig_frame.pack(fill=tk.BOTH, expand=True) # 修改变量名

    # --- QTRAN UI Setup (新增) ---
    def setup_qtran_tab(self, parent): # 新增函数
        paned_window = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        json_frame = ttk.LabelFrame(paned_window, text="JSON 数据")
        paned_window.add(json_frame, weight=1)
        self.qtran_text = tk.Text(json_frame, wrap=tk.WORD, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(json_frame, orient=tk.VERTICAL, command=self.qtran_text.yview)
        self.qtran_text.configure(yscrollcommand=scrollbar.set)
        self.qtran_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        viz_frame = ttk.LabelFrame(paned_window, text="可视化")
        paned_window.add(viz_frame, weight=1)
        self.qtran_fig_frame = ttk.Frame(viz_frame)
        self.qtran_fig_frame.pack(fill=tk.BOTH, expand=True)

    def setup_deterministic_tab(self, parent): # 新增确定性算法标签页设置
        paned_window = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        json_frame = ttk.LabelFrame(paned_window, text="JSON 数据")
        paned_window.add(json_frame, weight=1)
        self.deterministic_text = tk.Text(json_frame, wrap=tk.WORD, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(json_frame, orient=tk.VERTICAL, command=self.deterministic_text.yview)
        self.deterministic_text.configure(yscrollcommand=scrollbar.set)
        self.deterministic_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        viz_frame = ttk.LabelFrame(paned_window, text="可视化")
        paned_window.add(viz_frame, weight=1)
        self.deterministic_fig_frame = ttk.Frame(viz_frame)
        self.deterministic_fig_frame.pack(fill=tk.BOTH, expand=True)

    def setup_compare_tab(self, parent):
        compare_frame = ttk.Frame(parent)
        compare_frame.pack(fill=tk.BOTH, expand=True)
        perf_frame = ttk.LabelFrame(compare_frame, text="性能对比")
        perf_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.compare_fig_frame = ttk.Frame(perf_frame)
        self.compare_fig_frame.pack(fill=tk.BOTH, expand=True)
        detail_frame = ttk.LabelFrame(compare_frame, text="详细对比")
        detail_frame.pack(fill=tk.BOTH, expand=True)
        self.compare_text = tk.Text(detail_frame, wrap=tk.WORD, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.compare_text.yview)
        self.compare_text.configure(yscrollcommand=scrollbar.set)
        self.compare_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_log_tab(self, parent):
        log_frame = ttk.Frame(parent)
        log_frame.pack(fill=tk.BOTH, expand=True)
        control_frame = ttk.Frame(log_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(control_frame, text="清空日志", command=self.clear_log).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="保存日志", command=self.save_log).pack(side=tk.LEFT)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=('Consolas', 10), height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_status_label = ttk.Label(log_frame, text="就绪", foreground="green")
        self.log_status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # --- General Functions ---
    def select_input_json(self, alg):
        file_path = filedialog.askopenfilename(
            title=f"选择 {alg} 输入JSON文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.inputs[alg] = file_path
            filename = os.path.basename(file_path)
            self.alg_input_labels[alg].config(text=f"输入文件: {filename}", foreground="green")

    def select_model(self, alg):
        file_path = filedialog.askopenfilename(
            title=f"选择 {alg} 模型文件",
            filetypes=[("Model files", "*.zip"), ("All files", "*.*")]
        )
        if file_path:
            self.models[alg] = file_path
            filename = os.path.basename(file_path)
            self.alg_model_labels[alg].config(text=f"模型: {filename}", foreground="green")

    def load_latest_model(self, alg):
        try:
            model_dir = "save_models"
            if not os.path.exists(model_dir):
                messagebox.showwarning("警告", f"找不到模型目录: {model_dir}")
                return
            pattern = os.path.join(model_dir, f"{alg}_best_model.zip")
            files = glob.glob(pattern)
            if not files:
                messagebox.showinfo("信息", f"未找到 {alg} 模型文件")
                return
            latest_model = max(files, key=os.path.getmtime)
            self.models[alg] = latest_model
            filename = os.path.basename(latest_model)
            self.alg_model_labels[alg].config(text=f"模型: {filename} (最新)", foreground="green")
            self.log(f"已加载 {alg} 最新模型: {filename}")
            messagebox.showinfo("成功", f"已加载 {alg} 最新模型: {filename}")
        except Exception as e:
            self.log(f"加载最新模型时出错: {str(e)}")
            messagebox.showerror("错误", f"加载最新模型时出错: {str(e)}")

    def run_inference(self, alg):
        def run():
            try:
                cmd = [
                    sys.executable, 'infer/infer_mul_alg.py',
                    '--alg_name', alg,
                    '--mode', 'test',
                    '--num_episodes', '1'
                ]
                if self.inputs[alg]:
                    cmd.extend(['--input_json', self.inputs[alg]])
                if self.models[alg]:
                    cmd.extend(['--model_path', self.models[alg]])
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', cwd=os.getcwd())
                if result.returncode == 0:
                    messagebox.showinfo("成功", f"{alg} 推理完成！")
                    self.find_and_load_latest_result(alg)
                else:
                    messagebox.showerror("错误", f"{alg} 推理失败:\n{result.stderr}")
            except Exception as e:
                messagebox.showerror("错误", f"运行 {alg} 推理时出错: {str(e)}")
        if not os.path.exists('infer/infer_mul_alg.py'):
            messagebox.showerror("错误", "找不到推理脚本")
            return
        threading.Thread(target=run, daemon=True).start()

    def find_and_load_latest_result(self, alg):
        try:
            pattern = f"outputs_json/{alg.lower()}/{alg.lower()}_episode_output_*.json"
            files = glob.glob(pattern)
            if files:
                latest = max(files, key=os.path.getctime)
                self.load_result_file(alg, latest)
            else:
                messagebox.showinfo("信息", f"未找到 {alg} 结果文件")
        except Exception as e:
            messagebox.showwarning("警告", f"查找 {alg} 结果文件时出错: {str(e)}")

    def load_result(self, alg):
        file_path = filedialog.askopenfilename(
            title=f"选择 {alg} 结果文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.load_result_file(alg, file_path)

    def load_result_file(self, alg, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.results[alg] = json.load(f)
            self.alg_texts[alg].delete(1.0, tk.END)
            self.alg_texts[alg].insert(1.0, json.dumps(self.results[alg], indent=2, ensure_ascii=False))
            self.update_alg_visualization(alg)
            self.update_comparison()
            messagebox.showinfo("成功", f"{alg} 结果文件加载成功: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"加载 {alg} 文件失败: {str(e)}")

    def update_alg_visualization(self, alg):
        frame = self.alg_fig_frames[alg]
        for widget in frame.winfo_children():
            widget.destroy()
        data = self.results.get(alg)
        if not data:
            return
        fig = Figure(figsize=(12, 10))
        if 'steps' in data and len(data['steps']) > 0:
            steps = [s['step'] for s in data['steps']]
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            rewards = [s.get('reward', 0) for s in data['steps']]
            cumulative_rewards = []
            cum_sum = 0
            for r in rewards:
                cum_sum += r
                cumulative_rewards.append(cum_sum)
            ax1.plot(steps, rewards, 'b-', label='单步奖励', linewidth=1.5)
            ax1.plot(steps, cumulative_rewards, 'r-', label='累计奖励', linewidth=1.5)
            ax1.set_title(f"{alg} 奖励曲线")
            ax1.set_xlabel("步数")
            ax1.set_ylabel("奖励")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            if 'friendly_remaining' in data['steps'][0]:
                friendly_interceptor = [s['friendly_remaining'].get('interceptor', 0) for s in data['steps']]
                friendly_recon = [s['friendly_remaining'].get('recon', 0) for s in data['steps']]
                enemy_ground_attack = [s['enemy_remaining'].get('ground_attack', 0) for s in data['steps']]
                enemy_recon = [s['enemy_remaining'].get('recon', 0) for s in data['steps']]
                ax2.plot(steps, friendly_interceptor, 'g-', label='友方拦截机', linewidth=1.5)
                ax2.plot(steps, friendly_recon, 'b-', label='友方侦察机', linewidth=1.5)
                ax2.set_title("友方无人机数量")
                ax2.set_xlabel("步数")
                ax2.set_ylabel("数量")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax3.plot(steps, enemy_ground_attack, 'r-', label='敌方攻击机', linewidth=1.5)
                ax3.plot(steps, enemy_recon, 'm-', label='敌方侦察机', linewidth=1.5)
                ax3.set_title("敌方无人机数量")
                ax3.set_xlabel("步数")
                ax3.set_ylabel("数量")
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, '无无人机数据', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title("友方无人机数量")
                ax3.text(0.5, 0.5, '无无人机数据', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title("敌方无人机数量")
            if 'action' in data['steps'][0]:
                interceptor_actions = []
                recon_actions = []
                escort_actions = []
                for s in data['steps']:
                    action = s.get('action', {})
                    interceptor_actions.append(
                        action.get('interceptor', {}).get('count', 0) if isinstance(action.get('interceptor', {}),
                                                                                    dict) else action.get('interceptor',
                                                                                                          0))
                    recon_actions.append(action.get('recon', {}).get('count', 0) if isinstance(action.get('recon', {}),
                                                                                               dict) else action.get(
                        'recon', 0))
                    escort_actions.append(
                        action.get('escort', {}).get('count', 0) if isinstance(action.get('escort', {}),
                                                                               dict) else action.get('escort', 0))
                x_pos = np.arange(len(steps))
                width = 0.25
                ax4.bar(x_pos - width, interceptor_actions, width, label='拦截机动作', alpha=0.8)
                ax4.bar(x_pos, recon_actions, width, label='侦察机动作', alpha=0.8)
                ax4.bar(x_pos + width, escort_actions, width, label='护航机动作', alpha=0.8)
                ax4.set_title("动作分布")
                ax4.set_xlabel("步数")
                ax4.set_ylabel("动作数量")
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, '无动作数据', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title("动作分布")
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, '无步骤数据', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f"{alg} 结果")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- QMIX Functions (已修改) ---
    def select_qmix_input_json(self): # 新增选择 QMIX 输入文件
        file_path = filedialog.askopenfilename(
            title="选择 QMIX 输入JSON文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.qmix_input_json = file_path
            filename = os.path.basename(file_path)
            self.qmix_input_label.config(text=f"输入文件: {filename}", foreground="green")

    def run_qmix_inference(self): # 修改函数名和逻辑
        def run():
            try:
                if not self.qmix_input_json:
                    messagebox.showwarning("警告", "请先选择 QMIX 的输入JSON文件")
                    return
                # 生成带时间戳的唯一输出文件名
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"qmix_inference_output_{timestamp}.json"
                output_dir = os.path.join(os.getcwd(), "outputs_json", "Qmix")
                os.makedirs(output_dir, exist_ok=True) # 确保目录存在
                output_path = os.path.join(output_dir, output_filename)
                cmd = [
                    sys.executable, 'infer/infer_qmix.py', # 修改调用脚本
                    '--config', self.qmix_input_json, # 使用 --config 参数
                    '--device', 'cpu' # 可根据需要添加其他参数
                ]
                self.log(f"运行 QMIX 推理: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', cwd=os.getcwd())
                if result.returncode == 0:
                    self.log("QMIX 推理完成！")
                    messagebox.showinfo("成功", f"QMIX 推理完成！输出文件: {output_filename}")
                    import time
                    # 自动加载生成的结果文件
                    self.find_and_load_latest_qmix_result()
                else:
                    error_msg = f"QMIX 推理失败:\n{result.stderr}"
                    self.log(error_msg)
                    messagebox.showerror("错误", error_msg)
            except Exception as e:
                error_msg = f"运行 QMIX 推理时出错: {str(e)}"
                self.log(error_msg)
                messagebox.showerror("错误", error_msg)
        if not os.path.exists('infer/infer_qmix.py'): # 检查新脚本
            messagebox.showerror("错误", "找不到 QMIX 推理脚本: infer/infer_qmix.py")
            return
        threading.Thread(target=run, daemon=True).start()

    def find_and_load_latest_qmix_result(self): # 修改函数名和查找逻辑
        try:
            # 查找 outputs_json/Qmix 目录下最新的 qmix_inference_output_*.json 文件
            pattern = os.path.join(os.getcwd(), "outputs_json", "Qmix", "qmix_inference_output_*.json")
            files = glob.glob(pattern)
            if files:
                latest = max(files, key=os.path.getctime)
                self.load_qmix_file(latest)
            else:
                messagebox.showinfo("信息", "未找到 QMIX 结果文件")
        except Exception as e:
            messagebox.showwarning("警告", f"查找 QMIX 结果文件时出错: {str(e)}")

    def load_qmix_results(self): # 修改函数名
        file_path = filedialog.askopenfilename(
            title="选择 QMIX 结果文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.load_qmix_file(file_path)

    def load_qmix_file(self, file_path): # 修改函数名和变量名
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.results[QMIX_ALG_NAME] = json.load(f) # 使用 QMIX_ALG_NAME
            self.qmix_text.delete(1.0, tk.END) # 修改变量名
            self.qmix_text.insert(1.0, json.dumps(self.results[QMIX_ALG_NAME], indent=2, ensure_ascii=False)) # 修改变量名
            self.update_qmix_visualization() # 修改函数名
            self.update_comparison()
            messagebox.showinfo("成功", f"QMIX 结果文件加载成功: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"加载 QMIX 文件失败: {str(e)}")

    def update_qmix_visualization(self): # 修改函数名和变量名
        frame = self.qmix_fig_frame # 修改变量名
        for widget in frame.winfo_children():
            widget.destroy()
        data = self.results.get(QMIX_ALG_NAME) # 使用 QMIX_ALG_NAME
        if not data:
            # 如果没有数据，显示提示信息
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, '无数据可显示\n请先运行 QMIX 推理或加载结果文件', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title("QMIX 结果")
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            return
        # (复用通用可视化逻辑)
        self._common_visualization(frame, data, "QMIX") # 修改标题前缀

    # --- QTRAN Functions (新增) ---
    def select_qtran_input_json(self): # 新增选择 QTRAN 输入文件
        file_path = filedialog.askopenfilename(
            title="选择 QTRAN 输入JSON文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.qtran_input_json = file_path
            filename = os.path.basename(file_path)
            self.qtran_input_label.config(text=f"输入文件: {filename}", foreground="green")

    def run_qtran_inference(self): # 新增运行 QTRAN 推理
        def run():
            try:
                if not self.qtran_input_json:
                    messagebox.showwarning("警告", "请先选择 QTRAN 的输入JSON文件")
                    return
                # 生成带时间戳的唯一输出文件名
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"qtran_inference_output_{timestamp}.json"
                output_dir = os.path.join(os.getcwd(), "outputs_json", "Qtran")
                os.makedirs(output_dir, exist_ok=True) # 确保目录存在
                output_path = os.path.join(output_dir, output_filename)
                cmd = [
                    sys.executable, 'infer/infer_qtrans.py', # 调用 QTRAN 脚本
                    '--config', self.qtran_input_json, # 使用 --config 参数
                    '--device', 'cpu' # 可根据需要添加其他参数
                ]
                self.log(f"运行 QTRAN 推理: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', cwd=os.getcwd())
                if result.returncode == 0:
                    self.log("QTRAN 推理完成！")
                    messagebox.showinfo("成功", f"QTRAN 推理完成！输出文件: {output_filename}")
                    # 自动加载生成的结果文件
                    self.find_and_load_latest_qtran_result()
                else:
                    error_msg = f"QTRAN 推理失败:\n{result.stderr}"
                    self.log(error_msg)
                    messagebox.showerror("错误", error_msg)
            except Exception as e:
                error_msg = f"运行 QTRAN 推理时出错: {str(e)}"
                self.log(error_msg)
                messagebox.showerror("错误", error_msg)
        if not os.path.exists('infer/infer_qtrans.py'): # 检查新脚本
            messagebox.showerror("错误", "找不到 QTRAN 推理脚本: infer/infer_qtrans.py")
            return
        threading.Thread(target=run, daemon=True).start()

    def find_and_load_latest_qtran_result(self): # 新增查找最新 QTRAN 结果
        try:
            # 查找 outputs_json/Qtran 目录下最新的 qtran_inference_output_*.json 文件
            base_dir = os.path.dirname(os.path.abspath(__file__))
            pattern = os.path.join(base_dir, "outputs_json", "Qtran", "qtran_inference_output_*.json")
            files = glob.glob(pattern)
            if files:
                latest = max(files, key=os.path.getctime)
                self.load_qtran_file(latest)
            else:
                messagebox.showinfo("信息", "未找到 QTRAN 结果文件")
        except Exception as e:
            messagebox.showwarning("警告", f"查找 QTRAN 结果文件时出错: {str(e)}")

    def load_qtran_results(self): # 新增加载 QTRAN 结果
        file_path = filedialog.askopenfilename(
            title="选择 QTRAN 结果文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.load_qtran_file(file_path)

    def load_qtran_file(self, file_path): # 新增加载 QTRAN 文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.results[QTRAN_ALG_NAME] = json.load(f) # 使用 QTRAN_ALG_NAME
            self.qtran_text.delete(1.0, tk.END)
            self.qtran_text.insert(1.0, json.dumps(self.results[QTRAN_ALG_NAME], indent=2, ensure_ascii=False))
            self.update_qtran_visualization() # 更新可视化
            self.update_comparison() # 更新对比
            messagebox.showinfo("成功", f"QTRAN 结果文件加载成功: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"加载 QTRAN 文件失败: {str(e)}")

    def update_qtran_visualization(self): # 新增更新 QTRAN 可视化
        frame = self.qtran_fig_frame
        for widget in frame.winfo_children():
            widget.destroy()
        data = self.results.get(QTRAN_ALG_NAME) # 使用 QTRAN_ALG_NAME
        if not data:
            # 如果没有数据，显示提示信息
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, '无数据可显示\n请先运行 QTRAN 推理或加载结果文件', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title("QTRAN 结果")
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            return
        # (复用通用可视化逻辑)
        self._common_visualization(frame, data, "QTRAN") # 修改标题前缀

    # --- Deterministic Algorithm Functions (New) ---
    def select_deterministic_input_json(self): # 新增选择确定性算法输入文件
        file_path = filedialog.askopenfilename(
            title="选择确定性算法输入JSON文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.deterministic_input_json = file_path
            filename = os.path.basename(file_path)
            self.deterministic_input_label.config(text=f"输入文件: {filename}", foreground="green")

    def run_deterministic_inference(self): # 新增运行确定性算法推理
        def run():
            try:
                if not self.deterministic_input_json:
                    messagebox.showwarning("警告", "请先选择确定性算法的输入JSON文件")
                    return
                # 生成带时间戳的唯一输出文件名
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"deterministic_output_{timestamp}.json"
                output_path = os.path.join(os.getcwd(), "outputs_json", "deterministic", output_filename) # 确保在当前工作目录
                cmd = [
                    sys.executable, 'infer/deterministic.py',
                    '--input_json', self.deterministic_input_json,
                    '--output_json', output_path
                ]
                self.log(f"运行确定性算法推理: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', cwd=os.getcwd())
                if result.returncode == 0:
                    self.log("确定性算法推理完成！")
                    messagebox.showinfo("成功", f"确定性算法推理完成！输出文件: {output_filename}")
                    # 自动加载生成的结果文件
                    self.load_deterministic_file(output_path)
                else:
                    error_msg = f"确定性算法推理失败:\n{result.stderr}"
                    self.log(error_msg)
                    messagebox.showerror("错误", error_msg)
            except Exception as e:
                error_msg = f"运行确定性算法推理时出错: {str(e)}"
                self.log(error_msg)
                messagebox.showerror("错误", error_msg)
        if not os.path.exists('infer/deterministic.py'):
            messagebox.showerror("错误", "找不到确定性算法推理脚本: infer/deterministic.py")
            return
        threading.Thread(target=run, daemon=True).start()

    def find_and_load_latest_deterministic_result(self): # 新增查找最新确定性算法结果 (可选调用)
         try:
             files = [f for f in os.listdir('.') if f.startswith('deterministic_output_') and f.endswith('.json')]
             if files:
                 latest = max(files, key=os.path.getctime)
                 self.load_deterministic_file(latest)
             else:
                 messagebox.showinfo("信息", "未找到确定性算法结果文件")
         except Exception as e:
             messagebox.showwarning("警告", f"查找确定性算法结果文件时出错: {str(e)}")

    def load_deterministic_results(self): # 新增加载确定性算法结果
        file_path = filedialog.askopenfilename(
            title="选择确定性算法结果文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.load_deterministic_file(file_path)

    def load_deterministic_file(self, file_path): # 新增加载确定性算法文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.results[DETERMINISTIC_ALG_NAME] = json.load(f)
            self.deterministic_text.delete(1.0, tk.END)
            self.deterministic_text.insert(1.0, json.dumps(self.results[DETERMINISTIC_ALG_NAME], indent=2, ensure_ascii=False))
            self.update_deterministic_visualization() # 更新可视化
            self.update_comparison() # 更新对比
            messagebox.showinfo("成功", f"确定性算法结果文件加载成功: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"加载确定性算法文件失败: {str(e)}")

    def update_deterministic_visualization(self): # 新增更新确定性算法可视化
        frame = self.deterministic_fig_frame
        for widget in frame.winfo_children():
            widget.destroy()
        data = self.results.get(DETERMINISTIC_ALG_NAME)
        if not data:
            # 如果没有数据，显示提示信息
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, '无数据可显示\n请先运行确定性算法推理或加载结果文件', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title("确定性算法结果")
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            return
        fig = Figure(figsize=(12, 10))
        if 'steps' in data and len(data['steps']) > 0:
            steps = [s['step'] for s in data['steps']]
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            # 奖励曲线
            rewards = [s.get('reward', 0) for s in data['steps']]
            cumulative_rewards = []
            cum_sum = 0
            for r in rewards:
                cum_sum += r
                cumulative_rewards.append(cum_sum)
            ax1.plot(steps, rewards, 'b-', label='单步奖励', linewidth=1.5)
            ax1.plot(steps, cumulative_rewards, 'r-', label='累计奖励', linewidth=1.5)
            ax1.set_title("确定性算法 奖励曲线")
            ax1.set_xlabel("步数")
            ax1.set_ylabel("奖励")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            # 无人机数量变化
            if 'friendly_remaining' in data['steps'][0]:
                friendly_interceptor = [s['friendly_remaining'].get('interceptor', 0) for s in data['steps']]
                friendly_recon = [s['friendly_remaining'].get('recon', 0) for s in data['steps']]
                enemy_ground_attack = [s['enemy_remaining'].get('ground_attack', 0) for s in data['steps']]
                enemy_recon = [s['enemy_remaining'].get('recon', 0) for s in data['steps']]
                ax2.plot(steps, friendly_interceptor, 'g-', label='友方拦截机', linewidth=1.5)
                ax2.plot(steps, friendly_recon, 'b-', label='友方侦察机', linewidth=1.5)
                ax2.set_title("友方无人机数量")
                ax2.set_xlabel("步数")
                ax2.set_ylabel("数量")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax3.plot(steps, enemy_ground_attack, 'r-', label='敌方攻击机', linewidth=1.5)
                ax3.plot(steps, enemy_recon, 'm-', label='敌方侦察机', linewidth=1.5)
                ax3.set_title("敌方无人机数量")
                ax3.set_xlabel("步数")
                ax3.set_ylabel("数量")
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, '无无人机数据', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title("友方无人机数量")
                ax3.text(0.5, 0.5, '无无人机数据', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title("敌方无人机数量")
            # 动作分布（如果存在）
            if 'action' in data['steps'][0]:
                interceptor_actions = []
                recon_actions = []
                escort_actions = []
                for s in data['steps']:
                    action = s.get('action', {})
                    interceptor_actions.append(
                        action.get('interceptor', {}).get('count', 0) if isinstance(action.get('interceptor', {}),
                                                                                    dict) else action.get('interceptor',
                                                                                                          0))
                    recon_actions.append(action.get('recon', {}).get('count', 0) if isinstance(action.get('recon', {}),
                                                                                               dict) else action.get(
                        'recon', 0))
                    escort_actions.append(
                        action.get('escort', {}).get('count', 0) if isinstance(action.get('escort', {}),
                                                                               dict) else action.get('escort', 0))
                x_pos = np.arange(len(steps))
                width = 0.25
                ax4.bar(x_pos - width, interceptor_actions, width, label='拦截机动作', alpha=0.8)
                ax4.bar(x_pos, recon_actions, width, label='侦察机动作', alpha=0.8)
                ax4.bar(x_pos + width, escort_actions, width, label='护航机动作', alpha=0.8)
                ax4.set_title("动作分布")
                ax4.set_xlabel("步数")
                ax4.set_ylabel("动作数量")
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, '无动作数据', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title("动作分布")
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, '无步骤数据', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title("确定性算法 结果")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- DT Functions ---
    def train_dt_model(self):
        """训练DT模型"""
        if self.is_training:
            messagebox.showwarning("警告", "训练正在进行中，请先停止当前训练")
            return
        if not os.path.exists('train/dt_uav.py'):
            messagebox.showerror("错误", "找不到DT训练脚本: train/dt_uav.py")
            return
        if not os.path.exists('datasets/uav_combat_dataset.hdf5'):
            messagebox.showwarning("警告", "默认数据集不存在，请确保数据集路径正确")
        self.log("开始训练DT模型...")
        self.log_status_label.config(text="训练中...", foreground="orange")
        self.is_training = True
        threading.Thread(target=self.run_training_thread, daemon=True).start()

    def run_training_thread(self):
        """训练线程"""
        try:
            cmd = [
                sys.executable, 'train/dt_uav.py',
                '--dataset', 'datasets/uav_combat_dataset.hdf5',
            ]
            self.log(f"执行命令: {' '.join(cmd)}")
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=False,
                cwd=os.getcwd()
            )
            while self.is_training:
                line = self.training_process.stdout.readline()
                if not line:
                    break
                try:
                    decoded_line = line.decode('utf-8', errors='ignore').strip()
                    if decoded_line:
                        self.log(decoded_line)
                except UnicodeDecodeError:
                    try:
                        decoded_line = line.decode('latin-1', errors='ignore').strip()
                        if decoded_line:
                            self.log(decoded_line)
                    except:
                        pass
            return_code = self.training_process.wait()
            self.training_process = None
            if return_code == 0:
                self.log("DT模型训练完成！")
                self.log_status_label.config(text="训练完成", foreground="green")
                messagebox.showinfo("成功", "DT模型训练完成！")
                self.load_latest_dt_model()
            else:
                error_msg = f"DT训练失败，返回码: {return_code}"
                self.log(error_msg)
                self.log_status_label.config(text="训练失败", foreground="red")
                messagebox.showerror("错误", error_msg)
        except Exception as e:
            error_msg = f"运行DT训练时出错: {str(e)}"
            self.log(error_msg)
            self.log_status_label.config(text="训练出错", foreground="red")
            messagebox.showerror("错误", error_msg)
        finally:
            self.is_training = False

    def train_model(self, alg):
        if self.is_training:
            messagebox.showwarning("警告", "训练正在进行中，请先停止当前训练")
            return
        train_script = f'infer/infer_mul_alg.py'
        if not os.path.exists(train_script):
            messagebox.showerror("错误", f"找不到训练脚本: {train_script}")
            return
        self.log(f"开始训练 {alg} 模型...")
        self.log_status_label.config(text="训练中...", foreground="orange")
        self.is_training = True
        def run():
            try:
                cmd = [sys.executable, train_script]
                cmd.extend(['--mode', "train"])
                cmd.extend(['--alg_name', alg])
                self.log(f"执行命令: {' '.join(cmd)}")
                self.training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=False,
                    cwd=os.getcwd()
                )
                while self.is_training:
                    line = self.training_process.stdout.readline()
                    if not line:
                        break
                    try:
                        decoded_line = line.decode('utf-8', errors='ignore').strip()
                        if decoded_line:
                            self.log(decoded_line)
                    except:
                        pass
                return_code = self.training_process.wait()
                self.training_process = None
                if return_code == 0:
                    self.log(f"{alg} 模型训练完成！")
                    self.log_status_label.config(text="训练完成", foreground="green")
                    messagebox.showinfo("成功", f"{alg} 模型训练完成！")
                    self.load_latest_model(alg)
                else:
                    self.log(f"{alg} 训练失败，返回码: {return_code}")
                    self.log_status_label.config(text="训练失败", foreground="red")
                    messagebox.showerror("错误", f"{alg} 训练失败")
            except Exception as e:
                self.log(f"运行 {alg} 训练时出错: {str(e)}")
                self.log_status_label.config(text="训练出错", foreground="red")
                messagebox.showerror("错误", f"运行 {alg} 训练时出错: {str(e)}")
            finally:
                self.is_training = False
        threading.Thread(target=run, daemon=True).start()

    def stop_training(self):
        if self.is_training and self.training_process:
            try:
                self.log("正在停止训练...")
                self.training_process.terminate()
                self.training_process.wait(timeout=5)
                self.log("训练已停止")
            except:
                try:
                    self.training_process.kill()
                    self.log("训练已被强制终止")
                except:
                    self.log("无法停止训练进程")
            finally:
                self.is_training = False
                self.training_process = None
                self.log_status_label.config(text="训练已停止", foreground="red")
        else:
            self.log("没有正在进行的训练")

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        self.log("日志已清空")

    def save_log(self):
        file_path = filedialog.asksaveasfilename(
            title="保存日志文件",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self.log(f"日志已保存到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存日志失败: {str(e)}")

    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def select_dt_input_json(self):
        file_path = filedialog.askopenfilename(title="选择DT输入JSON文件", filetypes=[("JSON files", "*.json")])
        if file_path:
            self.dt_input_json = file_path
            self.dt_input_label.config(text=f"输入文件: {os.path.basename(file_path)}", foreground="green")

    def select_dt_model(self):
        file_path = filedialog.askopenfilename(title="选择DT模型文件", filetypes=[("Model files", "*.pt")])
        if file_path:
            self.dt_model_path = file_path
            self.dt_model_label.config(text=f"模型: {os.path.basename(file_path)}", foreground="green")

    def load_latest_dt_model(self):
        try:
            files = glob.glob("save_models/dt_checkpoint.pt")
            if files:
                latest = max(files, key=os.path.getmtime)
                self.dt_model_path = latest
                self.dt_model_label.config(text=f"模型: {os.path.basename(latest)} (最新)", foreground="green")
                self.log("已加载最新DT模型")
            else:
                messagebox.showinfo("信息", "未找到DT模型文件")
        except Exception as e:
            self.log(f"加载DT模型时出错: {str(e)}")

    def run_dt_inference(self):
        def run():
            try:
                cmd = [sys.executable, 'infer/infer_dt_new.py']
                if self.dt_input_json:
                    cmd.extend(['--input_json', self.dt_input_json])
                if self.dt_model_path:
                    cmd.extend(['--checkpoint', self.dt_model_path])
                cmd.extend(['--target_return', '80.0', '--device', 'cpu', '--no-render'])
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    messagebox.showinfo("成功", "DT推理完成！")
                    self.find_and_load_latest_dt_result()
                else:
                    messagebox.showerror("错误", f"DT推理失败:\n{result.stderr}")
            except Exception as e:
                messagebox.showerror("错误", f"运行DT推理时出错: {str(e)}")
        threading.Thread(target=run, daemon=True).start()

    def find_and_load_latest_dt_result(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            pattern = os.path.join(base_dir, "outputs_json", "dt", "dt_episode_output_*.json")
            files = glob.glob(pattern)
            if files:
                latest = max(files, key=os.path.getctime)
                self.load_dt_file(latest)
            else:
                messagebox.showinfo("信息", "未找到DT结果文件")
        except Exception as e:
            messagebox.showwarning("警告", f"查找DT结果文件时出错: {str(e)}")

    def load_dt_results(self):
        file_path = filedialog.askopenfilename(title="选择DT结果文件", filetypes=[("JSON files", "*.json")])
        if file_path:
            self.load_dt_file(file_path)

    def load_dt_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.results['DT'] = json.load(f)
            self.dt_text.delete(1.0, tk.END)
            self.dt_text.insert(1.0, json.dumps(self.results['DT'], indent=2, ensure_ascii=False))
            self.update_dt_visualization()
            self.update_comparison()
            messagebox.showinfo("成功", f"DT结果文件加载成功: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"加载DT文件失败: {str(e)}")

    def update_dt_visualization(self):
        for widget in self.dt_fig_frame.winfo_children():
            widget.destroy()
        data = self.results.get('DT')
        if not data:
            return
        # (可视化代码与 update_alg_visualization 类似，为简洁起见，这里复用逻辑)
        self._common_visualization(self.dt_fig_frame, data, "DT")

    # --- Common Visualization Helper ---
    # --- Common Visualization Helper ---
    def _common_visualization(self, frame, data, title_prefix):
        for widget in frame.winfo_children():
            widget.destroy()

        if not data:
            # 如果没有数据，显示提示信息
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, '无数据可显示\n请先运行推理或加载结果文件', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{title_prefix} 结果")
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            return

        fig = Figure(figsize=(12, 10))
        if 'steps' in data and len(data['steps']) > 0:
            steps = [s['step'] for s in data['steps']]
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)

            # --- 修改：兼容 'reward' 和 'rewards' ---
            if isinstance(data['steps'][0].get('reward'), list):
                rewards = [sum(s.get('reward', [0])) for s in data['steps']]  # 如果是列表，求和
            elif isinstance(data['steps'][0].get('rewards'), list):  # 检查 'rewards' (复数)
                rewards = [sum(s.get('rewards', [0])) for s in data['steps']]  # 如果是列表，求和
            else:
                rewards = [s.get('reward', 0) for s in data['steps']]  # 原来的单值逻辑
            # --- 结束修改 ---

            cumulative_rewards = []
            cum_sum = 0
            for r in rewards:
                cum_sum += r
                cumulative_rewards.append(cum_sum)
            ax1.plot(steps, rewards, 'b-', label='单步奖励', linewidth=1.5)
            ax1.plot(steps, cumulative_rewards, 'r-', label='累计奖励', linewidth=1.5)
            ax1.set_title(f"{title_prefix} 奖励曲线")
            ax1.set_xlabel("步数")
            ax1.set_ylabel("奖励")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            # ... (其余代码保持不变) ...
            # (为了简洁，这里省略了未修改的部分，您保持原样即可)
            # ... (例如无人机数量、动作分布的绘制逻辑) ...
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, '无步骤数据', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f"{title_prefix} 结果")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    # --- Comparison Functions ---
    def load_compare_model1(self):
        """加载模型1的结果文件"""
        model1 = self.compare_model1.get()
        # 处理确定性算法的特殊情况
        if model1 == DETERMINISTIC_ALG_NAME:
             file_path = filedialog.askopenfilename(
                 title=f"选择 {model1} 结果文件",
                 filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
             )
             if file_path:
                 try:
                     with open(file_path, 'r', encoding='utf-8') as f:
                         self.results[model1] = json.load(f)
                     self.update_comparison()
                     messagebox.showinfo("成功", f"{model1} 结果文件加载成功: {os.path.basename(file_path)}")
                 except Exception as e:
                     messagebox.showerror("错误", f"加载 {model1} 文件失败: {str(e)}")
        else:
            file_path = filedialog.askopenfilename(
                title=f"选择 {model1} 结果文件",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if file_path:
                try:
                    # 复用现有的加载逻辑，但需要区分不同模型类型
                    if model1 in SUPPORTED_ALGS:
                        self.load_result_file(model1, file_path)
                    elif model1 == 'DT':
                        self.load_dt_file(file_path)
                    elif model1 == QMIX_ALG_NAME: # 修改判断条件
                        self.load_qmix_file(file_path) # 修改函数调用
                    elif model1 == QTRAN_ALG_NAME: # 新增判断条件
                        self.load_qtran_file(file_path) # 新增函数调用
                    # DETERMINISTIC handled above
                except Exception as e:
                     messagebox.showerror("错误", f"加载 {model1} 文件失败: {str(e)}")

    def load_compare_model2(self):
        """加载模型2的结果文件"""
        model2 = self.compare_model2.get()
        # 处理确定性算法的特殊情况
        if model2 == DETERMINISTIC_ALG_NAME:
             file_path = filedialog.askopenfilename(
                 title=f"选择 {model2} 结果文件",
                 filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
             )
             if file_path:
                 try:
                     with open(file_path, 'r', encoding='utf-8') as f:
                         self.results[model2] = json.load(f)
                     self.update_comparison()
                     messagebox.showinfo("成功", f"{model2} 结果文件加载成功: {os.path.basename(file_path)}")
                 except Exception as e:
                     messagebox.showerror("错误", f"加载 {model2} 文件失败: {str(e)}")
        else:
            file_path = filedialog.askopenfilename(
                title=f"选择 {model2} 结果文件",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if file_path:
                try:
                    if model2 in SUPPORTED_ALGS:
                        self.load_result_file(model2, file_path)
                    elif model2 == 'DT':
                        self.load_dt_file(file_path)
                    elif model2 == QMIX_ALG_NAME: # 修改判断条件
                        self.load_qmix_file(file_path) # 修改函数调用
                    elif model2 == QTRAN_ALG_NAME: # 新增判断条件
                        self.load_qtran_file(file_path) # 新增函数调用
                    # DETERMINISTIC handled above
                except Exception as e:
                     messagebox.showerror("错误", f"加载 {model2} 文件失败: {str(e)}")

    def update_comparison(self):
        for widget in self.compare_fig_frame.winfo_children():
            widget.destroy()
        self.compare_text.delete(1.0, tk.END)

        m1 = self.compare_model1.get()
        m2 = self.compare_model2.get()
        data1 = self.results.get(m1)
        data2 = self.results.get(m2)

        if not data1 or not data2:
            self.compare_text.insert(1.0, "请先加载两个模型的结果文件进行对比")
            return

        # --- 新增：判断是否为 QMIX vs QTRAN 或 QTRAN vs QMIX ---
        # 假设 QMIX_ALG_NAME = 'QMIX' 和 QTRAN_ALG_NAME = 'QTRAN' 已在类 __init__ 中定义
        is_qmix_qtran_comparison = (
                (m1 == getattr(self, 'QMIX_ALG_NAME', 'QMIX') and m2 == getattr(self, 'QTRAN_ALG_NAME', 'QTRAN')) or
                (m1 == getattr(self, 'QTRAN_ALG_NAME', 'QTRAN') and m2 == getattr(self, 'QMIX_ALG_NAME', 'QMIX'))
        )
        # --- 结束新增 ---

        # --- 修改：根据比较类型调整图表数量和布局 ---
        if is_qmix_qtran_comparison:
            # 简化版布局：1行2列
            fig = Figure(figsize=(14, 6))  # 调整 figsize
            ax1 = fig.add_subplot(121)  # 奖励对比
            ax2 = fig.add_subplot(122)  # 累计奖励对比
            # 增加子图间距
            fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, hspace=0.3, wspace=0.3)
        else:
            # 完整版布局：2行3列
            fig = Figure(figsize=(18, 12))  # 调整 figsize 使其更大
            ax1 = fig.add_subplot(231)
            ax2 = fig.add_subplot(232)
            ax3 = fig.add_subplot(233)
            ax4 = fig.add_subplot(234)
            ax5 = fig.add_subplot(235)
            ax6 = fig.add_subplot(236)
            # 增加子图间距
            fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.08, hspace=0.4, wspace=0.3)
        # --- 结束修改 ---

        # --- 通用数据准备 ---
        steps1 = [s['step'] for s in data1['steps']] if 'steps' in data1 and data1['steps'] else []
        steps2 = [s['step'] for s in data2['steps']] if 'steps' in data2 and data2['steps'] else []

        # 兼容 'reward' (单数) 和 'rewards' (复数/列表) 字段
        if 'steps' in data1 and data1['steps']:
            if isinstance(data1['steps'][0].get('reward'), list):
                rewards1 = [sum(s.get('reward', [0])) for s in data1['steps']]  # 如果是列表，求和
            elif isinstance(data1['steps'][0].get('rewards'), list):  # 检查 'rewards' (复数)
                rewards1 = [sum(s.get('rewards', [0])) for s in data1['steps']]  # 如果是列表，求和
            else:
                rewards1 = [s.get('reward', 0) for s in data1['steps']]  # 原来的单值逻辑
        else:
            rewards1 = []

        if 'steps' in data2 and data2['steps']:
            if isinstance(data2['steps'][0].get('reward'), list):
                rewards2 = [sum(s.get('reward', [0])) for s in data2['steps']]
            elif isinstance(data2['steps'][0].get('rewards'), list):
                rewards2 = [sum(s.get('rewards', [0])) for s in data2['steps']]
            else:
                rewards2 = [s.get('reward', 0) for s in data2['steps']]
        else:
            rewards2 = []

        cum_rewards1 = []
        cum_sum = 0
        for r in rewards1:
            cum_sum += r
            cum_rewards1.append(cum_sum)

        cum_rewards2 = []
        cum_sum = 0
        for r in rewards2:
            cum_sum += r
            cum_rewards2.append(cum_sum)
        # --- 结束通用数据准备 ---

        # --- 绘制通用图表 (奖励和累计奖励) ---
        ax1.plot(steps1, rewards1, 'b-', label=f'{m1} 奖励', linewidth=1.5)
        ax1.plot(steps2, rewards2, 'r-', label=f'{m2} 奖励', linewidth=1.5)
        ax1.set_title("奖励对比")
        ax1.set_xlabel("步数")
        ax1.set_ylabel("奖励")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps1, cum_rewards1, 'b-', label=f'{m1} 累计奖励', linewidth=1.5)
        ax2.plot(steps2, cum_rewards2, 'r-', label=f'{m2} 累计奖励', linewidth=1.5)
        ax2.set_title("累计奖励对比")
        ax2.set_xlabel("步数")
        ax2.set_ylabel("累计奖励")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # --- 结束绘制通用图表 ---

        # --- 绘制非 QMIX/QTRAN 特定图表 ---
        if not is_qmix_qtran_comparison:
            # 友方无人机总数对比
            friendly1_total = []
            friendly2_total = []
            if ('steps' in data1 and data1['steps'] and 'friendly_remaining' in data1['steps'][0] and
                    'steps' in data2 and data2['steps'] and 'friendly_remaining' in data2['steps'][0]):
                try:
                    friendly1_total = [
                        s['friendly_remaining'].get('interceptor', 0) +
                        s['friendly_remaining'].get('recon', 0) +
                        s['friendly_remaining'].get('escort', 0)  # 假设可能有escort
                        for s in data1['steps']
                    ]
                    friendly2_total = [
                        s['friendly_remaining'].get('interceptor', 0) +
                        s['friendly_remaining'].get('recon', 0) +
                        s['friendly_remaining'].get('escort', 0)
                        for s in data2['steps']
                    ]
                    ax3.plot(steps1, friendly1_total, 'b-', label=f'{m1} 友方总数', linewidth=1.5)
                    ax3.plot(steps2, friendly2_total, 'r-', label=f'{m2} 友方总数', linewidth=1.5)
                except (KeyError, TypeError):
                    ax3.text(0.5, 0.5, '数据格式错误', ha='center', va='center', transform=ax3.transAxes)
            else:
                ax3.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("友方无人机总数对比")
            ax3.set_xlabel("步数")
            ax3.set_ylabel("数量")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 敌方无人机总数对比
            enemy1_total = []
            enemy2_total = []
            if ('steps' in data1 and data1['steps'] and 'enemy_remaining' in data1['steps'][0] and
                    'steps' in data2 and data2['steps'] and 'enemy_remaining' in data2['steps'][0]):
                try:
                    enemy1_total = [
                        s['enemy_remaining'].get('ground_attack', 0) +
                        s['enemy_remaining'].get('recon', 0) +
                        # s['enemy_remaining'].get('air_attack', 0) # 如果有其他类型也加上
                        s['enemy_remaining'].get('other_type', 0)  # 占位符，根据实际字段调整
                        for s in data1['steps']
                    ]
                    # 注意：需要根据 data2 的实际字段调整
                    enemy2_total = [
                        s['enemy_remaining'].get('ground_attack', 0) +
                        s['enemy_remaining'].get('recon', 0) +
                        s['enemy_remaining'].get('other_type', 0)
                        for s in data2['steps']
                    ]
                    # 为了演示，我们假设 enemy2 的字段与 enemy1 相同或已调整
                    # 如果字段不同，需要分别处理
                    ax4.plot(steps1, enemy1_total, 'b-', label=f'{m1} 敌方总数', linewidth=1.5)
                    ax4.plot(steps2, enemy2_total, 'r-', label=f'{m2} 敌方总数', linewidth=1.5)
                except (KeyError, TypeError):
                    ax4.text(0.5, 0.5, '数据格式错误', ha='center', va='center', transform=ax4.transAxes)
            else:
                ax4.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("敌方无人机总数对比")
            ax4.set_xlabel("步数")
            ax4.set_ylabel("数量")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # 总体性能指标对比 (使用累计奖励和步数)
            steps_count1 = len(steps1)
            steps_count2 = len(steps2)
            metrics = ['总奖励', '步数']
            m1_values = [cum_rewards1[-1] if cum_rewards1 else 0, steps_count1]
            m2_values = [cum_rewards2[-1] if cum_rewards2 else 0, steps_count2]
            x_pos = np.arange(len(metrics))
            width = 0.35
            ax5.bar(x_pos - width / 2, m1_values, width, label=m1, alpha=0.8)
            ax5.bar(x_pos + width / 2, m2_values, width, label=m2, alpha=0.8)
            ax5.set_title("总体性能指标对比")
            ax5.set_xlabel("指标")
            ax5.set_ylabel("数值")
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(metrics)
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # 最终状态对比 (需要从 episode_info 提取，如果不存在则使用最后一步的数据)
            final_friendly1 = 0
            final_friendly2 = 0
            final_enemy1 = 0
            final_enemy2 = 0
            # 尝试从 episode_info 获取
            if 'episode_info' in data1:
                final_friendly1 = sum(data1['episode_info'].get('final_friendly_remaining', {}).values())
                final_enemy1 = sum(data1['episode_info'].get('final_enemy_remaining', {}).values())
            elif 'steps' in data1 and data1['steps']:
                # Fallback: 使用最后一步的 remaining 数据
                last_step1 = data1['steps'][-1]
                if 'friendly_remaining' in last_step1:
                    final_friendly1 = sum(last_step1['friendly_remaining'].values())
                if 'enemy_remaining' in last_step1:
                    final_enemy1 = sum(last_step1['enemy_remaining'].values())

            if 'episode_info' in data2:
                final_friendly2 = sum(data2['episode_info'].get('final_friendly_remaining', {}).values())
                final_enemy2 = sum(data2['episode_info'].get('final_enemy_remaining', {}).values())
            elif 'steps' in data2 and data2['steps']:
                last_step2 = data2['steps'][-1]
                if 'friendly_remaining' in last_step2:
                    final_friendly2 = sum(last_step2['friendly_remaining'].values())
                if 'enemy_remaining' in last_step2:
                    final_enemy2 = sum(last_step2['enemy_remaining'].values())

            final_metrics = ['友方剩余', '敌方剩余']
            final_m1_values = [final_friendly1, final_enemy1]
            final_m2_values = [final_friendly2, final_enemy2]
            x_pos_final = np.arange(len(final_metrics))
            width_final = 0.35
            ax6.bar(x_pos_final - width_final / 2, final_m1_values, width_final, label=m1, alpha=0.8)
            ax6.bar(x_pos_final + width_final / 2, final_m2_values, width_final, label=m2, alpha=0.8)
            ax6.set_title("最终状态对比")
            ax6.set_xlabel("状态")
            ax6.set_ylabel("数量")
            ax6.set_xticks(x_pos_final)
            ax6.set_xticklabels(final_metrics)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        # --- 结束绘制非 QMIX/QTRAN 特定图表 ---

        # --- 绘制文本比较 (保持不变或根据需要微调) ---
        text = f"=== {m1} vs {m2} 详细对比 ===\n"
        total_reward1 = cum_rewards1[-1] if cum_rewards1 else 0
        total_reward2 = cum_rewards2[-1] if cum_rewards2 else 0
        steps_count1 = len(steps1)
        steps_count2 = len(steps2)

        text += f"{m1} 总奖励: {total_reward1:.2f}\n"
        text += f"{m2} 总奖励: {total_reward2:.2f}\n"
        text += f"{m1} 总步数: {steps_count1}\n"
        text += f"{m2} 总步数: {steps_count2}\n"
        text += f"{m1} 最终友方剩余: {final_friendly1}\n"
        text += f"{m2} 最终友方剩余: {final_friendly2}\n"
        text += f"{m1} 最终敌方剩余: {final_enemy1}\n"
        text += f"{m2} 最终敌方剩余: {final_enemy2}\n"

        if total_reward1 > total_reward2:
            text += f"🏆 {m1} 在总奖励方面表现更好\n"
        elif total_reward2 > total_reward1:
            text += f"🏆 {m2} 在总奖励方面表现更好\n"
        else:
            text += "⚖️ 两者总奖励相同\n"

        # 对于 QMIX/QTRAN 比较，可以简化文本或添加提示
        if is_qmix_qtran_comparison:
            text += "\n(注意: QMIX 与 QTRAN 比较时，图表仅显示奖励相关数据)\n"
        else:
            # 其他比较的文本逻辑 (如果需要更详细的文本，可以在这里添加)
            if final_friendly1 > final_friendly2:
                text += f"🛡️ {m1} 保存了更多友方无人机\n"
            elif final_friendly2 > final_friendly1:
                text += f"🛡️ {m2} 保存了更多友方无人机\n"
            else:
                text += "⚖️ 两者保存的友方无人机数量相同\n"

            if final_enemy1 < final_enemy2:  # 敌方剩余少意味着消灭得多
                text += f"⚔️ {m1} 消灭了更多敌方无人机\n"
            elif final_enemy2 < final_enemy1:
                text += f"⚔️ {m2} 消灭了更多敌方无人机\n"
            else:
                text += "⚖️ 两者消灭的敌方无人机数量相同\n"
        # --- 结束绘制文本比较 ---

        # --- 更新 UI ---
        canvas = FigureCanvasTkAgg(fig, self.compare_fig_frame)
        canvas.draw()
        # 清除旧的画布部件（如果有的话，虽然上面已经destroy了frame children）
        # 但为了保险起见，可以再检查一次compare_fig_frame的子部件
        for child in self.compare_fig_frame.winfo_children():
            if isinstance(child, FigureCanvasTkAgg):
                child.get_tk_widget().destroy()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.compare_text.insert(1.0, text)
        # --- 结束更新 UI ---

def main():
    root = tk.Tk()
    app = UAVVisualizationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()