import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sp
import math
import time
import base64
from io import BytesIO

# 导入LyoPRONTO核心模块
from src import constant
from src import freezing
from src import calc_knownRp
from src import calc_unknownRp
from src import design_space
from src import opt_Pch_Tsh
from src import opt_Pch
from src import opt_Tsh
from src import functions

# 设置页面配置
st.set_page_config(
    page_title="生物制剂冻干工艺优化平台",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 工具类型映射
TOOL_MAP = {
    "冷冻计算器": "Freezing Calculator",
    "初级干燥计算器": "Primary Drying Calculator",
    "设计空间生成器": "Design-Space-Generator",
    "优化器": "Optimizer"
}

# 西林瓶规格数据
VIAL_SIZES = {
    "2R": {"diameter": 16.0, "height": 22.0, "Wall_thickness": 1.0, "bottom_thickness":0.7}, 
    "6R": {"diameter": 22.0, "height": 26.0, "Wall_thickness": 1.0, "bottom_thickness":0.7},
    "10R": {"diameter": 24.0, "height": 30.0, "Wall_thickness": 1.0, "bottom_thickness":0.7},
    "20R": {"diameter": 30.0, "height": 35.0, "Wall_thickness": 1.2, "bottom_thickness":1.0}
}

# 格式化输出CSV
def format_csv(data):
    csv = data.to_csv(index=False).encode('utf-8')
    return csv

# 生成下载链接
def create_download_link(data, filename):
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">下载CSV文件</a>'
    return href

# 主应用
def main():
    st.title("❄️ 生物制剂冻干工艺优化平台")
    st.markdown("基于开源LyoPRONTO工具的冻干工艺开发与优化系统")
    
    # 创建标签页
    tabs = st.tabs(["参数设置", "模拟结果", "设计空间", "优化分析"])
    
    with tabs[0]:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("工具选择")
            tool = st.selectbox("选择工具", list(TOOL_MAP.keys()))
            sim_tool = TOOL_MAP[tool]
            
            kv_known = st.radio("Kv是否已知?", ["是", "否"], horizontal=True)
            rp_known = st.radio("Rp是否已知?", ["是", "否"], horizontal=True)
            
            if sim_tool == "Optimizer":
                var_pch = st.checkbox("可变腔室压力")
                var_tsh = st.checkbox("可变板层温度")
        
        with col2:
            st.subheader("通用参数")
            dt = st.slider("时间步长 (分钟)", 0.1, 10.0, 1.0) / 60.0  # 转换为小时
            
            # 小瓶参数
            st.markdown("**小瓶参数**")
            vial_size = st.selectbox("西林瓶规格", list(VIAL_SIZES.keys()))
            vial_diameter = VIAL_SIZES[vial_size]["diameter"] / 10.0  # mm to cm
            vial_thickness = VIAL_SIZES[vial_size]["Wall_thickness"] / 10.0  # mm to cm
            vial_area = 3.14 * ((vial_diameter - vial_thickness) / 2) ** 2  # cm²
            vial_av = vial_area  # 简化，后续需要继续补充该部分内容
            
            fill_volume = st.number_input("灌装体积 (mL)", 1.0, 20.0, 2.0, step=0.1)
            fill_depth = fill_volume / vial_area  # cm
            st.info(f"灌装高度: {fill_depth:.2f} cm")
            
            vial_params = {
                'Av': vial_av,
                'Ap': vial_area,
                'Vfill': fill_volume
            }
        
        # 产品参数
        st.subheader("产品参数")
        col1, col2 = st.columns(2)
        
        with col1:
            cSolid = st.slider("固含量 (%)", 0.1, 30.0, 5.0, step=0.1) / 100.0
            
            if sim_tool == "Freezing Calculator":
                Tpr0 = st.number_input("初始产品温度 (°C)", -10.0, 30.0, 20.0)
                Tf = st.number_input("冷冻温度 (°C)", -10.0, 0.0, -1.54)
                Tn = st.number_input("成核温度 (°C)", -10.0, 0.0, -5.84)
            else:
                if rp_known == "是":
                    R0 = st.number_input("R0 (cm²-hr-Torr/g)", 0.01, 10.0, 1.4, step=0.01)
                    A1 = st.number_input("A1 (cm-hr-Torr/g)", 0.01, 100.0, 16.0, step=0.1)
                    A2 = st.number_input("A2 (1/cm)", 0.01, 10.0, 0.0, step=0.01)
                
                if sim_tool != "Primary Drying Calculator":
                    T_pr_crit = st.number_input("临界产品温度 (°C)", -50.0, -10.0, -30.0)
        
        with col2:
            if sim_tool == "Freezing Calculator":
                h_freezing = st.number_input("热传递系数 (W/m²/K)", 10.0, 100.0, 38.0)
            elif kv_known == "是":
                st.markdown("**热传递参数**")
                KC = st.number_input("KC (cal/s/K/cm²)", 0.0001, 0.01, 0.0005, step=0.0001, format="%.4f")
                KP = st.number_input("KP (cal/s/K/cm²/Torr)", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
                KD = st.number_input("KD (1/Torr)", 0.01, 10.0, 0.46, step=0.01)
                ht_params = {'KC': KC, 'KP': KP, 'KD': KD}
        
        # 压力设置
        st.subheader("压力设置")
        if sim_tool == "Design-Space-Generator":
            p_min = st.number_input("最小压力 (Torr)", 0.01, 1.0, 0.05, step=0.01)
            p_max = st.number_input("最大压力 (Torr)", 0.1, 5.0, 1.5, step=0.1)
            p_steps = st.slider("压力点数", 2, 10, 4)
            p_setpt = list(np.linspace(p_min, p_max, p_steps))
        else:
            p_setpt = [st.number_input("腔室压力 (Torr)", 0.01, 5.0, 0.15, step=0.01)]
            p_ramp_rate = st.number_input("压力变化速率 (Torr/min)", 0.01, 1.0, 0.5, step=0.01)
        
        # 温度设置
        st.subheader("温度设置")
        if sim_tool == "Design-Space-Generator":
            t_init = st.number_input("初始温度 (°C)", -50.0, 20.0, -5.0)
            t_min = st.number_input("最小温度 (°C)", -50.0, -10.0, -5.0)
            t_max = st.number_input("最大温度 (°C)", -10.0, 20.0, 5.0)
            t_steps = st.slider("温度点数", 2, 10, 4)
            t_setpt = list(np.linspace(t_min, t_max, t_steps))
            t_ramp_rate = st.number_input("温度变化速率 (°C/min)", 0.1, 5.0, 1.0, step=0.1)
        else:
            t_init = st.number_input("初始温度 (°C)", -50.0, 20.0, -35.0)
            t_setpt = [st.number_input("板层温度 (°C)", -50.0, 20.0, 20.0)]
            t_ramp_rate = st.number_input("温度变化速率 (°C/min)", 0.1, 5.0, 1.0, step=0.1)
        
        # 设备参数
        if sim_tool != "Freezing Calculator" and sim_tool != "Primary Drying Calculator":
            st.subheader("设备参数")
            eq_a = st.number_input("设备参数 a (kg/hr)", -1.0, 1.0, -0.182, step=0.001)
            eq_b = st.number_input("设备参数 b (kg/hr/Torr)", 0.0, 0.1, 0.0117, step=0.0001)
            nVial = st.number_input("小瓶数量", 1, 1000, 398)
        
        # 保存参数到session
        params = {
            'sim_tool': sim_tool,
            'kv_known': kv_known,
            'rp_known': rp_known,
            'vial': vial_params,
            'dt': dt,
            'p_setpt': p_setpt,
            't_init': t_init,
            't_setpt': t_setpt,
            't_ramp_rate': t_ramp_rate
        }
        
        if sim_tool == "Freezing Calculator":
            params.update({
                'cSolid': cSolid,
                'Tpr0': Tpr0,
                'Tf': Tf,
                'Tn': Tn,
                'h_freezing': h_freezing
            })
        else:
            params.update({'cSolid': cSolid})
            
            if rp_known == "是":
                params.update({
                    'R0': R0,
                    'A1': A1,
                    'A2': A2
                })
            
            if sim_tool != "Primary Drying Calculator":
                params.update({'T_pr_crit': T_pr_crit})
            
            if kv_known == "是":
                params.update({'ht': ht_params})
            
            if sim_tool != "Freezing Calculator" and sim_tool != "Primary Drying Calculator":
                params.update({
                    'eq_cap': {'a': eq_a, 'b': eq_b},
                    'nVial': nVial
                })
        
        if st.button("开始模拟", type="primary", use_container_width=True):
            st.session_state.params = params
            st.session_state.simulation_done = False
            st.rerun()
    
    # 模拟执行和结果显示
    if 'params' in st.session_state:
        params = st.session_state.params
        
        with tabs[1]:
            if not st.session_state.get('simulation_done', False):
                with st.spinner("正在进行冻干工艺模拟..."):
                    try:
                        # 根据选择的工具执行不同的模拟
                        if params['sim_tool'] == "Freezing Calculator":
                            Tshelf_params = {
                                'init': params['t_init'],
                                'setpt': params['t_setpt'],
                                'ramp_rate': params['t_ramp_rate']
                            }
                            
                            output = freezing.freeze(
                                params['vial'],
                                {
                                    'cSolid': params['cSolid'],
                                    'Tpr0': params['Tpr0'],
                                    'Tf': params['Tf'],
                                    'Tn': params['Tn']
                                },
                                params['h_freezing'],
                                Tshelf_params,
                                params['dt']
                            )
                            
                            df = pd.DataFrame(output, columns=['时间 (小时)', '板层温度 (°C)', '产品温度 (°C)'])
                            st.session_state.output = df
                        
                        elif params['sim_tool'] == "Primary Drying Calculator":
                            Pchamber_params = {
                                'setpt': params['p_setpt'],
                                'ramp_rate': 0.5,
                                'dt_setpt': [1800.0]
                            }
                            
                            Tshelf_params = {
                                'init': params['t_init'],
                                'setpt': params['t_setpt'],
                                'dt_setpt': [1800.0],
                                'ramp_rate': params['t_ramp_rate']
                            }
                            
                            if params['rp_known'] == "是":
                                output = calc_knownRp.dry(
                                    params['vial'],
                                    {
                                        'cSolid': params['cSolid'],
                                        'R0': params['R0'],
                                        'A1': params['A1'],
                                        'A2': params['A2']
                                    },
                                    params['ht'],
                                    Pchamber_params,
                                    Tshelf_params,
                                    params['dt']
                                )
                                
                                df = pd.DataFrame(output, columns=[
                                    '时间 (小时)', '升华温度 (°C)', '瓶底温度 (°C)', 
                                    '板层温度 (°C)', '腔室压力 (mTorr)', '升华速率 (kg/hr/m²)', '干燥百分比 (%)'
                                ])
                                st.session_state.output = df
                        
                        st.session_state.simulation_done = True
                        st.success("模拟完成！")
                    except Exception as e:
                        st.error(f"模拟过程中出错: {str(e)}")
                        st.session_state.simulation_done = False
            
            if st.session_state.get('simulation_done', False):
                df = st.session_state.output
                
                st.subheader("模拟结果")
                st.dataframe(df.style.format("{:.2f}"), height=300)
                
                # 数据下载
                st.subheader("数据导出")
                csv_data = format_csv(df)
                st.markdown(create_download_link(csv_data, "lyo_simulation.csv"), unsafe_allow_html=True)
                
                # 可视化
                st.subheader("温度曲线")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if params['sim_tool'] == "Freezing Calculator":
                    ax.plot(df['时间 (小时)'], df['板层温度 (°C)'], 'r-', label='板层温度')
                    ax.plot(df['时间 (小时)'], df['产品温度 (°C)'], 'b-', label='产品温度')
                    ax.set_ylabel("温度 (°C)")
                else:
                    ax.plot(df['时间 (小时)'], df['升华温度 (°C)'], 'g-', label='升华温度')
                    ax.plot(df['时间 (小时)'], df['瓶底温度 (°C)'], 'b-', label='瓶底温度')
                    ax.plot(df['时间 (小时)'], df['板层温度 (°C)'], 'r-', label='板层温度')
                    ax.set_ylabel("温度 (°C)")
                
                ax.set_xlabel("时间 (小时)")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
                
                if params['sim_tool'] != "Freezing Calculator":
                    st.subheader("干燥进度")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df['时间 (小时)'], df['干燥百分比 (%)'], 'm-', label='干燥百分比')
                    ax.set_xlabel("时间 (小时)")
                    ax.set_ylabel("干燥百分比 (%)")
                    ax.grid(True)
                    ax.set_ylim([0, 100])
                    st.pyplot(fig)
        
        # 设计空间分析
        with tabs[2]:
            if params['sim_tool'] == "Design-Space-Generator" and st.session_state.get('simulation_done', False):
                st.subheader("设计空间分析")
                
                # 执行设计空间分析
                Pchamber_params = {
                    'setpt': params['p_setpt'],
                    'ramp_rate': 0.5
                }
                
                Tshelf_params = {
                    'init': params['t_init'],
                    'setpt': params['t_setpt'],
                    'ramp_rate': params['t_ramp_rate']
                }
                
                DS_shelf, DS_pr, DS_eq_cap = design_space.dry(
                    params['vial'],
                    {
                        'cSolid': params['cSolid'],
                        'R0': params['R0'],
                        'A1': params['A1'],
                        'A2': params['A2'],
                        'T_pr_crit': params['T_pr_crit']
                    },
                    params['ht'],
                    Pchamber_params,
                    Tshelf_params,
                    params['dt'],
                    params['eq_cap'],
                    params['nVial']
                )
                
                # 可视化设计空间
                st.subheader("升华速率设计空间")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # 压力转换为mTorr
                p_mtorr = [p * constant.Torr_to_mTorr for p in params['p_setpt']]
                
                # 设备能力曲线
                ax.plot(p_mtorr, DS_eq_cap[2], 'k-o', label="设备能力")
                
                # 产品温度限制曲线
                ax.plot([p_mtorr[0], p_mtorr[-1]], [DS_pr[2][0], DS_pr[2][1]], 'r-o', 
                        label=f"T_pr = {params['T_pr_crit']}°C")
                
                # 不同板层温度下的曲线
                for i, temp in enumerate(params['t_setpt']):
                    ax.plot(p_mtorr, DS_shelf[2][i], '--', 
                            label=f"T_sh = {temp}°C")
                
                ax.set_xlabel("腔室压力 (mTorr)")
                ax.set_ylabel("升华速率 (kg/hr/m²)")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
                
                st.subheader("干燥时间设计空间")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(p_mtorr, DS_eq_cap[1], 'k-o', label="设备能力")
                ax.plot([p_mtorr[0], p_mtorr[-1]], [DS_pr[1][0], DS_pr[1][1]], 'r-o', 
                        label=f"T_pr = {params['T_pr_crit']}°C")
                
                for i, temp in enumerate(params['t_setpt']):
                    ax.plot(p_mtorr, DS_shelf[1][i], '--', 
                            label=f"T_sh = {temp}°C")
                
                ax.set_xlabel("腔室压力 (mTorr)")
                ax.set_ylabel("干燥时间 (小时)")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
        
        # 优化分析
        with tabs[3]:
            if params['sim_tool'] == "Optimizer" and st.session_state.get('simulation_done', False):
                st.subheader("工艺优化分析")
                
                Pchamber_params = {
                    'min': min(params['p_setpt']),
                    'max': max(params['p_setpt'])
                }
                
                Tshelf_params = {
                    'min': min(params['t_setpt']),
                    'max': max(params['t_setpt'])
                }
                
                # 根据优化类型执行不同的优化
                if var_pch and var_tsh:
                    output = opt_Pch_Tsh.dry(
                        params['vial'],
                        {
                            'cSolid': params['cSolid'],
                            'R0': params['R0'],
                            'A1': params['A1'],
                            'A2': params['A2'],
                            'T_pr_crit': params['T_pr_crit']
                        },
                        params['ht'],
                        Pchamber_params,
                        Tshelf_params,
                        params['dt'],
                        params['eq_cap'],
                        params['nVial']
                    )
                elif var_pch:
                    output = opt_Pch.dry(
                        params['vial'],
                        {
                            'cSolid': params['cSolid'],
                            'R0': params['R0'],
                            'A1': params['A1'],
                            'A2': params['A2'],
                            'T_pr_crit': params['T_pr_crit']
                        },
                        params['ht'],
                        Pchamber_params,
                        Tshelf_params,
                        params['dt'],
                        params['eq_cap'],
                        params['nVial']
                    )
                elif var_tsh:
                    output = opt_Tsh.dry(
                        params['vial'],
                        {
                            'cSolid': params['cSolid'],
                            'R0': params['R0'],
                            'A1': params['A1'],
                            'A2': params['A2'],
                            'T_pr_crit': params['T_pr_crit']
                        },
                        params['ht'],
                        Pchamber_params,
                        Tshelf_params,
                        params['dt'],
                        params['eq_cap'],
                        params['nVial']
                    )
                
                df = pd.DataFrame(output, columns=[
                    '时间 (小时)', '升华温度 (°C)', '瓶底温度 (°C)', 
                    '板层温度 (°C)', '腔室压力 (mTorr)', '升华速率 (kg/hr/m²)', '干燥百分比 (%)'
                ])
                
                st.subheader("优化结果")
                st.dataframe(df.style.format("{:.2f}"), height=300)
                
                # 可视化优化结果
                st.subheader("温度曲线")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df['时间 (小时)'], df['升华温度 (°C)'], 'g-', label='升华温度')
                ax.plot(df['时间 (小时)'], df['瓶底温度 (°C)'], 'b-', label='瓶底温度')
                ax.plot(df['时间 (小时)'], df['板层温度 (°C)'], 'r-', label='板层温度')
                ax.set_xlabel("时间 (小时)")
                ax.set_ylabel("温度 (°C)")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
                
                st.subheader("干燥进度")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df['时间 (小时)'], df['干燥百分比 (%)'], 'm-', label='干燥百分比')
                ax.set_xlabel("时间 (小时)")
                ax.set_ylabel("干燥百分比 (%)")
                ax.grid(True)
                ax.set_ylim([0, 100])
                st.pyplot(fig)
                
                # 优化建议
                st.subheader("优化建议")
                dry_time = df['时间 (小时)'].iloc[-1]
                max_temp = df['瓶底温度 (°C)'].max()
                
                if max_temp > params['T_pr_crit']:
                    st.warning("⚠️ 瓶底温度超过临界产品温度，建议:")
                    st.markdown("- 降低板层温度")
                    st.markdown("- 提高腔室压力")
                    st.markdown("- 缩短干燥时间")
                else:
                    st.success("瓶底温度在安全范围内")
                
                if dry_time > 24:
                    st.warning("⚠️ 干燥时间过长，建议:")
                    st.markdown("- 提高板层温度")
                    st.markdown("- 降低腔室压力")
                    st.markdown("- 优化产品配方")
                else:
                    st.success("干燥时间在合理范围内")

if __name__ == "__main__":
    main()
