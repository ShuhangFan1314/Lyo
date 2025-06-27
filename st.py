import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sp
import base64
import math
from io import BytesIO

# 定义常量类
class Constant:
    kg_To_g = 1000.0
    cm_To_m = 1.0e-2
    hr_To_s = 3600.0
    hr_To_min = 60.0
    min_To_s = 60.0
    Torr_to_mTorr = 1000.0
    cal_To_J = 4.184
    rho_ice = 0.918  # g/mL
    rho_solute = 1.5  # g/mL
    rho_solution = 1.0  # g/mL
    dHs = 678.0  # Heat of sublimation in cal/g
    k_ice = 0.0059  # Thermal conductivity of ice in cal/cm/s/K
    dHf = 79.7  # Heat of fusion in cal/g
    Cp_ice = 2030.0  # Constant pressure specific heat of ice in J/kg/K
    Cp_solution = 4000.0  # Constant pressure specific heat of water in J/kg/K

# 函数类
class LyophilizationFunctions:
    @staticmethod
    def vapor_pressure(T_sub):
        """计算水蒸气压力 (Torr)"""
        return 2.698e10 * math.exp(-6144.96 / (273.15 + T_sub))
    
    @staticmethod
    def Lpr0_FUN(Vfill, Ap, cSolid):
        """计算初始填充高度 (cm)"""
        return Vfill / (Ap * Constant.rho_ice) * (
            Constant.rho_solution - cSolid * (Constant.rho_solution - Constant.rho_ice) / Constant.rho_solute
        )
    
    @staticmethod
    def Rp_FUN(l, R0, A1, A2):
        """计算产品阻力 (cm²-hr-Torr/g)"""
        return R0 + A1 * l / (1.0 + A2 * l)
    
    @staticmethod
    def Kv_FUN(KC, KP, KD, Pch):
        """计算小瓶热传递系数 (cal/s/K/cm²)"""
        return KC + KP * Pch / (1.0 + KD * Pch)
    
    @staticmethod
    def T_sub_solver_FUN(T_unknown, *data):
        """求解升华温度"""
        Pch, Av, Ap, Kv, Lpr0, Lck, Rp, Tsh = data
        P_sub = LyophilizationFunctions.vapor_pressure(T_unknown)
        return (P_sub - Pch) * (Av / Ap * Kv / Constant.k_ice * (Lpr0 - Lck) + 1) - \
               Av / Ap * Kv * Rp * Constant.hr_To_s / Constant.dHs * (Tsh - T_unknown)
    
    @staticmethod
    def sub_rate(Ap, Rp, T_sub, Pch):
        """计算升华速率 (kg/hr)"""
        P_sub = LyophilizationFunctions.vapor_pressure(T_sub)
        return Ap / Rp / Constant.kg_To_g * (P_sub - Pch)
    
    @staticmethod
    def T_bot_FUN(T_sub, Lpr0, Lck, Pch, Rp):
        """计算小瓶底部温度 (°C)"""
        P_sub = LyophilizationFunctions.vapor_pressure(T_sub)
        return T_sub + (Lpr0 - Lck) * (P_sub - Pch) * Constant.dHs / Rp / Constant.hr_To_s / Constant.k_ice

# 冻干曲线计算类
class KnownRpDryer:
    @staticmethod
    def dry(vial, product, ht, Pchamber, Tshelf, dt):
        """已知产品阻力的冻干过程模拟"""
        # 初始化
        Lpr0 = LyophilizationFunctions.Lpr0_FUN(vial['Vfill'], vial['Ap'], product['cSolid'])
        iStep = 0
        t = 0.0
        Lck = 0.0
        percent_dried = Lck / Lpr0 * 100.0
        
        # 初始化温度和压力
        Tsh = Tshelf['init']
        Tshelf['setpt'] = np.insert(Tshelf['setpt'], 0, Tshelf['init'])
        Tshelf['t_setpt'] = np.array([0])
        for dt_i in Tshelf['dt_setpt']:
            Tshelf['t_setpt'] = np.append(Tshelf['t_setpt'], Tshelf['t_setpt'][-1] + dt_i / Constant.hr_To_min)
        
        Pch = Pchamber['setpt'][0]
        Pchamber['setpt'] = np.insert(Pchamber['setpt'], 0, Pchamber['setpt'][0])
        Pchamber['t_setpt'] = np.array([0])
        for dt_j in Pchamber['dt_setpt']:
            Pchamber['t_setpt'] = np.append(Pchamber['t_setpt'], Pchamber['t_setpt'][-1] + dt_j / Constant.hr_To_min)
        
        T0 = Tsh
        
        # 主干燥过程
        while Lck <= Lpr0:
            Kv = LyophilizationFunctions.Kv_FUN(ht['KC'], ht['KP'], ht['KD'], Pch)
            Rp = LyophilizationFunctions.Rp_FUN(Lck, product['R0'], product['A1'], product['A2'])
            
            # 求解升华温度
            Tsub = sp.fsolve(
                LyophilizationFunctions.T_sub_solver_FUN, 
                T0, 
                args=(Pch, vial['Av'], vial['Ap'], Kv, Lpr0, Lck, Rp, Tsh)
            )[0]
            
            dmdt = LyophilizationFunctions.sub_rate(vial['Ap'], Rp, Tsub, Pch)
            if dmdt < 0:
                st.warning("Shelf temperature is too low for sublimation.")
                dmdt = 0.0
            
            Tbot = LyophilizationFunctions.T_bot_FUN(Tsub, Lpr0, Lck, Pch, Rp)
            
            # 计算升华长度
            dL = (dmdt * Constant.kg_To_g) * dt / (1 - product['cSolid'] * Constant.rho_solution / Constant.rho_solute) / \
                  (vial['Ap'] * Constant.rho_ice) * (1 - product['cSolid'] * (Constant.rho_solution - Constant.rho_ice) / Constant.rho_solute)
            
            # 保存输出
            if iStep == 0:
                output_saved = np.array([[t, float(Tsub), float(Tbot), Tsh, Pch * Constant.Torr_to_mTorr, 
                                          dmdt / (vial['Ap'] * Constant.cm_To_m**2), percent_dried]])
            else:
                output_saved = np.append(output_saved, [[t, float(Tsub), float(Tbot), Tsh, Pch * Constant.Torr_to_mTorr, 
                                                         dmdt / (vial['Ap'] * Constant.cm_To_m**2), percent_dried]], axis=0)
            
            # 更新计数器
            Lck_prev = Lck
            Lck = Lck + dL
            if Lck_prev < Lpr0 and Lck > Lpr0:
                Lck = Lpr0
                dL = Lck - Lck_prev
                t = iStep * dt + dL / ((dmdt * Constant.kg_To_g) / (1 - product['cSolid'] * Constant.rho_solution / Constant.rho_solute) / 
                                       (vial['Ap'] * Constant.rho_ice) * (1 - product['cSolid'] * (Constant.rho_solution - Constant.rho_ice) / Constant.rho_solute))
            else:
                t = (iStep + 1) * dt
                percent_dried = Lck / Lpr0 * 100.0
            
            # 更新温度和压力
            if len(np.where(Tshelf['t_setpt'] > t)[0]) == 0:
                st.warning("Total time exceeded. Drying incomplete")
                break
            else:
                i = np.where(Tshelf['t_setpt'] > t)[0][0]
                if Tshelf['setpt'][i] >= Tshelf['setpt'][i-1]:
                    Tsh = min(Tshelf['setpt'][i-1] + Tshelf['ramp_rate'] * Constant.hr_To_min * (t - Tshelf['t_setpt'][i-1]), Tshelf['setpt'][i])
                else:
                    Tsh = max(Tshelf['setpt'][i-1] - Tshelf['ramp_rate'] * Constant.hr_To_min * (t - Tshelf['t_setpt'][i-1]), Tshelf['setpt'][i])
                
                if len(np.where(Pchamber['t_setpt'] > t)[0]) == 0:
                    st.warning("Total time exceeded. Drying incomplete")
                    break
                else:
                    j = np.where(Pchamber['t_setpt'] > t)[0][0]
                    if Pchamber['setpt'][j] >= Pchamber['setpt'][j-1]:
                        Pch = min(Pchamber['setpt'][j-1] + Pchamber['ramp_rate'] * Constant.hr_To_min * (t - Pchamber['t_setpt'][j-1]), Pchamber['setpt'][j])
                    else:
                        Pch = max(Pchamber['setpt'][j-1] - Pchamber['ramp_rate'] * Constant.hr_To_min * (t - Pchamber['t_setpt'][j-1]), Pchamber['setpt'][j])
            
            iStep += 1
        
        Tshelf['setpt'] = Tshelf['setpt'][1:]
        return output_saved

# Streamlit应用主界面
def main():
    st.set_page_config(
        page_title="LyoPRONTO冻干曲线优化系统",
        page_icon="❄️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("❄️ LyoPRONTO冻干曲线优化系统")
    st.markdown("""
    **基于LyoPRONTO开源工具的生物制剂冻干工艺开发与优化平台**
    """)
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["参数设置", "冻干曲线模拟", "结果分析"])
    
    with tab1:
        # 创建带标签的侧边栏
        with st.sidebar:
            st.header("⚙️ 工艺参数设置")
            
            # 蛋白参数
            protein_conc = st.slider("蛋白浓度 (mg/mL)", 5.0, 100.0, 25.0, step=0.1)
            cSolid = st.slider("固含量 (%)", 0.1, 20.0, 5.0, step=0.1) / 100.0
            
            # 处方组成
            st.subheader("处方组成")
            excipients = {}
            col1, col2 = st.columns(2)
            with col1:
                excipients['蔗糖'] = st.number_input("蔗糖 (%)", 0.0, 20.0, 5.0, step=0.1)
                excipients['海藻糖'] = st.number_input("海藻糖 (%)", 0.0, 15.0, 0.0, step=0.1)
            with col2:
                excipients['甘露醇'] = st.number_input("甘露醇 (%)", 0.0, 10.0, 0.0, step=0.1)
                excipients['NaCl'] = st.number_input("NaCl (%)", 0.0, 5.0, 0.0, step=0.1)
            
            # 产品阻力参数
            st.subheader("产品阻力参数")
            R0 = st.number_input("R0 (cm²-hr-Torr/g)", 0.01, 1.0, 0.04, step=0.01)
            A1 = st.number_input("A1 (cm-hr-Torr/g)", 0.01, 10.0, 1.0, step=0.1)
            A2 = st.number_input("A2 (1/cm)", 0.01, 10.0, 0.1, step=0.01)
            T_pr_crit = st.number_input("临界产品温度 (°C)", -50.0, -10.0, -30.0, step=1.0)
            
            # 灌装参数
            st.subheader("灌装参数")
            fill_volume = st.number_input("灌装体积 (mL)", 1.0, 10.0, 3.0, step=0.1)
            vial_type = st.selectbox("西林瓶类型", ["模制瓶", "管制瓶"])
            vial_size = st.selectbox("西林瓶规格", ["2R", "6R", "10R"])
            
            # 计算小瓶参数
            vial_sizes = {"2R": {"diameter": 22.0, "height": 38.0}, 
                         "6R": {"diameter": 28.0, "height": 48.0},
                         "10R": {"diameter": 40.0, "height": 60.0}}
            
            vial_diameter = vial_sizes[vial_size]["diameter"] / 10.0  # mm to cm
            vial_area = 3.14 * (vial_diameter / 2) ** 2  # cm²
            vial_av = vial_area  # 简化假设
            
            # 计算灌装高度
            fill_depth = fill_volume / vial_area  # cm
            st.info(f"**计算灌装高度:** {fill_depth:.2f} cm")
            
            # 热传递参数
            st.subheader("热传递参数")
            col1, col2, col3 = st.columns(3)
            with col1:
                KC = st.number_input("KC (cal/s/K/cm²)", 0.0001, 0.01, 0.0005, step=0.0001, format="%.4f")
            with col2:
                KP = st.number_input("KP (cal/s/K/cm²/Torr)", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
            with col3:
                KD = st.number_input("KD (1/Torr)", 0.01, 10.0, 0.1, step=0.01)
            
            # 冻干机参数
            st.subheader("冻干机参数")
            condenser_temp = st.number_input("冷凝器温度 (°C)", -60.0, -30.0, -45.0, step=1.0)
            initial_temp = st.number_input("初始温度 (°C)", -10.0, 30.0, 20.0, step=1.0)
            final_temp = st.number_input("最终温度 (°C)", -60.0, -30.0, -40.0, step=1.0)
            cooling_rate = st.slider("冷却速率 (°C/min)", 0.1, 2.0, 0.5, step=0.1)
            
            # 干燥参数
            st.subheader("干燥参数")
            dt = st.slider("时间步长 (分钟)", 0.1, 10.0, 1.0, step=0.1) / 60.0  # 转换为小时
            
            # 压力设置
            st.subheader("压力设置")
            P_min = st.number_input("最小压力 (Torr)", 0.01, 1.0, 0.05, step=0.01)
            P_max = st.number_input("最大压力 (Torr)", 0.1, 5.0, 0.2, step=0.1)
            P_steps = st.slider("压力步数", 1, 10, 3)
            P_setpt = np.linspace(P_min, P_max, P_steps)
            P_ramp_rate = st.number_input("压力变化速率 (Torr/min)", 0.01, 1.0, 0.1, step=0.01)
            P_dt_setpt = [10.0] * (P_steps - 1)  # 简化设置
            
            # 温度设置
            st.subheader("温度设置")
            T_min = st.number_input("最小温度 (°C)", -50.0, -10.0, -40.0, step=1.0)
            T_max = st.number_input("最大温度 (°C)", -30.0, 10.0, -20.0, step=1.0)
            T_steps = st.slider("温度步数", 1, 10, 3)
            T_setpt = np.linspace(T_min, T_max, T_steps)
            T_ramp_rate = st.number_input("温度变化速率 (°C/min)", 0.1, 5.0, 1.0, step=0.1)
            T_dt_setpt = [20.0] * (T_steps - 1)  # 简化设置
            
            # 生成参数字典
            vial_params = {
                'Vfill': fill_volume,
                'Ap': vial_area,
                'Av': vial_av
            }
            
            product_params = {
                'cSolid': cSolid,
                'R0': R0,
                'A1': A1,
                'A2': A2,
                'T_pr_crit': T_pr_crit
            }
            
            ht_params = {
                'KC': KC,
                'KP': KP,
                'KD': KD
            }
            
            Pchamber_params = {
                'min': P_min,
                'setpt': P_setpt,
                'ramp_rate': P_ramp_rate,
                'dt_setpt': P_dt_setpt
            }
            
            Tshelf_params = {
                'init': initial_temp,
                'min': T_min,
                'max': T_max,
                'setpt': T_setpt,
                'ramp_rate': T_ramp_rate,
                'dt_setpt': T_dt_setpt
            }
            
            # 修改这里：移除了不支持的 type 参数
            if st.button("开始模拟"):
                st.session_state.vial = vial_params
                st.session_state.product = product_params
                st.session_state.ht = ht_params
                st.session_state.Pchamber = Pchamber_params
                st.session_state.Tshelf = Tshelf_params
                st.session_state.dt = dt
                st.session_state.simulation_done = False
                st.rerun()
    
    # 参数摘要显示在主区域
    if 'vial' in st.session_state:
        st.subheader("当前参数设置")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**小瓶参数**")
            st.text(f"灌装体积: {st.session_state.vial['Vfill']} mL")
            st.text(f"产品面积: {st.session_state.vial['Ap']:.4f} cm²")
            st.text(f"瓶底面积: {st.session_state.vial['Av']:.4f} cm²")
            
            st.markdown("**产品参数**")
            st.text(f"固含量: {st.session_state.product['cSolid']*100:.2f}%")
            st.text(f"R0: {st.session_state.product['R0']:.4f} cm²-hr-Torr/g")
            st.text(f"A1: {st.session_state.product['A1']:.4f} cm-hr-Torr/g")
            st.text(f"A2: {st.session_state.product['A2']:.4f} 1/cm")
            st.text(f"临界温度: {st.session_state.product['T_pr_crit']} °C")
        
        with col2:
            st.markdown("**热传递参数**")
            st.text(f"KC: {st.session_state.ht['KC']:.6f} cal/s/K/cm²")
            st.text(f"KP: {st.session_state.ht['KP']:.6f} cal/s/K/cm²/Torr")
            st.text(f"KD: {st.session_state.ht['KD']:.4f} 1/Torr")
            
            st.markdown("**压力设置**")
            st.text(f"压力范围: {st.session_state.Pchamber['min']:.3f} - {st.session_state.Pchamber['setpt'][-1]:.3f} Torr")
            st.text(f"压力变化速率: {st.session_state.Pchamber['ramp_rate']} Torr/min")
            
            st.markdown("**温度设置**")
            st.text(f"初始温度: {st.session_state.Tshelf['init']} °C")
            st.text(f"温度范围: {st.session_state.Tshelf['min']} - {st.session_state.Tshelf['max']} °C")
            st.text(f"温度变化速率: {st.session_state.Tshelf['ramp_rate']} °C/min")
    
    with tab2:
        if 'vial' in st.session_state and not st.session_state.get('simulation_done', False):
            with st.spinner("正在进行冻干曲线模拟..."):
                try:
                    # 运行模拟
                    output = KnownRpDryer.dry(
                        st.session_state.vial,
                        st.session_state.product,
                        st.session_state.ht,
                        st.session_state.Pchamber,
                        st.session_state.Tshelf,
                        st.session_state.dt
                    )
                    
                    # 保存结果
                    st.session_state.output = output
                    st.session_state.simulation_done = True
                except Exception as e:
                    st.error(f"模拟过程中出错: {str(e)}")
                    st.session_state.simulation_done = False
        
        if st.session_state.get('simulation_done', False):
            output = st.session_state.output
            st.success("冻干曲线模拟完成！")
            
            # 创建DataFrame
            df = pd.DataFrame(output, columns=[
                '时间 (小时)', '升华温度 (°C)', '瓶底温度 (°C)', 
                '板层温度 (°C)', '腔室压力 (mTorr)', '升华速率 (kg/hr/m²)', '干燥百分比 (%)'
            ])
            
            st.subheader("冻干曲线")
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # 温度曲线
            ax1.plot(df['时间 (小时)'], df['升华温度 (°C)'], 'b-', label='升华温度')
            ax1.plot(df['时间 (小时)'], df['瓶底温度 (°C)'], 'g-', label='瓶底温度')
            ax1.plot(df['时间 (小时)'], df['板层温度 (°C)'], 'r-', label='板层温度')
            ax1.set_xlabel('时间 (小时)')
            ax1.set_ylabel('温度 (°C)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.legend(loc='upper left')
            
            # 压力曲线
            ax2 = ax1.twinx()
            ax2.plot(df['时间 (小时)'], df['腔室压力 (mTorr)'], 'm--', label='腔室压力')
            ax2.set_ylabel('压力 (mTorr)', color='m')
            ax2.tick_params(axis='y', labelcolor='m')
            ax2.legend(loc='upper right')
            
            # 标注关键阶段
            dry_complete_idx = df[df['干燥百分比 (%)'] >= 99].index.min()
            if not np.isnan(dry_complete_idx):
                dry_complete_time = df.loc[dry_complete_idx, '时间 (小时)']
                ax1.axvline(x=dry_complete_time, color='k', linestyle='--', alpha=0.5)
                ax1.text(dry_complete_time, ax1.get_ylim()[0], '干燥完成', rotation=90, 
                         verticalalignment='bottom', horizontalalignment='right')
            
            plt.title('冻干曲线')
            plt.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            st.subheader("关键参数")
            col1, col2, col3 = st.columns(3)
            max_sub_temp = df['升华温度 (°C)'].max()
            max_bot_temp = df['瓶底温度 (°C)'].max()
            min_pressure = df['腔室压力 (mTorr)'].min()
            max_flux = df['升华速率 (kg/hr/m²)'].max()
            dry_time = df[df['干燥百分比 (%)'] >= 99]['时间 (小时)'].min()
            
            col1.metric("最高升华温度", f"{max_sub_temp:.1f} °C")
            col1.metric("最高瓶底温度", f"{max_bot_temp:.1f} °C")
            col2.metric("最低腔室压力", f"{min_pressure:.1f} mTorr")
            col2.metric("最大升华速率", f"{max_flux:.4f} kg/hr/m²")
            col3.metric("干燥完成时间", f"{dry_time:.1f} 小时" if not np.isnan(dry_time) else "未完成")
            col3.metric("最终干燥度", f"{df['干燥百分比 (%)'].iloc[-1]:.1f} %")
    
    with tab3:
        if st.session_state.get('simulation_done', False):
            df = pd.DataFrame(st.session_state.output, columns=[
                '时间 (小时)', '升华温度 (°C)', '瓶底温度 (°C)', 
                '板层温度 (°C)', '腔室压力 (mTorr)', '升华速率 (kg/hr/m²)', '干燥百分比 (%)'
            ])
            
            st.subheader("详细数据")
            st.dataframe(df.style.format({
                '时间 (小时)': '{:.2f}',
                '升华温度 (°C)': '{:.2f}',
                '瓶底温度 (°C)': '{:.2f}',
                '板层温度 (°C)': '{:.2f}',
                '腔室压力 (mTorr)': '{:.1f}',
                '升华速率 (kg/hr/m²)': '{:.4f}',
                '干燥百分比 (%)': '{:.1f}'
            }))
            
            # 数据下载
            st.subheader("数据导出")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下载CSV数据",
                data=csv,
                file_name='lyophilization_curve.csv',
                mime='text/csv'
            )
            
            # 生成报告
            st.subheader("工艺报告")
            report = f"""
            ### 冻干工艺模拟报告
            
            **产品参数**
            - 蛋白浓度: {protein_conc} mg/mL
            - 固含量: {cSolid*100:.1f}%
            - 产品阻力: R0={R0:.4f}, A1={A1:.4f}, A2={A2:.4f}
            
            **工艺参数**
            - 灌装体积: {fill_volume} mL
            - 灌装高度: {fill_depth:.2f} cm
            - 小瓶类型: {vial_type} ({vial_size})
            - 时间步长: {st.session_state.dt*60:.1f} 分钟
            
            **关键结果**
            - 干燥完成时间: {dry_time:.1f} 小时
            - 最高升华温度: {max_sub_temp:.1f} °C
            - 最高瓶底温度: {max_bot_temp:.1f} °C
            - 最大升华速率: {max_flux:.4f} kg/hr/m²
            - 最终干燥度: {df['干燥百分比 (%)'].iloc[-1]:.1f} %
            """
            
            st.markdown(report)
            
            # 参数优化建议
            st.subheader("优化建议")
            if max_bot_temp > st.session_state.product['T_pr_crit']:
                st.warning("⚠️ 瓶底温度超过临界产品温度，建议:")
                st.markdown("- 降低板层温度")
                st.markdown("- 提高腔室压力")
                st.markdown("- 缩短干燥时间")
            else:
                st.success("瓶底温度在安全范围内")
                
            if df['干燥百分比 (%)'].iloc[-1] < 99:
                st.warning("⚠️ 干燥未完全完成，建议:")
                st.markdown("- 延长干燥时间")
                st.markdown("- 提高板层温度")
                st.markdown("- 降低腔室压力")
            else:
                st.success("干燥完全完成")
                
            if max_flux > 1.0:  # 假设最大安全升华速率为1.0 kg/hr/m²
                st.warning("⚠️ 升华速率过高，可能影响产品质量，建议:")
                st.markdown("- 降低板层温度")
                st.markdown("- 提高腔室压力")
            else:
                st.success("升华速率在安全范围内")
        else:
            st.info("请先完成模拟以查看分析结果")

if __name__ == "__main__":
    main()
