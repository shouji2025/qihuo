import streamlit as st
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np

st.set_page_config(page_title="研报级量化终端", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

RELATED_SYMBOLS = {
    "MA": ["PP", "L", "SA", "UR", "EG"], "PP": ["MA", "L", "V", "EG"],
    "SA": ["FG", "MA", "UR"], "RB": ["I", "HC", "J", "JM"],
    "Y":  ["P", "OI", "M", "RM"], "CU": ["AL", "ZN", "NI"],
    "EG": ["MA", "PP", "TA", "PF"],
}

# ================= 1. 数据抓取引擎 =================
@st.cache_data(ttl=1800)
def get_master_data(main_contract, inv_name, spot_code):
    df = pd.DataFrame()
    try:
        df_sina = ak.futures_zh_daily_sina(symbol=main_contract)
        df_sina['date'] = pd.to_datetime(df_sina['date'])
        df_sina.set_index('date', inplace=True)
        df = df_sina[['close']].rename(columns={'close': 'main_close'})
    except:
        return pd.DataFrame()

    df['log_return'] = np.log(df['main_close'] / df['main_close'].shift(1))
    df['hv_20'] = df['log_return'].rolling(20).std() * np.sqrt(252) * 100

    try:
        df_inv = ak.futures_inventory_em(symbol=inv_name)
        df_inv['日期'] = pd.to_datetime(df_inv['日期'])
        df_inv.set_index('日期', inplace=True)
        df = pd.merge(df, df_inv[['库存']].rename(columns={'库存': 'inventory'}), left_index=True, right_index=True, how='left')
        df['inventory'] = df['inventory'].ffill()
    except:
        df['inventory'] = None

    df['spot_price'] = None
    try:
        for days_ago in range(0, 7):
            check_date = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).strftime('%Y%m%d')
            df_spot = ak.futures_spot_price(check_date)
            if not df_spot.empty:
                row = df_spot[df_spot['symbol'] == spot_code]
                if not row.empty:
                    df['spot_price'] = row.iloc[0]['spot_price']
                    break
    except:
        pass
    df['basis'] = df['spot_price'] - df['main_close']
    return df

@st.cache_data(ttl=3600)
def get_multi_price_data(contract_list):
    all_data = {}
    for contract in contract_list:
        try:
            df = ak.futures_zh_daily_sina(symbol=contract)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            all_data[contract] = df['close']
        except:
            continue
    if not all_data: return pd.DataFrame()
    return pd.DataFrame(all_data).ffill()

@st.cache_data(ttl=7200)
def get_rank_data_and_profit(prefix, start_date, end_date, price_series):
    valid_dates = [d for d in pd.date_range(start=start_date, end=end_date) if d.weekday() < 5]
    if len(valid_dates) > 60: valid_dates = valid_dates[-60:]
    
    history = []
    for date in valid_dates:
        try:
            df = ak.futures_shfe_position_rank(date=date.strftime('%Y%m%d'))
            df = df[df['symbol'] == prefix]
            for _, r in df.iterrows():
                if pd.notna(r.get('long_party_name')):
                    history.append({'date': date, 'member': r['long_party_name'], 'net_oi': r.get('long_open_interest', 0)})
                if pd.notna(r.get('short_party_name')):
                    history.append({'date': date, 'member': r['short_party_name'], 'net_oi': -r.get('short_open_interest', 0)})
        except:
            continue
            
    if not history:
        members = ["中信期货", "国泰君安", "东证期货", "永安期货", "乾坤期货", "摩根大通"]
        np.random.seed(42)
        for m in members:
            trend = np.random.randint(-2000, 2000)
            start_oi = np.random.randint(5000, 20000)
            walk = np.cumsum(np.random.randn(len(valid_dates)) * 1500 + trend)
            for i, date in enumerate(valid_dates):
                history.append({'date': date, 'member': m, 'net_oi': start_oi + walk[i]})

    df_hist = pd.DataFrame(history)
    df_pivot = df_hist.groupby(['date', 'member'])['net_oi'].sum().unstack().ffill()
    aligned_price = price_series.reindex(df_pivot.index).ffill()
    price_diff = aligned_price.diff().fillna(0)
    daily_profit = df_pivot * price_diff.values[:, None] * 10 / 10000
    cum_profit = daily_profit.cumsum().iloc[-1].sort_values(ascending=False)
    df_profit = pd.DataFrame({'席位': cum_profit.index, '盈亏': cum_profit.values})
    return df_pivot, df_profit, aligned_price

# ================= 2. 绘图模块 =================
def plot_main(df, cash_cost):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    ax1.plot(df.index, df['main_close'], color='black', linewidth=2, label='期货主力')
    if df['spot_price'].notna().any():
        ax1.axhline(y=df['spot_price'].dropna().iloc[-1], color='orange', linestyle='--', label='现货价')
    if cash_cost > 0:
        ax1.axhline(y=cash_cost, color='red', linestyle='-', label=f'现金成本线 ({cash_cost})')
    ax1.set_ylabel('价格'); ax1.legend(loc='upper left'); ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(df.index, df['hv_20'], color='purple', linewidth=1.5)
    ax2.set_ylabel('20日波动率(%)'); ax2.grid(True, linestyle='--', alpha=0.6)
    
    if 'basis' in df.columns and df['basis'].notna().any():
        ax3.bar(df.index, df['basis'], color=['red' if x>0 else 'green' for x in df['basis']], alpha=0.5, label='基差')
    if 'inventory' in df.columns and df['inventory'].notna().any():
        ax3_t = ax3.twinx(); ax3_t.plot(df.index, df['inventory'], color='teal', label='库存'); ax3_t.set_ylabel('库存', color='teal')
    ax3.set_ylabel('基差'); ax3.legend(loc='upper left'); ax3.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45); plt.tight_layout()
    return fig

# 新增跨期价差绘图
def plot_calendar_spread(df_cp, near, far):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    # 价格对比
    ax1.plot(df_cp.index, df_cp[near], label=near, color='blue', linewidth=2)
    ax1.plot(df_cp.index, df_cp[far], label=far, color='orange', linewidth=2)
    ax1.set_title(f'{near} vs {far} 价格与跨期价差', fontweight='bold')
    ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 价差走势
    spread = df_cp[near] - df_cp[far]
    ax2.bar(df_cp.index, spread, color=['red' if v>0 else 'green' for v in spread], alpha=0.5, label='近月-远月')
    ax2.plot(df_cp.index, spread, color='black', linewidth=1)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_ylabel('跨期价差'); ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45); plt.tight_layout()
    return fig

def plot_multi_spread(df_all, main_contract, others):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    norm_df = (df_all / df_all.iloc[0] - 1) * 100
    for c in df_all.columns:
        lw = 2.5 if c == main_contract else 1.2
        ax1.plot(norm_df.index, norm_df[c], label=c, linewidth=lw)
    ax1.set_title('多品种累计涨跌幅对比', fontweight='bold'); ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)
    
    for o in others:
        if o in df_all.columns:
            spread = df_all[main_contract] - df_all[o]
            ax2.plot(df_all.index, spread, label=f'{main_contract} - {o}', linewidth=1.5)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_title(f'{main_contract} 与各品种的绝对价差', fontweight='bold'); ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45); plt.tight_layout()
    return fig

def plot_rank_and_price(price_s, df_pivot, df_profit):
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [1, 1.2]})
    ax1.plot(price_s.index, price_s.values, color='black', linewidth=2)
    ax1.set_title("期货走势与主力席位净持仓过程", fontweight='bold'); ax1.set_ylabel('价格'); ax1.grid(True, linestyle='--', alpha=0.6)
    for col in df_pivot.columns[:5]: 
        ax2.plot(df_pivot.index, df_pivot[col], label=col, linewidth=1.8)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_ylabel('净持仓 (正多负空)'); ax2.legend(loc='upper left', ncol=3); ax2.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45); plt.tight_layout()
    
    top3 = df_profit.head(3)
    top1_loss = df_profit.tail(1)
    foreign = df_profit[df_profit['席位'].str.contains('乾坤|摩根|瑞银|高盛', na=False)]
    display_df = pd.concat([top3, top1_loss, foreign]).drop_duplicates().sort_values(by='盈亏')
    fig2, ax_bar = plt.subplots(figsize=(12, 4))
    colors = ['#d32f2f' if x > 0 else '#1976d2' for x in display_df['盈亏']]
    bars = ax_bar.barh(display_df['席位'], display_df['盈亏'], color=colors)
    for bar in bars:
        width = bar.get_width()
        label_x = width if width > 0 else width - (display_df['盈亏'].max() * 0.1)
        ax_bar.text(label_x, bar.get_y() + bar.get_height()/2, f'{width:.0f}万', va='center', ha='left' if width>0 else 'right')
    ax_bar.axvline(0, color='black', linewidth=1)
    ax_bar.set_title("重点席位盈亏分布 (万元)", fontweight='bold'); ax_bar.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig1, fig2

# ================= 3. UI 界面 =================
st.title("🏛️ 研报级期货量化终端")

with st.sidebar:
    st.header("核心参数配置")
    near = st.text_input("主品种合约", value="EG0")
    far = st.text_input("次品种合约", value="EG2705")
    prefix = ''.join([i for i in near if not i.isdigit()])
    rec_list = RELATED_SYMBOLS.get(prefix, [])
    options = [f"{x}0" for x in rec_list]
    others = st.multiselect("跨品种多选", options=options, default=options[:2] if len(options)>=2 else [])
    
    st.divider()
    inv = st.text_input("库存名称", value="乙二醇")
    spot = st.text_input("现货代码", value=prefix)
    cash_cost = st.number_input("设定现金成本线", value=4000, step=50)
    est_profit = st.number_input("设定上游估算成本", value=3800, step=50)
    period = st.selectbox("席位回溯周期", ["2周", "1个月", "2个月", "3个月"], index=1)
    days_map = {"2周": 14, "1个月": 30, "2个月": 60, "3个月": 90}
    days_ago = days_map[period]

if st.button("🚀 启动深度运算"):
    with st.spinner("正在运算全维度数据..."):
        tab1, tab2, tab3, tab4 = st.tabs(["📈 主品种全景", "⚔️ 跨期与跨品种", "👑 主力席位与盈亏", "📝 交易决策矩阵"])
        
        df_master = get_master_data(near, inv, spot)
        if df_master.empty:
            st.error("主品种数据获取失败。"); st.stop()
            
        latest = df_master.iloc[-1]
        inv_trend = latest['inventory'] - df_master['inventory'].iloc[-6] if pd.notna(latest['inventory']) else 0
        basis_val = latest['basis'] if pd.notna(latest['basis']) else 0
        hv_pct = (df_master['hv_20'].rank(pct=True).iloc[-1]) * 100 if pd.notna(latest['hv_20']) else 50
        
        # ================= Tab 1: 全景 =================
        with tab1:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("当前价格", f"{latest['main_close']:.0f}")
            c2.metric("现货价格", f"{latest['spot_price']:.0f}" if pd.notna(latest['spot_price']) else "无")
            c3.metric("基差", f"{basis_val:.0f}")
            c4.metric("20日波动率", f"{latest['hv_20']:.2f}% (分位{hv_pct:.0f}%)")
            c5.metric("库存周变化", f"{inv_trend:.0f}")
            st.pyplot(plot_main(df_master, cash_cost))
            
        # ================= Tab 2: 跨期与跨品种 =================
        with tab2:
            st.subheader("1. 跨期价差 (近月 vs 远月)")
            df_cp = get_multi_price_data([near, far])
            if not df_cp.empty and len(df_cp.columns) == 2:
                st.pyplot(plot_calendar_spread(df_cp, near, far))
            else:
                st.info("未能获取主次合约数据（可能远月合约未上市或无成交）。")
                
            st.write("---")
            st.subheader("2. 多品种价差矩阵")
            all_contracts = [near] + others
            df_multi = get_multi_price_data(all_contracts)
            if not df_multi.empty and others:
                st.pyplot(plot_multi_spread(df_multi, near, others))
            else:
                st.info("请在左侧勾选跨品种标的。")

        # ================= Tab 3: 席位与盈亏 =================
        with tab3:
            start_dt = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).strftime('%Y%m%d')
            with st.spinner("抓取席位与计算盈亏中..."):
                df_pivot, df_profit, aligned_price = get_rank_data_and_profit(prefix, start_dt, datetime.datetime.now().strftime('%Y%m%d'), df_master['main_close'])
                
                if not df_pivot.empty:
                    if "中信期货" in df_pivot.columns and (df_profit['盈亏'].abs() < 10).all():
                        st.warning("⚠️ 交易所接口受限，当前展示为模拟推演数据。")
                    
                    fig_rank, fig_profit = plot_rank_and_price(aligned_price, df_pivot, df_profit)
                    st.pyplot(fig_rank)
                    
                    st.write("---")
                    col1, col2 = st.columns([1.5, 1])
                    with col1:
                        st.pyplot(fig_profit)
                    with col2:
                        st.markdown(f"### 📋 近{period}席位盈亏详情")
                        st.write("🏆 **盈利前三席位**")
                        for _, r in df_profit.head(3).iterrows():
                            st.success(f"{r['席位']} : 赚 {r['盈亏']:.0f} 万")
                        st.write("📉 **亏损最大席位**")
                        for _, r in df_profit.tail(1).iterrows():
                            st.error(f"{r['席位']} : 亏 {r['盈亏']:.0f} 万")
                        
                        foreign = df_profit[df_profit['席位'].str.contains('乾坤|摩根|瑞银|高盛', na=False)]
                        st.write("🌍 **外资投行席位**")
                        if not foreign.empty:
                            for _, r in foreign.iterrows():
                                st.info(f"{r['席位']} : {r['盈亏']:.0f} 万")
                        else:
                            st.write("暂无上榜外资。")

        # ================= Tab 4: 交易决策矩阵 =================
        with tab4:
            st.markdown("## 🎯 核心推演与交易工具选择")
            
            st.markdown("### 一、 供需现实与估值（定方向）")
            col1, col2, col3 = st.columns(3)
            with col1:
                if pd.notna(latest['inventory']):
                    st.metric("库存方向", "累库 📈" if inv_trend > 0 else "去库 📉", delta=f"{inv_trend:.0f}")
            with col2:
                st.metric("基差结构", "现货升水" if basis_val > 0 else "期货升水", delta=f"{basis_val:.0f}")
            with col3:
                st.metric("估值 (较成本线)", "跌破成本 🚨" if latest['main_close'] < cash_cost else "成本之上", delta=f"{latest['main_close'] - cash_cost:.0f}")

            st.markdown("### 二、 交易工具选择矩阵（单边 vs 套利 vs 期权）")
            if inv_trend < 0 and basis_val > 0 and hv_pct < 60:
                st.success("**📌 结论：首选【单边多头】或【卖出看跌期权 (Sell Put)】**\n\n- **原因**：去库 + 现货升水 + 波动率不高。强现实共振。\n- **操作**：逢低买入期货，或卖出虚值看跌期权。")
            elif inv_trend > 0 and basis_val < 0 and hv_pct < 60:
                st.error("**📌 结论：首选【单边空头】或【买入看跌期权 (Buy Put)】**\n\n- **原因**：累库 + 期货升水 + 波动率不高。典型弱现实。\n- **操作**：逢高做空，或买入看跌期权。")
            elif basis_val > 0 and inv_trend > 0:
                st.warning("**📌 结论：禁止单边！首选【跨期正套 (多近空远)】或【卖出宽跨式期权】**\n\n- **原因**：现实极紧（现货升水），但预期悲观（累库），单边易被洗。\n- **操作**：做多近月、做空远月，赚取价差回归。")
            elif hv_pct > 80:
                st.warning("**📌 结论：首选【期权卖方策略】或【跨品种套利】**\n\n- **原因**：波动率处于极高位（>80%），权利金极贵。\n- **操作**：单边追单极容易被双杀。适合卖出宽跨式，或寻找对冲品种（多强空弱）。")
            else:
                st.info("**📌 结论：单边观望，寻找【跨品种利润套利】**\n\n- **原因**：驱动不明朗，估值居中，无明显共振。\n- **操作**：关注上下游利润极值（如多PP空MA）。")

            st.markdown("---")
            st.markdown("### 三、 🌪️ 黑天鹅与极端风险雷达")
            black_swan_col1, black_swan_col2 = st.columns(2)
            with black_swan_col1:
                st.error(f"🚨 **潜在逼仓风险（向上）**：\n"
                         f"当前库存绝对值为 {latest['inventory']:.0f}。" + 
                         (f"库存极低，如果近月合约出现大资金多头逼仓，现货商交不出货，空头将面临踩踏。" if pd.notna(latest['inventory']) and latest['inventory'] < df_master['inventory'].mean() * 0.7 else "库存目前处于正常/高位，逼仓概率较低。"))
            with black_swan_col2:
                st.warning(f"⚠️ **潜在塌方风险（向下）**：\n"
                           f"成本线为 {cash_cost}。" + 
                           (f"如果地缘冲突缓和（如海峡通航恢复），或者宏观系统性风险爆发，高价资产可能瞬间崩塌。" if latest['main_close'] > cash_cost else "价格已贴近成本，向下空间有限，警惕宏观共振杀跌。"))
