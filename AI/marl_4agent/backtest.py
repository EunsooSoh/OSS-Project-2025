import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import (
    DEVICE, N_AGENTS, WINDOW_SIZE
)
from data_processor import DataProcessor
from environment import MARLStockEnv
from qmix_model import QMIX_Learner

# --- 백테스트 결과 그래프 함수 ---
def plot_backtest_results(portfolio_values, daily_pnls, test_prices, initial_capital):
    """백테스트 결과를 시각화하는 함수 (PnL 기준)"""
    dates = test_prices.index[:len(portfolio_values)]
    
    # 성과 지표 계산
    returns = pd.Series(daily_pnls) / initial_capital
    
    # Sharpe Ratio
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
    
    # Sortino Ratio (하방 변동성만 고려)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-9
    sortino = (returns.mean() / (downside_std + 1e-9)) * np.sqrt(252)
    
    # MDD (Maximum Drawdown)
    cumulative = np.array(portfolio_values)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    mdd = drawdown.min() * 100
    
    # PnL 계산 (누적 손익)
    qmix_pnl = np.array(portfolio_values) - initial_capital
    
    # Buy & Hold PnL 계산
    kospi_start = test_prices.iloc[0]
    shares_bought = initial_capital / kospi_start
    buyhold_values = [shares_bought * price for price in test_prices.iloc[:len(portfolio_values)]]
    buyhold_pnl = np.array(buyhold_values) - initial_capital
    
    # KOSPI 지수 PnL (지수 자체의 변화)
    kospi_index_pnl = (test_prices.iloc[:len(portfolio_values)] - kospi_start).values
    kospi_index_pnl_normalized = (kospi_index_pnl / kospi_start) * initial_capital
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 제목과 성과 지표
    final_qmix_pnl = qmix_pnl[-1]
    final_buyhold_pnl = buyhold_pnl[-1]
    title = f'QMIX 4-Agent 백테스트 성과 (PnL 비교)\n초기자금: {initial_capital:,.0f}원 | QMIX PnL: {final_qmix_pnl:,.0f}원 | Buy&Hold PnL: {final_buyhold_pnl:,.0f}원'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # QMIX Agent PnL
    ax.plot(dates, qmix_pnl, label='QMIX Agent', color='#2E86AB', linewidth=2.5, zorder=3)
    
    # Buy & Hold PnL
    ax.plot(dates, buyhold_pnl, label='Buy & Hold (삼성전자)', color='#A23B72', linewidth=2, linestyle='--', alpha=0.8, zorder=2)
    
    # KOSPI 지수 PnL
    ax.plot(dates, kospi_index_pnl_normalized, label='KOSPI 지수', color='#F18F01', linewidth=2, linestyle='-.', alpha=0.7, zorder=2)
    
    # 손익 0 기준선
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='손익 0')
    
    # 양수/음수 영역 색칠
    ax.fill_between(dates, 0, qmix_pnl, where=(qmix_pnl >= 0), alpha=0.1, color='green', interpolate=True)
    ax.fill_between(dates, 0, qmix_pnl, where=(qmix_pnl < 0), alpha=0.1, color='red', interpolate=True)
    
    # 축 설정
    ax.set_xlabel('날짜', fontsize=12, fontweight='bold')
    ax.set_ylabel('누적 손익 (PnL, 원)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Y축 포맷 (백만 원 단위)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # X축 포맷 (2개월 간격)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 성과 지표 텍스트 박스
    textstr = f'Sharpe: {sharpe:.3f}\nSortino: {sortino:.3f}\nMDD: {mdd:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return sharpe, sortino, mdd

# --- 4개 에이전트의 신호 변환 ---
def convert_joint_action_to_signal(joint_action, action_map):
    action_to_score = {"Long": 1, "Hold": 0, "Short": -1}
    score = sum(action_to_score[action_map[a]] for a in joint_action)
    
    if score >= 3:
        return "적극 매수"
    elif score == 2 or score == 1:
        return "매수"
    elif score == 0:
        return "보유"
    elif score == -1 or score == -2:
        return "매도"
    elif score <= -3:
        return "적극 매도"
    return "보유"

# --- AI 설명 생성 ---
def generate_ai_explanation(final_signal, agent_analyses):
    all_importances = {}
    for _, _, importance_list in agent_analyses:
        for feature, imp in importance_list:
            all_importances[feature] = all_importances.get(feature, 0.0) + imp
            
    sorted_features = sorted(all_importances.items(), key=lambda item: item[1], reverse=True)
    
    explanation = f"AI가 '{final_signal}'을 결정한 주된 이유는 다음과 같습니다.\n\n"
    
    if not sorted_features:
        return explanation + "데이터 분석 중입니다."
        
    top_feature_1 = sorted_features[0][0]
    explanation += f"  1. '{top_feature_1}' 지표의 최근 움직임을 가장 중요하게 고려했습니다.\n"
    
    if len(sorted_features) > 1:
        top_feature_2 = sorted_features[1][0]
        explanation += f"  2. '{top_feature_2}' 지표가 2순위로 결정에 영향을 미쳤습니다.\n"
        
    if len(sorted_features) > 2:
        top_feature_3 = sorted_features[2][0]
        explanation += f"  3. 마지막으로 '{top_feature_3}' 지표를 참고했습니다.\n"
        
    return explanation

# --- UI 출력 함수 ---
def print_ui_output(final_signal, ai_explanation, current_indicators, best_q_total_value):
    print("\n\n=============================================")
    print("      [ 📱 리브리 AI 분석 결과 (삼성전자) ]")
    print("=============================================")
    
    print("\n--- 1. AI 최종 신호 ---")
    print(f"    {final_signal}")
    print(f"    (예상 팀 Q-Value: {best_q_total_value:.4f})")
    
    print("\n--- 2. AI 설명 ---")
    print(ai_explanation)
    
    print("\n--- 3. 기술적 분석 상세 (최종일 기준) ---")
    print("    (AI가 입수하여 분석한 원본 데이터입니다.)\n")
    technical_indicators = [
        'SMA20', 'MACD', 'MACD_Signal', 'RSI', 'Stoch_K', 'Stoch_D', 
        'ATR', 'Bollinger_B', 'VIX'
    ]
    fundamental_indicators = ['ROA', 'DebtRatio', 'AnalystRating']
    
    for indicator in technical_indicators:
        if indicator in current_indicators:
            print(f"    - {indicator:<13}: {current_indicators[indicator]:.2f}")
            
    print("\n    (펀더멘탈 및 기타 데이터)\n")
    for indicator in fundamental_indicators:
         if indicator in current_indicators:
            print(f"    - {indicator:<13}: {current_indicators[indicator]:.2f}")
        
    print("=============================================")

# --- 메인 백테스트 함수 ---
def main():
    parser = argparse.ArgumentParser(description="QMIX Stock Trading Backtest (Load Trained Model)")
    parser.add_argument('--capital', type=float, default=10000000, help="투자 금액 (원)")
    parser.add_argument('--model', type=str, default='qmix_model.pth', help="학습된 모델 파일 경로")
    args = parser.parse_args()
    
    CAPITAL = args.capital
    MODEL_PATH = args.model
    
    print(f"\n=== 백테스트 설정 ===")
    print(f"투자 금액: {CAPITAL:,.0f}원")
    print(f"모델 파일: {MODEL_PATH}")
    print(f"사용 장치: {DEVICE}")

    # 데이터 로드
    processor = DataProcessor()
    (features_unnormalized_df, prices_df, feature_names,
     agent_0_cols, agent_1_cols, agent_2_cols, agent_3_cols) = processor.process()

    # 백테스팅 기간: 마지막 1년 (252 거래일)
    total_days = len(features_unnormalized_df)
    test_days = 252
    split_idx = total_days - test_days
    
    if split_idx < WINDOW_SIZE * 2:
        print("오류: 데이터가 너무 적어 백테스트가 불가능합니다.")
        return

    train_features_unnorm = features_unnormalized_df.iloc[:split_idx]
    test_features_unnorm = features_unnormalized_df.iloc[split_idx:]
    test_prices = prices_df.iloc[split_idx:]
    
    print(f"\n--- 데이터 분할 정보 ---")
    print(f"전체 데이터: {total_days}일")
    print(f"백테스팅 데이터: {len(test_features_unnorm)}일 ({test_prices.index[0]} ~ {test_prices.index[-1]})")

    # 정규화
    train_features, test_features = processor.normalize_data(train_features_unnorm, test_features_unnorm)

    # 환경 생성
    test_env = MARLStockEnv(
        test_features, test_prices, 
        agent_0_cols, agent_1_cols, agent_2_cols, agent_3_cols,
        n_agents=N_AGENTS, window_size=WINDOW_SIZE
    )
    
    obs_dims_list = [
        test_env.observation_dim_0,
        test_env.observation_dim_1,
        test_env.observation_dim_2,
        test_env.observation_dim_3
    ]
    state_dim = test_env.state_dim
    action_dim = test_env.action_dim

    # 학습된 모델 로드
    print(f"\n--- 학습된 모델 로드 중: {MODEL_PATH} ---")
    learner = QMIX_Learner(obs_dims_list, action_dim, state_dim, DEVICE)
    learner.load_model(MODEL_PATH)

    # 포트폴리오 초기화
    user_portfolio = {
        'capital': CAPITAL,
        'positions': [0] * N_AGENTS,
        'entry_prices': [0.0] * N_AGENTS,
        'shares': 0
    }

    print("\n--- 백테스트 수행 중 ---")
    print(f"--- 초기 투자 금액: {CAPITAL:,.0f}원 ---")
        
    obs_dict, info = test_env.reset(initial_portfolio=user_portfolio)
    global_state = info["global_state"]
    all_team_rewards = []
    all_raw_pnls = []
    portfolio_values = [CAPITAL]
    current_step = 0
    
    while current_step < test_env.max_steps:
        actions_dict = learner.select_actions(obs_dict, 0.0)  # Epsilon = 0.0 (탐험 없음)
        obs_dict, rewards_dict, dones_dict, _, info = test_env.step(actions_dict)
        all_team_rewards.append(rewards_dict['agent_0'])
        all_raw_pnls.append(info["raw_pnl"])
        portfolio_values.append(info["portfolio_value"])
        global_state = info["global_state"]
        current_step += 1
        if dones_dict['__all__']:
            break
    
    final_portfolio_value = portfolio_values[-1]
    final_shares = info["shares"]
    final_cash = info["cash"]

    print("\n--- 백테스트 성능 지표 ---")
    test_days_actual = len(all_team_rewards)
    if test_days_actual > 0:
        all_rewards_series = pd.Series(all_team_rewards)
        all_raw_pnls_series = pd.Series(all_raw_pnls)
        
        total_pnl = all_raw_pnls_series.sum()
        daily_avg_pnl = all_raw_pnls_series.mean()
        daily_std = all_rewards_series.std() + 1e-9
        sharpe_ratio = (all_rewards_series.mean() / daily_std) * np.sqrt(252)
        win_days = (all_raw_pnls_series > 0).sum()
        win_rate = (win_days / test_days_actual) * 100.0
        
        total_return_pct = ((final_portfolio_value - CAPITAL) / CAPITAL) * 100
        
        print(f"    - 백테스트 기간    : {test_days_actual} 일")
        print(f"    - 초기 투자 금액   : {CAPITAL:,.0f} 원")
        print(f"    - 최종 포트폴리오  : {final_portfolio_value:,.0f} 원")
        print(f"    - 보유 주식        : {final_shares} 주")
        print(f"    - 보유 현금        : {final_cash:,.0f} 원")
        print(f"    - 누적 수익(PnL)   : {total_pnl:,.0f} 원 ({total_return_pct:+.2f}%)")
        print(f"    - 일 평균 수익     : {daily_avg_pnl:,.0f} 원")
        print(f"    - 일 수익 변동성   : {daily_std:.4f}")
        print(f"    - 샤프 비율 (연환산): {sharpe_ratio:.3f}")
        print(f"    - 승률 (일별)      : {win_rate:.2f}% ({win_days}/{test_days_actual}일)")
        
        # 그래프 생성
        print("\n--- 백테스트 결과 그래프 생성 중 ---")
        graph_sharpe, graph_sortino, graph_mdd = plot_backtest_results(
            portfolio_values, all_raw_pnls, test_prices, CAPITAL
        )
        print(f"    Sharpe Ratio: {graph_sharpe:.3f}")
        print(f"    Sortino Ratio: {graph_sortino:.3f}")
        print(f"    MDD: {graph_mdd:.2f}%")
        print("    그래프가 저장되었습니다: backtest_results.png")
    else:
        print("    - 백테스트 기간이 0일이어서 성능을 측정할 수 없습니다.")

    # --- 최종일 상세 분석 ---
    print("\n--- 최종일 예측 상세 분석 ---")
    
    final_obs_dict = obs_dict
    action_map = {0: "Long", 1: "Hold", 2: "Short"}
    action_indices = list(action_map.keys())
    
    obs_tensors = [
        torch.FloatTensor(final_obs_dict[f'agent_{i}']).unsqueeze(0).to(DEVICE) 
        for i in range(N_AGENTS)
    ]
    state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(DEVICE)
    
    q_vals_all_agents = []
    with torch.no_grad():
        for i, agent in enumerate(learner.agents):
            q_vals_all_agents.append(agent.get_q_values(obs_tensors[i]))

    # 4D 그리드 계산
    agent_q_inputs = []
    action_tuples = []
    
    q_vals_0 = q_vals_all_agents[0].squeeze(0)
    q_vals_1 = q_vals_all_agents[1].squeeze(0)
    q_vals_2 = q_vals_all_agents[2].squeeze(0)
    q_vals_3 = q_vals_all_agents[3].squeeze(0)

    for a0_idx in action_indices:
        for a1_idx in action_indices:
            for a2_idx in action_indices:
                for a3_idx in action_indices:
                    q0 = q_vals_0[a0_idx]
                    q1 = q_vals_1[a1_idx]
                    q2 = q_vals_2[a2_idx]
                    q3 = q_vals_3[a3_idx]
                    agent_q_inputs.append(torch.stack([q0, q1, q2, q3]))
                    action_tuples.append((a0_idx, a1_idx, a2_idx, a3_idx))
    
    agent_q_batch = torch.stack(agent_q_inputs) 
    state_batch = state_tensor.repeat(len(action_tuples), 1)

    with torch.no_grad():
        all_q_totals = learner.mixer(agent_q_batch, state_batch)
    
    best_q_total_value = all_q_totals.max().item()
    best_joint_action_idx_flat = all_q_totals.argmax().item()
    best_joint_action_indices = action_tuples[best_joint_action_idx_flat]
    
    # XAI 분석
    agent_analyses = []
    feature_names_list = [agent_0_cols, agent_1_cols, agent_2_cols, agent_3_cols]
    n_features_list = [
        test_env.n_features_agent_0, 
        test_env.n_features_agent_1, 
        test_env.n_features_agent_2,
        test_env.n_features_agent_3
    ]
    
    for i, agent in enumerate(learner.agents):
        obs = final_obs_dict[f'agent_{i}']
        agent_feature_names = feature_names_list[i]
        n_features_agent = n_features_list[i]

        action_idx, q_values, importance = agent.get_prediction_with_reason(
            obs, 
            agent_feature_names,
            WINDOW_SIZE, 
            n_features_agent
        )
        agent_analyses.append((action_idx, q_values, importance))
        
    final_signal = convert_joint_action_to_signal(best_joint_action_indices, action_map)
    ai_explanation = generate_ai_explanation(final_signal, agent_analyses)
    
    current_indicator_values = test_features_unnorm.iloc[-1]
    
    # UI 포맷으로 출력
    print_ui_output(
        final_signal=final_signal,
        ai_explanation=ai_explanation,
        current_indicators=current_indicator_values,
        best_q_total_value=best_q_total_value
    )

if __name__ == "__main__":
    main()
