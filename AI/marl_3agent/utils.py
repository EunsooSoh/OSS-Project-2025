import numpy as np

def convert_joint_action_to_signal(joint_action, action_map):
    """3개 에이전트의 행동을 종합하여 최종 매매 신호 생성"""
    action_to_score = {"Long": 1, "Hold": 0, "Short": -1}
    score = sum(action_to_score[action_map[a]] for a in joint_action)
    
    if score >= 3: return "적극 매수"
    elif score > 0: return "매수"
    elif score == 0: return "보유"
    elif score < 0 and score > -3: return "매도"
    elif score <= -3: return "적극 매도"
    return "보유"

def generate_ai_explanation(final_signal, agent_analyses):
    """AI 판단 근거(XAI) 텍스트 생성"""
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

def print_ui_output(final_signal, ai_explanation, current_indicators, q_total_grid, best_q_total_value, action_names):
    """콘솔에 최종 결과 출력"""
    print("\n\n=============================================")
    print("      [ 📱 리브리 AI 분석 결과 ]")
    print("=============================================")
    print("\n--- 1. AI 최종 신호 ---")
    print(f"    {final_signal}")
    print(f"    (예상 팀 Q-Value: {best_q_total_value:.4f})")
    print("\n--- 2. AI 설명 ---")
    print(ai_explanation)
    print("\n--- 3. 주요 지표 현황 ---")
    for k, v in current_indicators.items():
        if k in ['SMA20', 'RSI', 'MACD', 'VIX']: # 주요 지표만 간략 출력
            print(f"    - {k:<10}: {v:.2f}")
    print("=============================================")