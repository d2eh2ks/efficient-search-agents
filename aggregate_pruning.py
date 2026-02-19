import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 설정 및 경로 정의 (Search-R1 디렉토리 기준)
# ==========================================
base_dir = "./eval_results_general_strict"

# 분석할 태스크와 폴더명 매칭
dataset_tasks = {
    "NQ": "nq",
    "TriviaQA": "triviaqa",
    "PopQA": "popqa"
}

# 분석할 스텝 (Step 1, 2, 3)
target_steps = [1, 2, 3]
# 수집된 검색 횟수 그룹 (Search 1, 2, 3)
search_ns = [1, 2, 3]

print("📈 데이터셋별 Pruning 결과 통합 취합 중...")

# ==========================================
# 2. 데이터 취합 (모든 그룹 합산)
# ==========================================
plot_data = {}

for label, folder in dataset_tasks.items():
    # 각 스텝별 {정답수, 전체수} 집계용 초기화
    step_metrics = {s: {"correct": 0, "total": 0} for s in target_steps}
    
    for n in search_ns:
        file_path = os.path.join(base_dir, folder, f"results_search_{n}_pruned.csv")
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                for s in target_steps:
                    col_name = f"Step_{s}_Correct"
                    if col_name in df.columns:
                        # 불리언/문자열 데이터 타입 보정
                        vals = df[col_name].map({
                            'True': True, 'False': False, 
                            'TRUE': True, 'FALSE': False,
                            True: True, False: False
                        })
                        step_metrics[s]["correct"] += vals.sum()
                        step_metrics[s]["total"] += len(df)
            except Exception as e:
                print(f"   ⚠️ {file_path} 처리 중 에러: {e}")

    # 최종 정확도 (%) 계산
    accs = []
    for s in target_steps:
        if step_metrics[s]["total"] > 0:
            accs.append((step_metrics[s]["correct"] / step_metrics[s]["total"]) * 100)
        else:
            accs.append(0) # 데이터가 없는 경우 0으로 표시
    
    plot_data[label] = accs
    print(f"   ✅ {label} 취합 완료: S1({accs[0]:.1f}%) | S2({accs[1]:.1f}%) | S3({accs[2]:.1f}%)")

# ==========================================
# 3. 통합 그래프 생성 (지시사항: 단일 PNG, Step별 구분)
# ==========================================
plt.figure(figsize=(12, 7))
x = np.arange(len(target_steps)) # Step 1, 2, 3 위치
width = 0.25 # 막대 너비

# 데이터셋별로 막대 그리기 (NQ, Trivia, Pop 순서대로 그룹화)
for i, (label, accs) in enumerate(plot_data.items()):
    offset = i * width
    bars = plt.bar(x + offset, accs, width=width, label=label)
    
    # 각 막대 상단에 수치 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 그래프 스타일 설정
plt.title("Search-R1 Pruning Performance: EM Accuracy by Reasoning Steps", fontsize=15, pad=20)
plt.xlabel("Pruning Level (Reasoning Progress)", fontsize=12)
plt.ylabel("Exact Match Accuracy (%)", fontsize=12)
plt.xticks(x + width, ["Step 1\n(Internal Knowledge)", "Step 2\n(After Search 1)", "Step 3\n(After Search 2)"])
plt.ylim(0, 115) # 수치 표시를 위한 여백
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title="Datasets", loc='upper left', fontsize=10)

plt.tight_layout()

# 🚨 지시사항: 단일 PNG 파일로 저장
output_name = "pruning_aggregated_report.png"
plt.savefig(output_name, dpi=300)
print(f"\n💾 통합 리포트 그래프 저장 완료: {output_name}")
