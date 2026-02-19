import json
import requests
import csv
import re
import os

# ==========================================
# 1. 환경 설정
# ==========================================
api_url = "http://0.0.0.0:8000/ask"
base_dir = "/home/tako/dongwan/Search-R1/data"
output_base_dir = "./eval_results_bamboogle_full"  # 폴더 구분
os.makedirs(output_base_dir, exist_ok=True)

# 🚨 타겟: Bamboogle Only, Steps: 1 ~ 10
dataset_configs = {
    "bamboogle_test.jsonl": list(range(1, 11))  # [1, 2, ..., 10]
}

# CSV 헤더 확장 (Step 1 ~ Step 10)
header = ["Original Query", "Total Steps"] + [f"Step {i}" for i in range(1, 11)] + ["Final Answer"]

# ==========================================
# 2. 전처리 및 유틸리티 함수
# ==========================================
def str_normalize(s):
    if not s: return ""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~')
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

# ==========================================
# 3. 메인 실행 루프
# ==========================================
def main():
    print(f"🚀 Bamboogle 전수 조사 시작 (Step 1~10, 총 125개 데이터)\n")

    for filename, target_steps in dataset_configs.items():
        task_name = filename.split('_')[0]
        file_path = os.path.join(base_dir, filename)

        if not os.path.exists(file_path):
            print(f"⚠️ {file_path} 없음. 스킵.")
            continue

        print(f"\n📂 [{task_name.upper()}] 시작... ({filename})")

        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        # 결과를 모을 딕셔너리 초기화 (1~10 step)
        collections = {step: [] for step in target_steps}
        counts = {step: 0 for step in target_steps}
        
        total_processed = 0

        try:
            for item in data:
                total_processed += 1
                question = item.get("question")
                gold_answers = item.get("golden_answers", [])

                try:
                    response = requests.get(api_url, params={"question": question}, timeout=600)
                    if response.status_code != 200: continue

                    res_data = response.json()
                    total_steps = res_data.get("total_steps")
                    steps_content = res_data.get("steps", [])

                    # 타겟 범위(1~10) 밖이면 스킵 (혹시 11 이상 나오면)
                    if total_steps not in target_steps:
                        print(f"⚠️ 범위 밖 Step 발생: {total_steps}")
                        continue

                    # 모델 정답 추출
                    model_answer_raw = ""
                    last_step_text = steps_content[-1]["content"] if steps_content else ""
                    match = re.search(r"<answer>(.*?)</answer>", last_step_text, re.DOTALL)
                    if match:
                        model_answer_raw = match.group(1).strip()

                    # 정답 검증 (EM)
                    norm_model = str_normalize(model_answer_raw)
                    is_correct = any(norm_model == str_normalize(gold) for gold in gold_answers)

                    if is_correct:
                        # CSV 행 생성 (최대 10개 스텝 공간 확보)
                        # 구조: [Query, TotalSteps, S1...S10, FinalAnswer]
                        row = [question, total_steps] + [""] * 10 + [model_answer_raw]
                        
                        for i, info in enumerate(steps_content[:total_steps]):
                            if i < 10: # 최대 10개까지만 기록
                                row[i + 2] = info["content"]

                        collections[total_steps].append(row)
                        counts[total_steps] += 1
                        
                        search_n = total_steps - 1
                        print(f"✅ [{total_processed}/{len(data)}] 검색 {search_n}회(S{total_steps}) 정답! (누적: {counts[total_steps]}개)")
                    else:
                        print(f"❌ [{total_processed}/{len(data)}] 오답")

                except Exception as e:
                    print(f"🔌 API 에러: {e}")
                    continue

        except KeyboardInterrupt:
            print("\n🚨 중단됨! 저장 시퀀스로 진입합니다.")

        # 파일 저장
        task_out_dir = os.path.join(output_base_dir, task_name)
        os.makedirs(task_out_dir, exist_ok=True)
        
        print(f"\n💾 결과 저장 중...")
        for step in target_steps:
            if collections[step]:
                search_n = step - 1
                out_file = os.path.join(task_out_dir, f"results_search_{search_n}.csv")
                with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(collections[step])
        
        # 최종 리포트
        print("\n📊 [최종 수집 현황]")
        for s in sorted(counts.keys()):
            if counts[s] > 0:
                print(f"  - Step {s} (검색 {s-1}회): {counts[s]}개")

if __name__ == "__main__":
    main()