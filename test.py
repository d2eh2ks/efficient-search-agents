import os
import re
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# 0) 경로 / 설정
# =========================
MODEL_ID = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"

# 🚨 Bamboogle 경로로 변경
INPUT_BASE_DIR = "./eval_results_bamboogle_full"
OUTPUT_DIR = "./pruning_results_bamboogle"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🚨 분석 대상 태스크: Bamboogle 단독
TASKS = ["bamboogle"]

# 🚨 데이터가 유의미한 Step 3(검색 2회), Step 4(검색 3회)만 분석
# Search 2회 -> Total Steps 3 (24개)
# Search 3회 -> Total Steps 4 (23개)
SEARCH_N_TO_TOTAL_STEPS = {2: 3, 3: 4}
SEARCH_N_LIST = [2, 3]

MAX_NEW_TOKENS = 20
DO_SAMPLE = False

SYSTEM_PROMPT = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# =========================
# 1) 유틸
# =========================
def str_normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_step_cols(df: pd.DataFrame):
    # "Step 1", "Step 2", ... 자동 인식 및 정렬
    step_cols = []
    for c in df.columns:
        m = re.match(r"Step\s+(\d+)$", str(c).strip())
        if m:
            step_cols.append((int(m.group(1)), c))
    step_cols.sort(key=lambda x: x[0])
    return [c for _, c in step_cols]


def answers_match(pred: str, gold: str) -> bool:
    # 포함/역포함 + normalize
    p = str_normalize(pred)
    g = str_normalize(gold)
    if p == "" or g == "":
        return False
    return (p == g) or (p in g) or (g in p)


# =========================
# 2) 모델 로딩
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("♻️ 모델 로딩 중... (GPU 메모리 확인하세요)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True
)


def generate_forced_answer(full_context: str) -> str:
    inputs = tokenizer(full_context, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=DO_SAMPLE,
        )
    gen = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    if "</answer>" in gen:
        return gen.split("</answer>")[0].strip()
    return gen.strip()


# =========================
# 3) pruning 평가 + 그래프용 집계
# =========================
# metrics[search_n][task][percent] = accuracy(0~100)
metrics = {sn: {t: {} for t in TASKS} for sn in SEARCH_N_LIST}

# 샘플 단위 결과 저장
row_level_records = []

print(f"🚀 Bamboogle Pruning 평가 시작 (Tasks: {TASKS})")

for search_n in SEARCH_N_LIST:
    total_steps = SEARCH_N_TO_TOTAL_STEPS[search_n]
    # prefix 길이: 0..total_steps
    prefix_lengths = list(range(0, total_steps + 1))
    percents = [round(100 * (k / total_steps)) for k in prefix_lengths]

    for task in TASKS:
        in_csv = os.path.join(INPUT_BASE_DIR, task, f"results_search_{search_n}.csv")
        
        if not os.path.exists(in_csv):
            print(f"⚠️ 파일 없음: {in_csv} (Search {search_n}회 데이터 없음)")
            continue

        df = pd.read_csv(in_csv).fillna("")
        if "Original Query" not in df.columns or "Final Answer" not in df.columns:
            print(f"❌ 컬럼 누락: {in_csv}")
            continue

        step_cols_all = extract_step_cols(df)
        if len(step_cols_all) < total_steps:
            print(f"❌ Step 컬럼 부족: {in_csv} (필요 {total_steps}개, 발견 {len(step_cols_all)}개)")
            continue

        step_cols = step_cols_all[:total_steps]

        correct_cnt = {p: 0 for p in percents}
        total_cnt = 0

        print(f"\n🧪 [{task.upper()} | Search {search_n}회 (Total {total_steps} Steps)] samples={len(df)}")

        for idx, row in df.iterrows():
            question = str(row["Original Query"]).strip()
            final_answer = str(row["Final Answer"]).strip()
            if question == "" or final_answer == "":
                continue

            total_cnt += 1

            base_prompt = SYSTEM_PROMPT.format(question=question)
            messages = [{"role": "user", "content": base_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            # --- Row별 기록 ---
            sample_record = {
                "task": task,
                "search_n": search_n,
                "total_steps": total_steps,
                "question": question,
                "final_answer": final_answer,
            }

            # --- Percentage Loop ---
            for k in prefix_lengths:
                percent = round(100 * (k / total_steps))

                # 100% (Full Context)는 모델이 생성했던 답(Final Answer)과 일치한다고 가정
                if k == total_steps:
                    pred = final_answer
                    is_same = True
                else:
                    accumulated = ""
                    if k > 0:
                        for j in range(k):
                            accumulated += str(row[step_cols[j]]).strip() + "\n"

                    # 강제 생성 트리거
                    trigger_prompt = formatted_prompt + accumulated + "</think>\n<answer>"
                    pred = generate_forced_answer(trigger_prompt)
                    is_same = answers_match(pred, final_answer)

                sample_record[f"pred_{percent}"] = pred
                sample_record[f"same_{percent}"] = is_same

                if is_same:
                    correct_cnt[percent] += 1

            row_level_records.append(sample_record)

            if (idx + 1) % 5 == 0: # 데이터가 적으니 5개 단위로 로그
                print(f"   - processed {idx+1}/{len(df)}")

        # 태스크별 통계 계산
        for percent in percents:
            acc = 0.0 if total_cnt == 0 else (100.0 * correct_cnt[percent] / total_cnt)
            metrics[search_n][task][percent] = acc

        print(f"✅ 완료. 유효 샘플={total_cnt}")
        for percent in percents:
            print(f"   - {percent}% Info: {metrics[search_n][task][percent]:.2f}% Match")


# =========================
# 4) 결과 저장 (CSV 요약 + Row-level)
# =========================
summary_rows = []
for search_n in SEARCH_N_LIST:
    total_steps = SEARCH_N_TO_TOTAL_STEPS[search_n]
    percents = [round(100 * (k / total_steps)) for k in range(0, total_steps + 1)]
    
    for task in TASKS:
        # 데이터가 아예 없었으면 스킵
        if not metrics[search_n][task]:
            continue
            
        row = {"task": task, "search_n": search_n, "total_steps": total_steps}
        for p in percents:
            row[f"acc_{p}"] = metrics[search_n][task].get(p, None)
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIR, "bamboogle_pruning_summary.csv")
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
print(f"\n💾 Summary Saved: {summary_path}")

row_df = pd.DataFrame(row_level_records)
row_path = os.path.join(OUTPUT_DIR, "bamboogle_pruning_rowlevel.csv")
row_df.to_csv(row_path, index=False, encoding="utf-8-sig")
print(f"💾 Row-level Saved: {row_path}")


# =========================
# 5) 그래프 생성 (Search 2회, 3회 두 개 패널)
# =========================
# 패널 개수: 2개
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.suptitle("Bamboogle Pruning Analysis (Small Sample)", fontsize=18)

# 패널 타이틀 설정
panel_titles = {
    2: "2-Step Search (Total 3 Steps) - n=24",
    3: "3-Step Search (Total 4 Steps) - n=23",
}

for col, sn in enumerate(SEARCH_N_LIST):
    ts = SEARCH_N_TO_TOTAL_STEPS[sn]
    percents = [round(100 * (k / ts)) for k in range(0, ts + 1)]
    
    # axes가 2개니까 리스트로 접근 가능
    ax = axes[col]

    ax.set_title(panel_titles.get(sn, f"Search {sn} Times"))
    
    for task in TASKS:
        if not metrics[sn][task]:
            continue
            
        ys = [metrics[sn][task].get(p, 0.0) for p in percents]
        ax.plot(percents, ys, marker="o", label=f"{task.upper()} (n={int(len(ys) > 0)})") # 단순 라벨

    ax.set_xlabel("% of Reasoning Sample Provided")
    ax.set_xticks(percents)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.4)

    if col == 0:
        ax.set_ylabel("% Same Answer as Complete Search")
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plot_path = os.path.join(OUTPUT_DIR, "bamboogle_pruning_plot.png")
plt.savefig(plot_path, dpi=200)
print(f"\n🖼️ Plot Saved: {plot_path}")

try:
    plt.show()
except Exception as e:
    print(f"(plt.show() skipped: {e})")