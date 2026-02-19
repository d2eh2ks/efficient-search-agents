import os
import re
import torch
import gc
import json
import faiss
import pandas as pd
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# 1. 로컬 검색 엔진 (WikiSearchEngine) 설정
# -----------------------------------------------------------------------------
class WikiSearchEngine:
    def __init__(self):
        # 서버 내 위키피디아 인덱스 및 데이터 경로
        self.BASE_PATH = "/server10/dongwan/wiki_2018/data"
        self.INDEX_FILE = os.path.join(self.BASE_PATH, "wiki_e5_base.index")
        self.PARQUET_FILE = os.path.join(self.BASE_PATH, "wiki_passages.parquet")
        self.MODEL_ID = "intfloat/e5-base-v2" # E5 임베딩 모델
        
        print("📂 로컬 위키 인덱스 로드 중...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_model = SentenceTransformer(self.MODEL_ID, device=self.device)
        
        if not os.path.exists(self.INDEX_FILE):
            raise FileNotFoundError(f"Index file not found: {self.INDEX_FILE}")
            
        self.index = faiss.read_index(self.INDEX_FILE)
        # 메모리 효율을 위해 title과 text 컬럼만 로드
        self.df = pd.read_parquet(self.PARQUET_FILE, columns=['title', 'text'])
        print(f"✅ 검색 엔진 로드 완료! (인덱스 크기: {len(self.df):,})")

    def search(self, query, k=3):
        """FAISS 인덱스를 통해 로컬 문서를 검색합니다."""
        query_with_prefix = f"query: {query}"
        
        with torch.no_grad():
            query_emb = self.embed_model.encode([query_with_prefix], convert_to_numpy=True, normalize_embeddings=True)
        
        distances, indices = self.index.search(query_emb, k)
        
        format_reference = ''
        for i in range(k):
            idx = indices[0][i]
            if idx < len(self.df):
                title = self.df.iloc[idx]['title']
                text = self.df.iloc[idx]['text']
                format_reference += f"Doc {i+1} (Title: {title})\nContent: {text}\n\n"
        
        return format_reference if format_reference else "No relevant documents found in local wiki."

# -----------------------------------------------------------------------------
# 2. 모델 및 앱 초기화
# -----------------------------------------------------------------------------
MODEL_ID = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Search-R1 Model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 검색 엔진 인스턴스 생성
wiki_engine = WikiSearchEngine()

# -----------------------------------------------------------------------------
# 3. API 엔드포인트
# -----------------------------------------------------------------------------
@app.get("/ask")
async def ask(question: str):
    torch.cuda.empty_cache()
    gc.collect()

    # Search-R1 프롬프트 규격
    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    curr_eos = [tokenizer.eos_token_id, 151645, 151643]
    steps_log = []
    current_full_prompt = prompt
    step_count = 0
    MAX_STEPS = 10 
    
    # 다음 스텝 로그에 표시할 텍스트 (초기값은 질문 원문)
    display_query = question 

    while step_count < MAX_STEPS:
        step_count += 1
        input_ids = tokenizer.encode(current_full_prompt, return_tensors='pt').to(device)
        
        # 🚀 [수정] 현재 스텝에서 처리할 쿼리(혹은 질문)를 출력
        print(f"🔄 Processing Step {step_count} (Query: {display_query[:50]}...)")

        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            stop_strings=["</search>"], 
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            use_cache=True
        )

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # 태그 보정
        if "<search>" in output_text and "</search>" not in output_text:
            output_text += "</search>"

        # 최종 정답 도출 시 루프 종료
        if any(token in outputs[0][-3:].tolist() for token in curr_eos) or "<answer>" in output_text:
            steps_log.append({"step": step_count, "content": output_text.strip()})
            break

        # 검색 쿼리 추출
        search_pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        match = search_pattern.search(output_text)
        
        if match:
            tmp_query = match.group(1).strip()
            
            # 🚀 [수정] 다음 스텝에서 출력될 쿼리 내용 업데이트
            display_query = tmp_query 
            
            # 로컬 위키 검색
            search_results = wiki_engine.search(tmp_query, k=3)
            
            combined_step_content = f"{output_text}\n<information>\n{search_results}</information>\n\n"
            steps_log.append({"step": step_count, "content": combined_step_content})
            current_full_prompt += combined_step_content
        else:
            steps_log.append({"step": step_count, "content": output_text.strip()})
            break 

    return {
        "question": question,
        "total_steps": step_count,
        "steps": steps_log
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)