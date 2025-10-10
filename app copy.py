import os
import io
import json
import base64
import random
import hashlib
import pickle
import threading
import time
from queue import Queue
from pathlib import Path

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from flask import Flask, request, jsonify, render_template, url_for, Response
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- Matplotlib Font Setup for Chinese ---
# This setup is important for charts to display Chinese characters correctly.
# You might need to provide a path to a font file on your system.
try:
    # On Render/Linux, you might need to install fonts, e.g., 'apt-get install -y fonts-noto-cjk'
    # For local Windows, provide a path like 'C:/Windows/Fonts/msyh.ttc' (Microsoft YaHei)
    font_path = 'C:/Windows/Fonts/simhei.ttf' # Example for Windows, change if needed
    if not Path(font_path).exists():
        # A common fallback path on Debian-based systems
        font_path = '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = [Path(font_path).stem]
    plt.rcParams['axes.unicode_minus'] = False # Fix for displaying minus sign
    print(f"中文图表字体设置成功: {Path(font_path).stem}")
except Exception as e:
    print(f"警告: 中文图表字体设置失败，图表中的中文可能显示为方框。错误: {e}")

# --- Basic Configuration ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- AI Client ---
client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("API_URL"))

# --- Vector Search Setup ---
VECTOR_INDEX_PATH = Path("vector_db/user_index.faiss")
VECTOR_METADATA_PATH = Path("vector_db/user_metadata.pkl")
embedding_model = None

# --- Real-time Streaming Setup ---
# A dictionary to hold queues for different analysis jobs
analysis_queues = {}

def row_to_text(row):
    """Converts a DataFrame row into a descriptive string for embedding."""
    return (
        f"用户画像：性别 {row['性别']}，年龄 {row['年龄']}岁，来自 {row['城市']} 的 {row['职业']}。"
        f"教育程度为 {row['教育程度']}，年收入 {row['收入区间']}。MBTI性格是 {row['MBTI/性格']}。"
        f"饮酒习惯：频率 {row['饮酒频率']}，酒龄 {row['酒龄']}年，偏好 {row['香型']} 白酒，"
        f"心理价位在 {row['白酒价格']}，主要用于 {row['用途']}。"
    )

def vectorize_database():
    """Creates and saves a FAISS index from the Excel data if it doesn't exist."""
    global embedding_model
    if VECTOR_INDEX_PATH.exists() and VECTOR_METADATA_PATH.exists():
        print("向量数据库已存在，直接加载。")
        return

    print("未找到向量数据库，开始首次创建...")
    try:
        df = pd.read_excel("panel_data_sample.xlsx")
        df = df.dropna().reset_index(drop=True)
    except FileNotFoundError:
        print("错误: panel_data_sample.xlsx 未找到!")
        return

    # Load a powerful multilingual model for creating embeddings
    embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    # Convert all user rows to text and then to vectors
    user_texts = df.apply(row_to_text, axis=1).tolist()
    print(f"正在为 {len(user_texts)} 位用户生成向量...")
    embeddings = embedding_model.encode(user_texts, show_progress_bar=True)
    
    # Create and save FAISS index
    VECTOR_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, str(VECTOR_INDEX_PATH))
    
    # Save metadata (mapping from index position to user ID)
    metadata = {'ids': df['用户ID'].tolist()}
    with open(VECTOR_METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
        
    print("向量数据库创建成功！")

def find_similar_users_knn(persona, top_n=5):
    """Finds most similar users using k-NN vector search."""
    global embedding_model
    if not VECTOR_INDEX_PATH.exists():
        return pd.DataFrame(), {} # Return empty if index is not ready

    # Load model only when needed
    if embedding_model is None:
        embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    # Load index and metadata
    index = faiss.read_index(str(VECTOR_INDEX_PATH))
    with open(VECTOR_METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    df = pd.read_excel("panel_data_sample.xlsx")

    # Create embedding for the new persona
    persona_text = (
        f"用户画像：性别 {persona.get('gender', '')}，年龄 {persona.get('age', '')}岁，来自 {persona.get('city', '')} 的 {persona.get('profession', '')}。"
        f"MBTI性格是 {persona.get('mbti', '')}。"
        f"其他偏好：教育程度 {persona.get('education', '')}，年收入 {persona.get('income', '')}，"
        f"心理价位 {persona.get('expected_price', '')}，偏好香型 {persona.get('preferred_aroma', '')}。"
    )
    persona_embedding = embedding_model.encode([persona_text])
    
    # Search the index
    distances, indices = index.search(persona_embedding.astype('float32'), top_n)
    
    # Get the user IDs and corresponding data
    matched_user_ids = [metadata['ids'][i] for i in indices[0]]
    top_df = df[df['用户ID'].isin(matched_user_ids)].copy()
    
    # Create simple insights from the matched users
    insights = {
        'top_aroma': top_df['香型'].mode()[0] if not top_df['香型'].mode().empty else "未知",
        'price_band': top_df['白酒价格'].mode()[0] if not top_df['白酒价格'].mode().empty else "未知",
        'typical_usage': top_df['用途'].mode()[0] if not top_df['用途'].mode().empty else "未知"
    }

    return top_df, insights

def generate_chart_base64(chart_type, data, title):
    """Generates a Matplotlib chart and returns it as a base64 string."""
    plt.figure(figsize=(6, 4))
    
    if chart_type == 'pie':
        if not data or sum(data.values()) == 0: return None
        plt.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', startangle=90, colors=['#4A90E2', '#F5A623', '#BD10E0', '#7ED321'])
        plt.title(title, pad=20)
    
    elif chart_type == 'line' or chart_type == 'bar':
        if not data: return None
        plt.bar(data.keys(), data.values(), color='#4A90E2')
        plt.title(title, pad=20)
        plt.ylabel("购买人数")
        plt.xticks(rotation=15, ha="right")

    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def get_ai_analysis_for_table(table_title, table_data):
    """Sends table data to AI for a brief analysis."""
    prompt = (
        f"你是一位数据分析师。以下是一个关于白酒购买行为的数据表格，标题为 '{table_title}'。\n\n"
        f"数据：\n{json.dumps(table_data, ensure_ascii=False, indent=2)}\n\n"
        f"请基于此数据，用一两句话给出一个简洁、精炼的商业洞察。直接陈述结论，无需客套。"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Use a fast and cheap model for this
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"表格分析API调用失败: {e}")
        return "AI洞察生成失败，请检查API连接。"

def build_persona_icon_prompt(persona):
    gender = map_gender(persona.get('gender')) or '未知性别'
    age = persona.get('age', '未知年龄')
    profession = persona.get('profession') or '职业未填写'
    mbti = (persona.get('mbti') or '').upper() or 'MBTI 未知'
    city = persona.get('city') or '未知城市'
    city_motif = f"subtle skyline or icon suggesting {city}" if city != '未知城市' else 'simple abstract background'
    mbti_style = MBTI_MOTIFS.get(mbti, 'balanced color palette, minimal abstract motif')
    prompt = (
        "Design a small flat avatar icon in a contemporary Chinese ink-and-watercolor fusion style with transparent background. "
        f"Depict a {age}-year-old {gender} working as {profession}, MBTI {mbti}. "
        f"Integrate {city_motif} as a faint backdrop and blend in brush-texture accents. "
        f"Style hints: {mbti_style}. "
        "Keep the composition minimalist, using gentle brush strokes and modern flat colors; no text."
        "Ensure a balanced, scalable composition in a circular frame, inspired by traditional Chinese art."
    )
    return prompt

def generate_persona_icon(persona, persona_index):
    try:
        ICON_DIR.mkdir(parents=True, exist_ok=True)
        persona_signature = hashlib.md5(
            json.dumps(persona, sort_keys=True, ensure_ascii=False).encode('utf-8')
        ).hexdigest()
        filename = f"persona_{persona_index + 1}_{persona_signature[:8]}.png"
        file_path = ICON_DIR / filename
        if file_path.exists():
            return filename

        prompt = build_persona_icon_prompt(persona)
        response = client.images.generate(
            model="gpt-4o-image-vip",
            prompt=prompt,
            size="1024x1024"
        )
        data_item = response.data[0]
        image_b64 = getattr(data_item, "b64_json", None) or getattr(data_item, "base64", None) or getattr(data_item, "image_base64", None)
        if image_b64:
            image_bytes = base64.b64decode(image_b64)
            file_path.write_bytes(image_bytes)
            return filename

        url = getattr(data_item, "url", None)
        if url:
            print(f"画像图标生成返回远程链接，直接使用: {url}")
            return url

        print("生成画像图标失败: 响应缺少图像内容")
        return None
    except Exception as exc:
        print(f"生成画像图标失败: {exc}")
        return None


def build_icon_url(filename):
    if not filename:
        return None
    if filename.startswith('http://') or filename.startswith('https://'):
        return filename
    return url_for('static', filename=f'persona_icons/{filename}', _external=False)


def long_running_analysis(job_id, personas, product_data):
    """The main analysis function, running in a background thread."""
    q = analysis_queues[job_id]
    
    try:
        product_description = product_data.get('description')
        base64_image = product_data.get('image').split(',')[1]
        
        # --- Stage 1: Individual Reports ---
        all_results = []
        for i, persona in enumerate(personas):
            # This is where we now use the fast k-NN search
            top_matches, ai_insights = find_similar_users_knn(persona, top_n=5)
            
            # ... (The logic to build `real_user_prompt` is complex, let's keep it similar)
            # This part can be copied from your colleague's `app.py`
            # For brevity, let's assume `real_user_prompt` is built here...

            # The COT prompt remains the same, an excellent idea
            structured_individual_prompt = "..." # Copy the COT prompt here

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[...], # same message structure
                max_tokens=1000
            )
            report_text = response.choices[0].message.content
            
            # Add decision parsing
            decision = "购买" if "【决策】购买" in report_text else "不购买"
            
            result_package = {
                "type": "individual_report",
                "data": {
                    "persona_id": i + 1,
                    "report": report_text,
                    "persona_details": persona, # Add icon URL later
                    "decision": decision
                }
            }
            q.put(result_package)
            all_results.append(result_package['data'])

        # --- Stage 2: Summary Report ---
        summary_prompt = "..." # Copy summary prompt here
        summary_response = client.chat.completions.create(...) # Summary call
        summary_report = summary_response.choices[0].message.content
        q.put({"type": "summary_report", "data": summary_report})

        # --- Stage 3: Chart & Table Analysis ---
        decisions = [res['decision'] for res in all_results]
        buy_count = decisions.count('购买')
        no_buy_count = decisions.count('不购买')

        # Chart 1: Overall Purchase Ratio (Pie)
        overall_data = {'购买': buy_count, '不购买': no_buy_count}
        chart_b64 = generate_chart_base64('pie', overall_data, "总体购买意向比例")
        analysis = get_ai_analysis_for_table("总体购买意向比例", overall_data)
        q.put({"type": "chart_and_analysis", "data": {"id": "overall", "title": "总体购买意向比例", "chart": chart_b64, "table": overall_data, "analysis": analysis}})
        
        # ... You can add many more charts here following the same pattern ...
        # Example for Gender Chart:
        gender_data = {'男': 0, '女': 0}
        for res in all_results:
            if res['decision'] == '购买':
                gender = res['persona_details']['gender']
                if gender == 'Male': gender_data['男'] += 1
                else: gender_data['女'] += 1
        
        chart_b64 = generate_chart_base64('pie', gender_data, "购买用户性别分布")
        analysis = get_ai_analysis_for_table("购买用户性别分布", gender_data)
        q.put({"type": "chart_and_analysis", "data": {"id": "gender", "title": "购买用户性别分布", "chart": chart_b64, "table": gender_data, "analysis": analysis}})
        
    except Exception as e:
        print(f"Analysis thread error: {e}")
        q.put({"type": "error", "data": str(e)})
    finally:
        q.put({"type": "done"})
        del analysis_queues[job_id]


@app.route('/generate_report', methods=['POST'])
def generate_report_start():
    """Starts the analysis and returns a job ID for streaming."""
    job_id = hashlib.md5(str(time.time()).encode()).hexdigest()
    analysis_queues[job_id] = Queue()
    
    data = request.json
    personas = data.get('personas', [])
    product_data = data.get('productData', {})
    
    thread = threading.Thread(target=long_running_analysis, args=(job_id, personas, product_data))
    thread.start()
    
    return jsonify({"job_id": job_id})

@app.route('/stream/<job_id>')
def stream(job_id):
    """The Server-Sent Events endpoint."""
    def event_stream():
        q = analysis_queues.get(job_id)
        if not q:
            # Maybe the job finished very fast or ID is wrong
            yield f"data: {json.dumps({'type': 'error', 'data': 'Invalid Job ID or Job Expired'})}\n\n"
            return

        while True:
            try:
                message = q.get(timeout=30) # Wait for 30s
                if message['type'] == 'done':
                    yield f"data: {json.dumps(message)}\n\n"
                    break
                yield f"data: {json.dumps(message)}\n\n"
            except Exception:
                # Timeout, but the job might still be running. Send a keep-alive or just break.
                yield f"data: {json.dumps({'type': 'error', 'data': 'Stream timed out.'})}\n\n"
                break
                
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    vectorize_database() # Run this once on startup
    app.run(host='0.0.0.0', port=5001, debug=True)