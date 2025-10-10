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
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, request, jsonify, render_template, url_for, Response
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker

# --- Matplotlib Font Setup for Chinese ---
try:
    # 直接在项目根目录寻找字体文件
    font_path = 'simhei.ttf' 
    if not Path(font_path).exists():
        raise FileNotFoundError(f"字体文件未在项目根目录找到: {font_path}")
    
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    plt.rcParams['axes.unicode_minus'] = False
    
    print(f"中文图表字体设置成功: {font_prop.get_name()}")

except Exception as e:
    print(f"警告: 中文字体加载失败，图表中的中文可能显示为方框。错误: {e}")
    # 在加载失败时提供一个备用方案，以防程序崩溃
    plt.rcParams['font.sans-serif'] = ['sans-serif']
    font_prop = fm.FontProperties()

# --- Basic Configuration ---
load_dotenv()
app = Flask(__name__)
CORS(app)
client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("API_URL"))

# --- Vector Search (TF-IDF Version) ---
VECTOR_DB_PATH = Path("vector_db/tfidf_database.pkl")

def row_to_text(row):
    return (
        f"用户画像：性别 {row['性别']}，年龄 {row['年龄']}岁，来自 {row['城市']} 的 {row['职业']}。"
        f"教育程度为 {row['教育程度']}，年收入 {row['收入区间']}。MBTI性格是 {row['MBTI/性格']}。"
        f"饮酒习惯：频率 {row['饮酒频率']}，酒龄 {row['酒龄']}年，偏好 {row['香型']} 白酒，"
        f"心理价位在 {row['白酒价格']}，主要用于 {row['用途']}。"
    )

def vectorize_database():
    if VECTOR_DB_PATH.exists():
        print("TF-IDF 数据库已存在，直接加载。")
        return
    print("未找到 TF-IDF 数据库，开始首次创建...")
    try:
        df = pd.read_excel("panel_data_sample.xlsx").dropna().reset_index(drop=True)
    except FileNotFoundError:
        print("错误: panel_data_sample.xlsx 未找到!"); return
    user_texts = df.apply(row_to_text, axis=1).tolist()
    vectorizer = TfidfVectorizer()
    user_vectors = vectorizer.fit_transform(user_texts)
    VECTOR_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    database = {'vectorizer': vectorizer, 'vectors': user_vectors, 'ids': df['用户ID'].tolist()}
    with open(VECTOR_DB_PATH, 'wb') as f: pickle.dump(database, f)
    print("TF-IDF 数据库创建成功！")

def find_similar_users_knn(persona, top_n=5):
    if not VECTOR_DB_PATH.exists(): return pd.DataFrame(), {}
    with open(VECTOR_DB_PATH, 'rb') as f: database = pickle.load(f)
    vectorizer, user_vectors, user_ids = database['vectorizer'], database['vectors'], database['ids']
    df = pd.read_excel("panel_data_sample.xlsx")
    persona_text = (
        f"用户画像：性别 {persona.get('gender', '')}，年龄 {persona.get('age', '')}岁，来自 {persona.get('city', '')} 的 {persona.get('profession', '')}。"
        f"MBTI性格是 {persona.get('mbti', '')}。其他偏好：教育程度 {persona.get('education', '')}，年收入 {persona.get('income', '')}，"
        f"心理价位 {persona.get('expected_price', '')}，偏好香型 {persona.get('preferred_aroma', '')}。"
    )
    persona_vector = vectorizer.transform([persona_text])
    similarities = cosine_similarity(persona_vector, user_vectors).flatten()
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    matched_user_ids = [user_ids[i] for i in top_indices]
    top_df = df[df['用户ID'].isin(matched_user_ids)].copy()
    top_df['__sort__'] = pd.Categorical(top_df['用户ID'], categories=matched_user_ids, ordered=True)
    top_df = top_df.sort_values('__sort__').drop('__sort__', axis=1)
    
    insights = {
        'top_aroma': top_df['香型'].mode()[0] if not top_df['香型'].mode().empty else "未知",
        'price_band': top_df['白酒价格'].mode()[0] if not top_df['白酒价格'].mode().empty else "未知",
        'typical_usage': top_df['用途'].mode()[0] if not top_df['用途'].mode().empty else "未知"
    }
    return top_df, insights

# --- Real-time Streaming Setup ---
analysis_queues = {}

def long_running_analysis(job_id, personas, product_data):
    q = analysis_queues[job_id]
    try:
        product_description = product_data.get('description')
        base64_image = product_data.get('image').split(',')[1]
        all_results = []

        # Stage 1: Individual Reports
        for i, persona in enumerate(personas):
            top_matches, ai_insights = find_similar_users_knn(persona, top_n=5)
            
            profession = persona.get('profession') or '未填写职业'
            base_lines = ["【基础信息】", f"- 年龄：{persona.get('age', '未填写')} 岁", f"- 性别：{persona.get('gender')}", f"- 常住城市：{persona.get('city', '未填写') or '未填写'}", f"- 职业：{profession}", f"- MBTI：{(persona.get('mbti') or '').upper()}"]
            optional_lines = []
            if persona.get('education'): optional_lines.append(f"- 教育程度：{persona['education']}")
            if persona.get('income'): optional_lines.append(f"- 年收入区间：{persona['income']}")
            if persona.get('drink_frequency'): optional_lines.append(f"- 饮酒频率：{persona['drink_frequency']}")
            if persona.get('drinking_history'): optional_lines.append(f"- 饮酒年限：{persona['drinking_history']} 年")
            if persona.get('expected_price'): optional_lines.append(f"- 心理价位：{persona['expected_price']}")
            if persona.get('preferred_aroma'): optional_lines.append(f"- 偏好香型：{persona['preferred_aroma']}")

            profile_sections = ["\n".join(base_lines)]
            if optional_lines: profile_sections.append("【额外画像线索】\n" + "\n".join(optional_lines))
            
            if not top_matches.empty:
                similarity_lines = ["【相似用户消费记录 (Top 5)】"]
                for _, row in top_matches.iterrows():
                    similarity_lines.append(f"- {row['用户ID']}：{row['年龄']}岁{row['性别']}，{row['城市']}，职业{row['职业']}，MBTI {row['MBTI/性格']}；最近一次购买 {row['品牌名称']}（香型：{row['香型']}，价格：{row['白酒价格']}），用途：{row['用途']}。")
                insight_lines = ["【相似用户购买洞察】", f"- 核心偏好香型：{ai_insights.get('top_aroma', '—')}", f"- 核心价格带：{ai_insights.get('price_band', '—')}", f"- 常见使用场景：{ai_insights.get('typical_usage', '—')}"]
                profile_sections.extend(["\n".join(similarity_lines), "\n".join(insight_lines)])
            else:
                profile_sections.append("【相似用户参考】\n- 数据库中未找到足够的匹配用户，以下分析将基于输入画像进行推断。")
            
            real_user_prompt = "\n\n".join(profile_sections)

            structured_individual_prompt = (
                f"背景：你将代入以下人物画像进行思考。\n人物画像与相似用户消费记录：\n---\n{real_user_prompt}\n---\n\n"
                f"任务：请严格按照思考链分析一款新产品，并以【JSON格式】输出你的完整分析报告。\n\n"
                f"产品文字描述：'{product_description}'\n产品图片已提供。\n\n"
                f"你的输出必须是一个严格的JSON对象，包含以下键：\n"
                f"1. `structured_report`: 一个包含分析文本的对象，必须有`packaging_analysis`, `fit_analysis`, `scenario_analysis`三个键。\n"
                f"2. `radar_scores`: 一个包含匹配度评分的对象，必须有`包装`, `价格`, `香型`, `场景`四个键，每个键的值为0-10的整数。\n"
                f"3. `decision`: 字符串，值为 '购买' 或 '不购买'。\n"
                f"4. `reason`: 字符串，对最终决策的总结性理由。\n\n"
                f"思考链指引（在内心完成，不要输出过程）：\n"
                f"1. 视觉分析：观察产品图片，评估包装设计、风格和档次感。将此思考总结写入 `structured_report.packaging_analysis`。\n"
                f"2. 契合度分析：结合产品描述和视觉分析，对比你的人设（偏好、收入、消费记录），评估产品在香型、价格、品质等方面是否匹配。将此思考总结写入 `structured_report.fit_analysis`。\n"
                f"3. 场景构思：构思1-2个你可能会使用该产品的具体场景。将此思考总结写入 `structured_report.scenario_analysis`。\n"
                f"4. 量化评分：基于以上分析，为`包装`、`价格`、`香型`、`场景`四个维度与你人设的匹配度分别打分（0-10分），填入`radar_scores`。\n"
                f"5. 最终决策：综合所有信息，做出最终`decision`和`reason`。\n"
                f"再次强调重要：输出中任何地方不能包含任何Markdown符号（如`*`、`#`、`-`）。"
            )

            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": [{"type": "text", "text": structured_individual_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}] }],
                max_tokens=2000
            )
            
            analysis_data = json.loads(response.choices[0].message.content)
            
            report_text_obj = analysis_data.get("structured_report", {})
            # 清洗 Markdown 符号
            cleaned_report_text = {k: str(v).replace('*', '').replace('#', '') for k, v in report_text_obj.items()}
            
            radar_scores = analysis_data.get("radar_scores", {})
            decision = analysis_data.get("decision", "不购买")
            reason = analysis_data.get("reason", "").replace('*', '').replace('#', '')

            radar_chart_b64 = generate_chart_base64('radar', radar_scores, "画像-产品匹配度雷达图")

            result_package = {
                "type": "individual_report",
                "data": { 
                    "persona_id": i + 1,
                    "report": cleaned_report_text,
                    "final_decision": {"decision": decision, "reason": reason},
                    "radar_chart": radar_chart_b64,
                    "persona_details": persona,
                    "decision": decision 
                }
            }
            q.put(result_package)
            all_results.append(result_package['data'])
            time.sleep(0.5)

        # Stage 2: Summary Report
        if len(all_results) > 10:
            print(f"数字人数量 ({len(all_results)}) > 10，使用精简版上下文进行汇总。")
            summary_context_lines = ["以下是各模拟用户的核心画像及其最终购买决策："]
            for item in all_results:
                p = item['persona_details']
                persona_summary = (
                    f"- 用户 {item['persona_id']} ({item['decision']}): "
                    f"{p.get('age')}岁{p.get('gender')}，来自{p.get('city', '未知城市')}，职业为{p.get('profession', '未知')}，"
                    f"MBTI为{p.get('mbti', '未知')}，年收入{p.get('income', '未知')}。"
                )
                summary_context_lines.append(persona_summary)
            all_reports_text = "\n".join(summary_context_lines)
        else:
            print(f"数字人数量 ({len(all_results)}) <= 10，使用完整版上下文进行汇总。")
            # For the full report, we need to reconstruct the plain text from the structured report object
            report_texts_for_summary = []
            for item in all_results:
                structured_report = item['report']
                full_text_report = (
                    f"包装视觉评估: {structured_report.get('packaging_analysis', '')}\n"
                    f"产品契合度分析: {structured_report.get('fit_analysis', '')}\n"
                    f"潜在消费场景: {structured_report.get('scenario_analysis', '')}\n"
                    f"【最终决策】{item['final_decision'].get('decision', '')}\n"
                    f"【决策理由】{item['final_decision'].get('reason', '')}"
                )
                report_texts_for_summary.append(f"--- 用户 {item['persona_id']} ({item['decision']}) 报告 ---\n{full_text_report}")
            all_reports_text = "\n\n".join(report_texts_for_summary)

        structured_summary_prompt = (
            f"你是一位顶级的市场研究总监，你的任务是基于以下 {len(all_results)} 份模拟用户数据，撰写一份深刻、专业、结构化的综合市场分析报告...\n" # (保持之前的详细prompt)
        )
        summary_response = client.chat.completions.create(model="gpt-4-turbo", messages=[{"role": "user", "content": structured_summary_prompt}], max_tokens=1500)
        summary_report = summary_response.choices[0].message.content.replace('*', '').replace('#', '')
        q.put({"type": "summary_report", "data": summary_report})

        # Stage 3: Chart & Table Analysis
        def generate_and_stream_chart_ux(chart_id, title, chart_type, data):
            chart_b64 = generate_chart_base64(chart_type, data, title)
            q.put({"type": "chart_and_table", "data": {"id": chart_id, "title": title, "chart": chart_b64, "table": data}})
            analysis = get_ai_analysis_for_table(title, data)
            q.put({"type": "table_analysis", "data": {"id": chart_id, "analysis": analysis}})
            time.sleep(0.5)

        def get_age_group(age):
            try: age = int(age)
            except (ValueError, TypeError): return "未知年龄"
            if age < 25: return "25岁以下"
            if age < 30: return "25-29岁"
            if age < 35: return "30-34岁"
            if age < 40: return "35-39岁"
            if age < 45: return "40-44岁"
            if age < 50: return "45-49岁"
            return "50岁及以上"
        
        buying_results = [r for r in all_results if r['decision'] == '购买']
        non_buying_results = [r for r in all_results if r['decision'] == '不购买']
        
        income_order = ["10万以下", "10-20万", "20-50万", "50万以上"]
        
        generate_and_stream_chart_ux("overall", "总体购买意向比例", "pie", {'购买': len(buying_results), '不购买': len(non_buying_results)})

        if buying_results:
            buyer_gender_data = dict(Counter(p['persona_details']['gender'] for p in buying_results))
            if buyer_gender_data: generate_and_stream_chart_ux("buyer_gender", "购买用户性别分布", "pie", buyer_gender_data)
            buyer_city_data = dict(Counter(p['persona_details']['city'] for p in buying_results if p['persona_details']['city']))
            if buyer_city_data: generate_and_stream_chart_ux("buyer_city", "购买用户城市分布", "bar", buyer_city_data)
            buyer_age_data = dict(Counter(get_age_group(p['persona_details']['age']) for p in buying_results))
            if buyer_age_data: generate_and_stream_chart_ux("buyer_age", "购买用户年龄分布", "bar", buyer_age_data)
            buyer_mbti_data = dict(Counter(p['persona_details']['mbti'] for p in buying_results if p['persona_details']['mbti']))
            if buyer_mbti_data: generate_and_stream_chart_ux("buyer_mbti", "购买用户MBTI分布", "pie", buyer_mbti_data)
            income_counts = Counter(p['persona_details']['income'] for p in buying_results if p['persona_details']['income'])
            if income_counts:
                sorted_income = {key: income_counts[key] for key in income_order if key in income_counts}
                if sorted_income: generate_and_stream_chart_ux("buyer_income", "购买用户收入分布", "pie", sorted_income)

        if non_buying_results:
            nonbuyer_gender_data = dict(Counter(p['persona_details']['gender'] for p in non_buying_results))
            if nonbuyer_gender_data: generate_and_stream_chart_ux("nonbuyer_gender", "未购买用户性别分布", "pie", nonbuyer_gender_data)
            
    except Exception as e:
        print(f"Analysis thread error: {e}")
        q.put({"type": "error", "data": str(e)})
    finally:
        q.put({"type": "done"})
        if job_id in analysis_queues: del analysis_queues[job_id]
        
# --- Chart Generation and AI Analysis Functions ---
def generate_chart_base64(chart_type, data, title):
    if not data or (chart_type == 'pie' and sum(data.values()) == 0): return None
    
    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    plt.title(title, pad=20, fontsize=16)

    if chart_type == 'pie':
        pie_labels = [f"{key} ({value}人)" for key, value in data.items()]
        ax.pie(data.values(), labels=pie_labels, autopct='%1.1f%%', startangle=120, textprops={'fontsize': 10})
        ax.axis('equal')

    elif chart_type in ['line', 'bar']:
        ax.bar(data.keys(), data.values(), color='#4A90E2')
        ax.set_ylabel("人数", fontsize=12)
        plt.xticks(rotation=30, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_ylim(bottom=0)

    # --- 雷达图的绘制逻辑 ---

def generate_chart_base64(chart_type, data, title):
    if not data or (chart_type == 'pie' and sum(data.values()) == 0): 
        return None
    
    fig = plt.figure(figsize=(9, 7))
    
    if chart_type == 'radar':
        ax = fig.add_subplot(111, polar=True)
    else:
        ax = fig.add_subplot(111)
        ax.set_title(title, pad=20, fontsize=16)

    if chart_type == 'pie':
        pie_labels = [f"{key} ({value}人)" for key, value in data.items()]
        ax.pie(data.values(), labels=pie_labels, autopct='%1.1f%%', startangle=120, textprops={'fontsize': 10})
        ax.axis('equal')

    elif chart_type in ['line', 'bar']:
        ax.bar(data.keys(), data.values(), color='#4A90E2')
        ax.set_ylabel("人数", fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_ylim(bottom=0)

    elif chart_type == 'radar':
        labels = list(data.keys())
        values = list(data.values())
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        # 重新获取坐标轴，确保状态干净
        ax = plt.subplot(111, polar=True)

        # --- 最终修正 ---

        # 1. 绘制数据区域
        ax.plot(angles, values, color='#6E0F1A', linewidth=3, zorder=3)
        ax.fill(angles, values, color='#6E0F1A', alpha=0.25, zorder=2)
        
        # 2. 显示所有内部网格线（环形和放射状）
        ax.grid(color='lightgrey', linestyle='--', linewidth=0.7)
        
        # 3. 显示外圈边框
        ax.spines['polar'].set_visible(True)
        ax.spines['polar'].set_color('lightgrey')
        
        # 4. 设置并显示环形刻度（分数）
        ax.set_yticks(np.arange(0, 11, 2))
        ax.set_yticklabels(["", "2", "4", "6", "8", "10"], color="grey", size=15)
        ax.set_ylim(0, 10)
        
        # 5. 设置并显示维度标签（如“价格”、“香型”）
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=19, fontproperties=font_prop)
        ax.tick_params(axis='x', pad=25)

    plt.tight_layout(pad=2.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, dpi=200)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def get_ai_analysis_for_table(table_title, table_data):
    if not table_data: return "无数据可供分析。"
    prompt = f"你是一位数据分析师。以下是关于 '{table_title}' 的数据：{json.dumps(table_data, ensure_ascii=False)}。请用一句话给出最核心的商业洞察。"
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], max_tokens=150)
        return response.choices[0].message.content
    except Exception as e: return f"AI洞察生成失败: {e}"

# --- Flask Routes ---
@app.route('/generate_report', methods=['POST'])
def generate_report_start():
    job_id = hashlib.md5(str(time.time()).encode()).hexdigest()
    analysis_queues[job_id] = Queue()
    data = request.json
    thread = threading.Thread(target=long_running_analysis, args=(job_id, data.get('personas', []), data.get('productData', {})))
    thread.start()
    return jsonify({"job_id": job_id})

@app.route('/stream/<job_id>')
def stream(job_id):
    def event_stream():
        q = analysis_queues.get(job_id)
        if not q: yield f"data: {json.dumps({'type': 'error', 'data': '无效的任务ID'})}\n\n"; return
        while True:
            try:
                message = q.get(timeout=120)
                yield f"data: {json.dumps(message)}\n\n"
                if message['type'] == 'done': break
            except Exception:
                yield f"data: {json.dumps({'type': 'error', 'data': '数据流超时或任务已结束'})}\n\n"; break
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    vectorize_database()
    app.run(host='0.0.0.0', port=5001, debug=True)