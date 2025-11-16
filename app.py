import os
import io
import json
import base64
import random
import hashlib
import pickle
import threading
import time
import math
import re
import ast
from queue import Queue
from pathlib import Path
from collections import Counter

from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 💡 新增后台功能 Imports ---
from flask import (
    Flask, request, jsonify, render_template, url_for, Response,
    session, redirect, send_from_directory
)
from werkzeug.security import safe_join
# --- 结束新增 ---

from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker

# --- Matplotlib Font Setup (无变动) ---
font_candidates = [
    Path('simhei.ttf'), Path('fonts/simhei.ttf'),
    Path('/System/Library/Fonts/PingFang.ttc'), Path('/System/Library/Fonts/PingFang SC.ttc'),
    Path('C:/Windows/Fonts/simhei.ttf'), Path('C:/Windows/Fonts/msyh.ttc'),
]
fallback_families = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'Arial Unicode MS']

def _find_font_from_families(families):
    for family in families:
        try:
            located = fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
            if Path(located).exists(): return Path(located)
        except Exception: continue
    return None

font_path = next((p for p in font_candidates if p.exists()), _find_font_from_families(fallback_families))
if font_path:
    try:
        fm.fontManager.addfont(str(font_path))
        font_prop = fm.FontProperties(fname=str(font_path))
        resolved_name = font_prop.get_name()
        plt.rcParams['font.sans-serif'] = [resolved_name, *fallback_families]
        plt.rcParams['axes.unicode_minus'] = False
        print(f"中文图表字体设置成功: {resolved_name} ({font_path})")
    except Exception as exc:
        print(f"警告: 中文字体加载失败: {exc}")
else:
    print("警告: 未找到可用的中文字体文件。")

# --- Basic Configuration ---
load_dotenv()
backend_env = Path('backend/.env')
if backend_env.exists(): load_dotenv(backend_env, override=True)
else: print("警告: 未找到 backend/.env，将使用默认环境变量。")

api_key = os.getenv("API_KEY")
api_url = os.getenv("API_URL")
if not api_key: raise RuntimeError("未检测到 API_KEY，请在 .env 或 backend/.env 中进行配置。")

app = Flask(__name__)
CORS(app)
client = OpenAI(api_key=api_key, base_url=api_url)

# --- 💡 新增：后台管理配置 ---
# 警告：确保在 .env 文件中设置了一个强 SECRET_KEY
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key_replace_in_production")
ARCHIVE_DIR = Path("archived_reports")
ARCHIVE_DIR.mkdir(exist_ok=True) # 确保归档目录存在

ADMIN_USERNAME = "guanliyuan@qq.com"
ADMIN_PASSWORD = "guanliyuan123mima"
# --- 结束新增 ---


# --- Data Loading and Preprocessing (无变动) ---
REAL_DATA_DF = pd.DataFrame()
UNIQUE_CITIES = []

def load_and_preprocess_data():
    global REAL_DATA_DF, UNIQUE_CITIES
    
    data_path = Path("real_survey_data.xlsx")
    if not data_path.exists():
        print(f"错误: 未在项目根目录找到 'real_survey_data.xlsx'。")
        print("请确保你已将原始 Excel 文件复制到项目目录并重命名。")
        return

    print("开始加载并预处理真实调研数据 (Excel)...")
    
    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        print(f"读取 Excel 文件时出错: {e}")
        print("请确保 'openpyxl' 已经通过 'pip install openpyxl' 安装。")
        return

    column_mapping = {
        '2. 性别': '性别',
        '3. 年龄': '年龄',
        '4. 职业': '职业',
        '5. 教育程度': '教育程度',
        '6. 收入区间': '收入区间',
        '7. 城市': '城市', # This is the column we need to clean
        '8. MBTI（按照您已知的进行选择）       外向(E) 与 内向(I): 描述能量获取方式，外向型从社交中获取能量，内向型从独处中恢复能量    ': 'mbti_ei',
        '   感觉(S) 与 直觉(N): 描述信息处理方式，感觉型关注细节和现实，直觉型关注整体和未来可能性': 'mbti_sn',
        '   思考(T) 与 情感(F): 描述决策方式，思考型依赖逻辑分析，情感型考虑个人价值观和他人感受。': 'mbti_tf',
        '   判断(J) 与 知觉(P): 描述生活态度，判断型偏好有计划有条理，知觉型更灵活开放。': 'mbti_jp',
        '9. 饮酒频率': '饮酒频率',
        '10. 饮酒历史（酒龄）': '酒龄',
        '11. 期望单瓶价格区间': '白酒价格',
        '12. 香型类别': '香型',
        '22. 主要用途': '用途'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # --- *** CITY CLEANING STEP *** ---
    df['城市'] = df['城市'].astype(str).apply(lambda x: x.split('-')[0] if pd.notna(x) else '未知')
    # --- *** END CITY CLEANING *** ---

    df['用户ID'] = [f"User_{i+1:04d}" for i in range(len(df))]
    
    # *** 🔴 旧的 MBTI 处理代码从这里移除 ***
    
    df['香型'] = df['香型'].astype(str).apply(lambda x: x.split(' ')[0] if pd.notna(x) else '未知')
    
    # --- 这是安全检查循环 ---
    # 我们把 mbti_ei, mbti_sn, mbti_tf, mbti_jp 也加入检查
    final_columns = list(set(column_mapping.values()) | {'用户ID', 'MBTI/性格', 'mbti_ei', 'mbti_sn', 'mbti_tf', 'mbti_jp'})
    
    for col in final_columns:
        if col not in df.columns:
            print(f"警告：数据中缺少必需的列 '{col}'，将用 '未知' 填充。")
            # 如果重命名失败, 这里会创建 'mbti_ei' 等列并填充 "未知"
            df[col] = "未知"
            
    # --- 🟢 新的 MBTI 处理代码粘贴到这里 ---
    # 现在我们可以安全地处理它们了, 即使重命名失败, 它们的值也是 "未知"
    df['mbti_ei'] = df['mbti_ei'].astype(str).str.extract(r'\((\w)\)').fillna('I')
    df['mbti_sn'] = df['mbti_sn'].astype(str).str.extract(r'\((\w)\)').fillna('S')
    df['mbti_tf'] = df['mbti_tf'].astype(str).str.extract(r'\((\w)\)').fillna('T')
    df['mbti_jp'] = df['mbti_jp'].astype(str).str.extract(r'\((\w)\)').fillna('P')
    
    # 这一行也必须在上面四行之后
    df['MBTI/性格'] = df['mbti_ei'] + df['mbti_sn'] + df['mbti_tf'] + df['mbti_jp']
    # --- 结束修改 ---
            
    # 从 final_columns 中移除临时的 mbti 组件, 只保留最终的 'MBTI/性格'
    final_columns_for_df = list(set(column_mapping.values()) | {'用户ID', 'MBTI/性格'})
    df = df[final_columns_for_df]
    
    # Fill any remaining NaNs after cleaning steps
    df.fillna('未知', inplace=True) 
    
    df = df.reset_index(drop=True)
    
    REAL_DATA_DF = df
    UNIQUE_CITIES = sorted(REAL_DATA_DF['城市'].astype(str).unique().tolist())
    
    if '未知' in UNIQUE_CITIES:
        UNIQUE_CITIES.remove('未知')

    print(f"真实数据加载成功！共处理 {len(REAL_DATA_DF)} 条有效用户数据。")
    print(f"发现 {len(UNIQUE_CITIES)} 个独立城市 (已清理)。")


# --- Vector Search (无变动) ---
VECTOR_DB_PATH = Path("vector_db/tfidf_database_real.pkl")

def row_to_text_real_data(row):
    return (
        f"用户画像：性别 {row.get('性别', '未知')}，年龄段 {row.get('年龄', '未知')}，来自 {row.get('城市', '未知')} 的 {row.get('职业', '未知')}。"
        f"教育程度为 {row.get('教育程度', '未知')}，年收入 {row.get('收入区间', '未知')}。MBTI性格是 {row.get('MBTI/性格', '未知')}。"
        f"饮酒习惯：频率 {row.get('饮酒频率', '未知')}，酒龄 {row.get('酒龄', '未知')}，偏好 {row.get('香型', '未知')} 白酒，"
        f"心理价位在 {row.get('白酒价格', '未知')}，主要用于 {row.get('用途', '未知')}。"
    )

def vectorize_database():
    if VECTOR_DB_PATH.exists():
        print("TF-IDF 真实数据向量库已存在，直接加载。")
        return
    if REAL_DATA_DF.empty:
        print("错误: 真实数据未能加载，无法创建向量库。")
        return
    print("未找到真实数据向量库，开始首次创建...")
    user_texts = REAL_DATA_DF.apply(row_to_text_real_data, axis=1).tolist()
    vectorizer = TfidfVectorizer()
    user_vectors = vectorizer.fit_transform(user_texts)
    VECTOR_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    database = {'vectorizer': vectorizer, 'vectors': user_vectors, 'ids': REAL_DATA_DF['用户ID'].tolist()}
    with open(VECTOR_DB_PATH, 'wb') as f:
        pickle.dump(database, f)
    print("TF-IDF 真实数据向量库创建成功！")

def find_similar_users_knn(persona, top_n=5):
    if not VECTOR_DB_PATH.exists() or REAL_DATA_DF.empty:
        return pd.DataFrame(), {}
    with open(VECTOR_DB_PATH, 'rb') as f:
        database = pickle.load(f)
    vectorizer, user_vectors, user_ids = database['vectorizer'], database['vectors'], database['ids']
    persona_text = (
        f"用户画像：性别 {persona.get('gender', '')}，年龄 {persona.get('age', '')}岁，来自 {persona.get('city', '')} 的 {persona.get('profession', '')}。"
        f"MBTI性格是 {persona.get('mbti', '')}。其他偏好：教育程度 {persona.get('education', '')}，年收入 {persona.get('income', '')}，"
        f"心理价位 {persona.get('expected_price', '')}，偏好香型 {persona.get('preferred_aroma', '')}。"
    )
    persona_vector = vectorizer.transform([persona_text])
    similarities = cosine_similarity(persona_vector, user_vectors).flatten()
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    matched_user_ids = [user_ids[i] for i in top_indices]
    top_df = REAL_DATA_DF[REAL_DATA_DF['用户ID'].isin(matched_user_ids)].copy()
    top_df['__sort__'] = pd.Categorical(top_df['用户ID'], categories=matched_user_ids, ordered=True)
    top_df = top_df.sort_values('__sort__').drop('__sort__', axis=1)
    insights = {
        'top_aroma': top_df['香型'].mode()[0] if not top_df['香型'].mode().empty else "未知",
        'price_band': top_df['白酒价格'].mode()[0] if not top_df['白酒价格'].mode().empty else "未知",
        'typical_usage': top_df['用途'].mode()[0] if not top_df['用途'].mode().empty else "未知"
    }
    return top_df, insights

# --- 💡 新增：HTML 归档辅助函数 ---

def _get_report_download_css():
    """返回用于保存 HTML 报告的 CSS 字符串"""
    return """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: auto; color: #333; }
        h1, h2, h3, h4, h5 { color: #6E0F1A; font-family: "Noto Serif SC", serif; }
        h1 { font-size: 24px; } h4 { font-size: 18px; } h5 { font-size: 16px; margin-bottom: 5px; }
        img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        p { margin-top: 0; }
        hr { border: 0; border-top: 1px dashed #ccc; margin: 20px 0; }
        .report-header-actions { border-bottom: 1px solid #eee; padding-bottom: 10px; }
        .persona-report { border-bottom: 2px solid #6E0F1A; margin-bottom: 20px; padding-bottom: 20px; }
        .report-details-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: center; margin: 16px 0; }
        .final-decision-section { border-top: 1px dashed #ccc; padding-top: 10px; margin-top: 10px; }
        .decision-buy { color: green; font-weight: bold; }
        .decision-nobuy { color: red; font-weight: bold; }
        .prose p { margin-bottom: 1em; }
        .charts-section img { max-width: 450px; display: block; margin: 10px auto; }
        .kv { border: 1px solid #ddd; border-radius: 8px; overflow: hidden; margin: 10px 0; }
        .kv .row { display: grid; grid-template-columns: 1fr 90px; border-bottom: 1px solid #eee; }
        .kv .row:last-child { border-bottom: none; }
        .kv .row div { padding: 8px 10px; }
        .note { background: #fef9e7; border: 1px solid #f7dc6f; border-radius: 4px; padding: 10px; font-size: 14px; }
        .muted { color: #7B6363; }
        .radar-chart-container { text-align: center; }
        .radar-chart-container h5 { margin-bottom: 8px; }
        .radar-chart-container img { max-width: 450px; margin: 0 auto; }
        @media (max-width: 600px) {
            .report-details-grid { grid-template-columns: 1fr; }
        }
    </style>
    """

def _save_report_to_html(content, filename_path, title):
    """将 HTML 内容和 CSS 保存到指定路径的文件中"""
    css = _get_report_download_css()
    html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        {css}
    </head>
    <body>
        <h1>{title}</h1>
        {content}
    </body>
    </html>
    """
    try:
        with open(filename_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"成功保存归档文件: {filename_path}")
    except Exception as e:
        print(f"错误：保存归档文件 {filename_path} 失败: {e}")

def _generate_individual_report_html(data):
    """根据独立报告的数据包生成其 HTML 字符串"""
    persona = data.get('persona_details', {})
    decision = data.get('decision', '未知')
    decision_class = 'decision-buy' if decision == '购买' else 'decision-nobuy'
    report_content = data.get('report', {})
    final_decision = data.get('final_decision', {})
    radar_chart_b64 = data.get('radar_chart')
    persona_id = data.get('persona_id', 'N/A')

    intro_parts = [
        f"{persona.get('age', '未知')}岁",
        persona.get('gender', '未知性别'),
        f"来自{persona.get('city', '未知城市')}",
        f"职业: {persona.get('profession', '未指定')}",
        f"教育: {persona.get('education', '未指定')}",
        f"年收入: {persona.get('income', '未指定')}",
        f"心理价位: {persona.get('expected_price', '未指定')}",
        f"MBTI: {persona.get('mbti', '未指定')}",
        f"饮酒频率: {persona.get('drink_frequency', '未指定')}",
        f"酒龄: {persona.get('drinking_history', 'N/A')}年",
        f"偏好香型: {persona.get('preferred_aroma', '未指定')}"
    ]
    persona_intro = '，'.join(intro_parts) + '。'
    
    radar_html = (
        f'<div class="radar-chart-container">'
        f'<h5>画像-产品匹配度雷达图</h5>'
        f'<img src="data:image/png;base64,{radar_chart_b64}" alt="雷达图">'
        f'</div>'
    ) if radar_chart_b64 else ''

    return f"""
    <div class="persona-report" id="persona-report-{persona_id}">
        <div class="report-header-actions">
            <h4 class="h4">画像 {persona_id} 独立报告</h4>
        </div>
        <p class="muted">画像简介：{persona_intro}</p>
        <div class="report-details-grid">
            <div class="report-text-sections">
                <div class="report-section"><h5>包装视觉评估</h5><p>{report_content.get('packaging_analysis', "AI未提供此项分析。")}</p></div>
                <div class="report-section"><h5>产品契合度分析</h5><p>{report_content.get('fit_analysis', "AI未提供此项分析。")}</p></div>
                <div class="report-section"><h5>潜在消费场景</h5><p>{report_content.get('scenario_analysis', "AI未提供此项分析。")}</p></div>
            </div>
            {radar_html}
        </div>
        <div class="final-decision-section">
            <p><strong class="{decision_class}">【最终决策】</strong> {final_decision.get('decision', '未明确')}</p>
            <p><strong>【决策理由】</strong> {final_decision.get('reason', 'AI未提供理由。')}</p>
        </div>
    </div>
    """
# --- 结束新增 ---


# --- Real-time Streaming, Report Generation, Charting (修改) ---
def allocate_counts_from_ratio(ratio_map, total):
    if total <= 0 or not ratio_map: return {}
    items = []
    for key, ratio in ratio_map.items():
        try: ratio_value = float(ratio) if ratio is not None else 0.0
        except (TypeError, ValueError): ratio_value = 0.0
        str_key = str(key) if key is not None else "None" 
        exact = ratio_value * total / 100
        base = math.floor(exact)
        remainder = exact - base
        items.append([str_key, base, remainder])
    assigned = sum(item[1] for item in items)
    remaining = max(0, total - assigned)
    items.sort(key=lambda entry: entry[2], reverse=True)
    idx = 0
    while remaining > 0 and items:
        items[idx % len(items)][1] += 1
        idx += 1
        remaining -= 1
    return {key: count for key, count, _ in items}

analysis_queues = {}

def long_running_analysis(job_id, personas, product_data, persona_file=None):
    q = analysis_queues[job_id]
    
    # --- 💡 新增：归档设置 ---
    beijing_tz = timezone(timedelta(hours=8))
    beijing_now = datetime.now(timezone.utc).astimezone(beijing_tz)
    timestamp = beijing_now.strftime("%Y%m%d_%H%M%S")
    report_archive_dir = ARCHIVE_DIR / timestamp
    report_archive_dir.mkdir(parents=True, exist_ok=True)
    
    # 用于收集 HTML 以便最后保存
    all_individual_reports_html = []
    all_charts_html = []
    summary_report_text = "" # 用于保存综合报告文本
    # --- 结束新增 ---

    try:
        persona_list = personas or []
        if persona_file:
            try:
                with open(persona_file, 'r', encoding='utf-8') as f:
                    file_personas = json.load(f)
                    if isinstance(file_personas, list) and file_personas:
                        persona_list = file_personas
            except Exception as exc:
                print(f"警告: 无法从 {persona_file} 读取画像：{exc}，继续使用请求中的数据。")
        if not persona_list:
            raise ValueError("未提供任何画像数据。")

        # --- 💡 归档：保存 personas.json ---
        personas_file_path = report_archive_dir / "personas.json"
        try:
            with open(personas_file_path, 'w', encoding='utf-8') as f:
                json.dump(persona_list, f, ensure_ascii=False, indent=2)
            print(f"[{job_id}] 成功归档: {personas_file_path}")
        except Exception as e:
            print(f"[{job_id}] 警告: 归档 personas.json 失败: {e}")
        # --- 结束归档 ---

        product_description = (product_data or {}).get('description', '')
        image_payload = (product_data or {}).get('image', '')
        base64_image = ''
        if isinstance(image_payload, str) and ',' in image_payload:
            base64_image = image_payload.split(',', 1)[1]
        elif isinstance(image_payload, str):
            base64_image = image_payload
        
        all_decision_data = []

        # --- Stage 1: 批量决策 (无变动) ---
        batch_size = 5
        total_personas = len(persona_list)
        print(f"[{job_id}] Progress: 启动快速决策引擎...总共 {total_personas} 位数字人。")

        for i in range(0, total_personas, batch_size):
            batch_personas = persona_list[i:i + batch_size]
            persona_summaries_for_prompt = []
            
            for j, p in enumerate(batch_personas):
                persona_id = i + j + 1
                top_matches, ai_insights = find_similar_users_knn(p, top_n=3)
                
                similarity_lines = ["  【相似真实用户参考 (Top 3)】:"]
                if not top_matches.empty:
                    for _, row in top_matches.iterrows():
                        similarity_lines.append(
                            f"  - 真实用户 ({row['城市']}, {row['职业']}) 偏好: {row['香型']}, 价位: {row['白酒价格']}, 用途: {row['用途']}。"
                        )
                    similarity_lines.append(f"  - 核心洞察: 偏好{ai_insights.get('top_aroma', '—')}, 价位{ai_insights.get('price_band', '—')}")
                else:
                    similarity_lines.append("  - 数据库中未找到相似真实用户。")
                
                similar_users_text = "\n".join(similarity_lines)

                summary = (
                    f"--- 画像 ID {persona_id} ---\n"
                    f"  【数字人画像】: {p.get('age')}岁{p.get('gender')}，来自{p.get('city', '未知')}的{p.get('profession', '未知')}。"
                    f"MBTI {p.get('mbti', '未知')}，年收入{p.get('income', '未知')}，偏好{p.get('preferred_aroma', '未知')}香型，"
                    f"心理价位{p.get('expected_price', '未知')}。\n"
                    f"{similar_users_text}"
                )
                persona_summaries_for_prompt.append(summary)

            batch_decision_prompt = (
                "你是一位高效的市场分析师，你的任务是快速判断不同消费者对一款新产品的购买意向。\n\n"
                f"产品文字描述：'{product_description}'\n产品图片已提供。\n\n"
                f"以下是本批次的 {len(batch_personas)} 位消费者画像，以及对应的几组与他们相似的【真实用户数据参考】：\n\n" + "\n\n".join(persona_summaries_for_prompt) + "\n\n"
                "任务：请为每一位消费者做出独立的购买决策。你的决策必须【重点参考】每个画像下方提供的【相似真实用户参考】数据，并结合数字人画像和产品信息进行判断。\n"
                "你的输出必须是一个严格的JSON对象，包含一个键 `decisions`，其值为一个数组。\n"
                "数组中的每个对象都必须包含两个键：`persona_id` (整数) 和 `decision` (字符串 '购买' 或 '不购买')。\n"
                "例如: {\"decisions\": [{\"persona_id\": 1, \"decision\": \"购买\"}, {\"persona_id\": 2, \"decision\": \"不购买\"}]}"
            )
            
            message_content = [{"type": "text", "text": batch_decision_prompt}]
            if base64_image:
                message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": message_content}],
                max_tokens=1000
            )
            
            batch_decisions = json.loads(response.choices[0].message.content).get('decisions', [])
            
            for decision_info in batch_decisions:
                persona_id = decision_info.get('persona_id')
                decision = decision_info.get('decision')
                if persona_id and decision and 1 <= persona_id <= total_personas:
                    all_decision_data.append({
                        "persona_details": persona_list[persona_id - 1],
                        "persona_id": persona_id,
                        "decision": decision
                    })
            print(f"[{job_id}] Progress: 快速决策完成: {len(all_decision_data)} / {total_personas} 位数字人。")
            time.sleep(0.5)

        # --- Stage 2: 生成整体市场报告 ---
        print(f"[{job_id}] Progress: 所有决策已完成，正在生成整体市场分析报告...")
        
        summary_context_lines = ["以下是各模拟用户的核心画像及其最终购买决策："]
        for item in all_decision_data:
            p = item['persona_details']
            persona_summary = (
                f"- 用户 {item['persona_id']} ({item['decision']}): "
                f"{p.get('age')}岁{p.get('gender')}，来自{p.get('city', '未知城市')}，职业为{p.get('profession', '未知')}，"
                f"MBTI为{p.get('mbti', '未知')}，年收入{p.get('income', '未知')}。"
            )
            summary_context_lines.append(persona_summary)
        all_reports_text = "\n".join(summary_context_lines)

        structured_summary_prompt = (
            f"你是一位顶级的市场研究总监，你的任务是基于以下 {len(all_decision_data)} 份模拟用户数据，撰写一份深刻、专业、结构化的综合市场分析报告。\n\n"
            f"重要：输出中任何地方不能包含任何Markdown符号（如`*`、`#`、`-`）。"
            f"【原始数据】\n{all_reports_text}\n\n"
            f"请严格按照以下大纲撰写你的综合分析报告，确保内容深刻、逻辑清晰、语言专业。报告中不能包含任何Markdown符号。\n\n"
            f"1. 核心洞察与购买意向\n- 首先，精确计算并明确展示总体购买率（购买人数/总人数）。\n- 提炼出本次模拟中最关键、最核心的市场洞察是什么。\n\n"
            f"2. 关键购买驱动力分析\n- 深入总结吸引用户做出“购买”决策的核心因素。请从产品属性、包装设计、品牌定位和价格感知等多个维度进行剖析。\n\n"
            f"3. 主要购买壁垒剖析\n- 同样，深入剖析导致用户做出“不购买”决策的关键障碍。分析这些障碍是源于产品自身、价格、包装，还是与目标用户的核心需求存在错位。\n\n"
            f"4. 综合结论与市场策略建议\n- 对产品的市场潜力给出一个简洁有力的综合结论。\n- 基于以上所有分析，为该产品的市场策略提供1-2条具体的、可执行的建议。"
            f"再次强调重要：输出中任何地方不能包含任何Markdown符号（如`*`、`#`、`-`）。"
        )
        
        summary_response = client.chat.completions.create(model="gpt-4-turbo", messages=[{"role": "user", "content": structured_summary_prompt}], max_tokens=1500)
        
        # --- 💡 归档：保存综合报告文本 ---
        summary_report_text = summary_response.choices[0].message.content.replace('*', '').replace('#', '')
        # --- 结束归档 ---

        q.put({"type": "summary_report", "data": summary_report_text})
        print(f"[{job_id}] Progress: 整体报告已生成并发送。")
        time.sleep(1)


        # --- Stage 3: 图表和表格分析 (修改) ---
        print(f"[{job_id}] Progress: 开始生成图表分析...")
        
        def generate_and_stream_chart_ux(chart_id, title, chart_type, data):
            chart_b64 = generate_chart_base64(chart_type, data, title)
            q.put({"type": "chart_and_table", "data": {"id": chart_id, "title": title, "chart": chart_b64, "table": data}})
            
            analysis = get_ai_analysis_for_table(title, data)
            q.put({"type": "table_analysis", "data": {"id": chart_id, "analysis": analysis}})
            
            # --- 💡 归档：保存图表 HTML ---
            try:
                table_rows = "".join([f'<div class="row"><div>{key}</div><div>{value}</div></div>' for key, value in data.items()])
                chart_img_html = f'<img src="data:image/png;base64,{chart_b64}" alt="{title}" style="max-width:100%; border-radius:12px; margin-bottom:12px; border:1px solid #E9E3DA;">' if chart_b64 else ''
                chart_block_html = f"""
                <div id="chart-container-{chart_id}" style="margin-bottom: 20px; border-bottom: 1px dashed #ccc; padding-bottom: 20px;">
                    <h4 class="h4">{title}</h4>
                    {chart_img_html}
                    <div class="kv">{table_rows}</div>
                    <div class="note"><strong>AI洞察：</strong> {analysis}</div>
                </div>
                """
                all_charts_html.append(chart_block_html)
            except Exception as e:
                print(f"[{job_id}] 警告: 归档图表 '{title}' HTML 失败: {e}")
            # --- 结束归档 ---
                
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
        
        buying_results = [r for r in all_decision_data if r['decision'] == '购买']
        non_buying_results = [r for r in all_decision_data if r['decision'] == '不购买']
        
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
            if buyer_mbti_data: generate_and_stream_chart_ux("buyer_mmbti", "购买用户MBTI分布", "pie", buyer_mbti_data)
            income_counts = Counter(p['persona_details']['income'] for p in buying_results if p['persona_details']['income'])
            if income_counts:
                sorted_income_keys = sorted(income_counts.keys())
                sorted_income = {key: income_counts[key] for key in sorted_income_keys}
                if sorted_income: generate_and_stream_chart_ux("buyer_income", "购买用户收入分布", "pie", sorted_income)

        if non_buying_results:
            nonbuyer_gender_data = dict(Counter(p['persona_details']['gender'] for p in non_buying_results))
            if nonbuyer_gender_data: generate_and_stream_chart_ux("nonbuyer_gender", "未购买用户性别分布", "pie", nonbuyer_gender_data)
        
        print(f"[{job_id}] Progress: 图表分析完成。")


        # --- Stage 4: 填充详细的个人报告 (修改) ---
        print(f"[{job_id}] Progress: 正在后台补充每个数字人的详细分析理由...")
        
        MAX_RETRIES = 10
        for item in all_decision_data:
            persona = item['persona_details']
            persona_id = item['persona_id']
            pre_determined_decision = item['decision']
            
            analysis_data = None
            for attempt in range(MAX_RETRIES):
                try:
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
                            similarity_lines.append(f"- {row['用户ID']}：{row['年龄']} {row['性别']}，{row['城市']}，职业{row['职业']}，MBTI {row['MBTI/性格']}；偏好{row['香型']}，价位{row['白酒价格']}，用途：{row['用途']}。")
                        insight_lines = ["【相似用户购买洞察】", f"- 核心偏好香型：{ai_insights.get('top_aroma', '—')}", f"- 核心价格带：{ai_insights.get('price_band', '—')}", f"- 常见使用场景：{ai_insights.get('typical_usage', '—')}"]
                        profile_sections.extend(["\n".join(similarity_lines), "\n".join(insight_lines)])
                    else:
                        profile_sections.append("【相似用户参考】\n- 数据库中未找到足够的匹配用户，以下分析将基于输入画像进行推断。")
                    
                    real_user_prompt = "\n\n".join(profile_sections)
                    
                    structured_individual_prompt = (
                        f"背景：你将代入以下人物画像进行思考。\n人物画像与相似用户消费记录：\n---\n{real_user_prompt}\n---\n\n"
                        f"产品文字描述：'{product_description}'\n产品图片已提供。\n\n"
                        f"任务：已知该用户的最终决策是 ‘{pre_determined_decision}’。请围绕这个既定决策，完成一份详细的分析报告。你的所有分析、评分和理由都必须与 ‘{pre_determined_decision}’ 这一最终结果保持逻辑一致。\n\n"
                        f"你的输出必须是一个严格的JSON对象，包含以下键：\n"
                        f"1. `structured_report`: 一个包含分析文本的对象，必须有`packaging_analysis`, `fit_analysis`, `scenario_analysis`三个键。\n"
                        f"2. `radar_scores`: 一个包含匹配度评分的对象，必须有`包装`, `价格`, `香型`, `场景`四个键，每个键的值为0-10的整数。\n"
                        f"3. `decision`: 字符串，其值必须是 '{pre_determined_decision}'。\n"
                        f"4. `reason`: 字符串，对最终决策 ‘{pre_determined_decision}’ 的总结性理由。\n\n"
                        f"思考链指引（在内心完成，不要输出过程）：\n"
                        f"1. 视觉分析：观察产品图片，评估包装设计、风格和档次感。将此思考总结写入 `structured_report.packaging_analysis`。\n"
                        f"2. 契合度分析：结合产品描述、视觉分析和【相似用户参考数据】，对比你的人设，评估产品在香型、价格、品质等方面是否匹配。将此思考总结写入 `structured_report.fit_analysis`。\n"
                        f"3. 场景构思：构思1-2个你可能会使用该产品的具体场景。将此思考总结写入 `structured_report.scenario_analysis`。\n"
                        f"4. 量化评分：基于以上分析，为`包装`、`价格`、`香型`、`场景`四个维度与你人设的匹配度分别打分，填入`radar_scores`。\n"
                        f"5. 最终决策：综合所有信息，撰写一个强有力的`reason`来支撑已确定的决策 '{pre_determined_decision}'。\n"
                        f"再次强调重要：输出中任何地方不能包含任何Markdown符号（如`*`、`#`、`-`）。"
                    )

                    message_content = [{"type": "text", "text": structured_individual_prompt}]
                    if base64_image:
                        message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", "content": message_content}],
                        max_tokens=2000
                    )
                    
                    raw_content = response.choices[0].message.content
                    if not raw_content or not raw_content.strip().startswith('{'):
                        raise json.JSONDecodeError("Response is not a valid JSON object.", raw_content, 0)
                    
                    analysis_data = json.loads(raw_content)
                    break 

                except (json.JSONDecodeError, IndexError) as e:
                    print(f"警告: 解析画像 {persona_id} 的报告失败 (尝试 {attempt + 1}/{MAX_RETRIES})。错误: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(1)
                    else:
                        print(f"错误: 画像 {persona_id} 的报告在 {MAX_RETRIES} 次尝试后仍然失败。跳过此画像。")
            
            if analysis_data is None:
                continue
            
            report_text_obj = analysis_data.get("structured_report", {})
            cleaned_report_text = {k: str(v).replace('*', '').replace('#', '') for k, v in report_text_obj.items()}
            radar_scores = analysis_data.get("radar_scores", {})
            decision = analysis_data.get("decision", pre_determined_decision)
            reason = analysis_data.get("reason", "").replace('*', '').replace('#', '')
            radar_chart_b64 = generate_chart_base64('radar', radar_scores, "画像-产品匹配度雷达图")

            result_package = {
                "type": "individual_report",
                "data": { 
                    "persona_id": persona_id,
                    "report": cleaned_report_text,
                    "final_decision": {"decision": decision, "reason": reason},
                    "radar_chart": radar_chart_b64,
                    "persona_details": persona,
                    "decision": decision 
                }
            }
            q.put(result_package)
            
            # --- 💡 归档：保存独立报告 HTML ---
            try:
                report_html = _generate_individual_report_html(result_package['data'])
                all_individual_reports_html.append(report_html)
            except Exception as e:
                print(f"[{job_id}] 警告: 归档画像 {persona_id} HTML 失败: {e}")
            # --- 结束归档 ---

            time.sleep(0.5)

        # --- 💡 归档：在任务的最后，保存收集到的 HTML 报告 ---
        
        # 1. 保存综合报告 (包含图表)
        summary_html_content = f'<div class="prose"><p>{summary_report_text.replace(r"\\n\\n", "</p><p>").replace(r"\\n", "<br>")}</p></div>'
        charts_html_content = "\n".join(all_charts_html)
        combined_summary_html = f"{summary_html_content}<hr><h2>数据洞察可视化</h2>{charts_html_content}"
        summary_html_path = report_archive_dir / "summary_report_with_charts.html"
        _save_report_to_html(combined_summary_html, summary_html_path, "综合市场分析与数据可视化报告")

        # 2. 保存所有独立报告
        combined_individual_html = "\n".join(all_individual_reports_html)
        individual_html_path = report_archive_dir / "individual_reports.html"
        _save_report_to_html(combined_individual_html, individual_html_path, "全体画像独立分析报告")
        # --- 结束归档 ---


    except Exception as e:
        print(f"Analysis thread error: {e}")
        import traceback
        traceback.print_exc()
        q.put({"type": "error", "data": str(e)})
    finally:
        q.put({"type": "done"})
        if job_id in analysis_queues: del analysis_queues[job_id]


# --- Charting and AI Analysis (无变动) ---
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
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, color='#6E0F1A', linewidth=3, zorder=3)
        ax.fill(angles, values, color='#6E0F1A', alpha=0.25, zorder=2)
        ax.grid(color='lightgrey', linestyle='--', linewidth=0.7)
        ax.spines['polar'].set_visible(True)
        ax.spines['polar'].set_color('lightgrey')
        ax.set_yticks(np.arange(0, 11, 2))
        ax.set_yticklabels(["", "2", "4", "6", "8", "10"], color="grey", size=15)
        ax.set_ylim(0, 10)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=25, fontproperties=font_prop)
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

# --- City Options Endpoint (无变动) ---
COMPREHENSIVE_CITY_LIST = [
    '北京', '上海', '天津', '重庆', '广东', '广州', '深圳', '东莞', '佛山',
    '江苏', '南京', '苏州', '无锡', '山东', '青岛', '济南', '浙江', '杭州', '宁波',
    '河南', '郑州', '四川', '成都', '湖北', '武汉', '湖南', '长沙', '福建', '厦门', '福州',
    '安徽', '合肥', '河北', '石家庄', '辽宁', '沈阳', '大连', '陕西', '西安',
    '江西', '黑龙江', '广西', '山西', '吉林', '内蒙古', '贵州', '甘肃', '新疆', '海南', '宁夏', '青海', '西藏', 
    '香港', '澳门', '台湾'
]
COMPREHENSIVE_CITY_LIST = sorted(list(set(COMPREHENSIVE_CITY_LIST)))

@app.route('/get_city_options', methods=['GET'])
def get_city_options():
    return jsonify(COMPREHENSIVE_CITY_LIST)

# --- Generate Personas Endpoint (无变动) ---
@app.route('/generate_personas', methods=['POST'])
def generate_personas():
    try:
        payload = request.get_json(silent=True) or {}

        def to_float_map(obj):
            result = {}
            if not isinstance(obj, dict): return result
            for key, value in obj.items():
                if key is None: continue
                try: result[str(key)] = float(value)
                except (TypeError, ValueError): result[str(key)] = 0.0
            return result

        try: count = int(payload.get('count', 0))
        except (TypeError, ValueError): count = 0
        if count <= 0: return jsonify({"error": "画像数量必须大于 0。"}), 400

        age_range = payload.get('age_range') or {}
        try:
            age_min = int(age_range.get('min', 0))
            age_max = int(age_range.get('max', 0))
        except (TypeError, ValueError): age_min, age_max = 0, 0
        if age_min < 18 or age_max > 80 or age_min >= age_max:
            return jsonify({"error": "年龄范围应在 18-80 岁之间，且最小值小于最大值。"}), 400

        gender_ratio = to_float_map(payload.get('gender_ratio') or {})
        drink_ratio = to_float_map(payload.get('drink_frequency_ratio') or {})
        flavor_ratio = to_float_map(payload.get('flavor_ratio') or {})
        mbti_payload = payload.get('mbti_ratio') or {}
        mbti_labels = {'energy': '能量倾向', 'info': '信息接收', 'decision': '决策方式', 'life': '生活态度'}
        mbti_ratio = {
            'energy': to_float_map(mbti_payload.get('energy') or {}),
            'info': to_float_map(mbti_payload.get('info') or {}),
            'decision': to_float_map(mbti_payload.get('decision') or {}),
            'life': to_float_map(mbti_payload.get('life') or {}),
        }
        city_ratio_entries = []
        for entry in payload.get('city_ratio') or []:
            city = entry.get('city')
            if not city: continue
            try: ratio_value = float(entry.get('ratio', 0))
            except (TypeError, ValueError): ratio_value = 0.0
            city_ratio_entries.append({'city': city, 'ratio': ratio_value})

        def ratio_total_close_to_hundred(total): return abs(total - 100) <= 1.5

        validation_errors = []
        for label, ratio_map in [('性别', gender_ratio), ('饮酒频率', drink_ratio), ('偏好香型', flavor_ratio)]:
            if ratio_map and not ratio_total_close_to_hundred(sum(ratio_map.values())):
                 validation_errors.append(f"{label} 比例总和需为 100，当前为 {sum(ratio_map.values()):.2f}。")

        for key, ratio_map in mbti_ratio.items():
            if ratio_map and not ratio_total_close_to_hundred(sum(ratio_map.values())):
                label = mbti_labels.get(key, key)
                validation_errors.append(f"MBTI {label} 比例总和需为 100，当前为 {sum(ratio_map.values()):.2f}。")

        if not city_ratio_entries:
             validation_errors.append("请至少选择一个城市。")
        elif not ratio_total_close_to_hundred(sum(item['ratio'] for item in city_ratio_entries)):
             validation_errors.append(f"城市比例总和需为 100，当前为 {sum(item['ratio'] for item in city_ratio_entries):.2f}。")
        else:
            seen_cities, duplicate_cities = set(), set()
            for item in city_ratio_entries:
                city = item['city']
                if city in seen_cities: duplicate_cities.add(city)
                seen_cities.add(city)
            if duplicate_cities:
                 validation_errors.append(f"城市 {', '.join(sorted(duplicate_cities))} 重复，请调整。")

        if validation_errors:
            return jsonify({"error": "\n".join(validation_errors)}), 400

        city_ratio_map = {item['city']: item['ratio'] for item in city_ratio_entries}
        gender_quota = allocate_counts_from_ratio(gender_ratio, count)
        drink_quota = allocate_counts_from_ratio(drink_ratio, count)
        flavor_quota = allocate_counts_from_ratio(flavor_ratio, count)
        city_quota = allocate_counts_from_ratio(city_ratio_map, count)

        def format_counts(label, counts):
            if not counts: return f"{label}：未指定"
            return f"{label}：" + "、".join(f"{k} {v}人" for k, v in counts.items())

        def format_mbti_ratio():
            lines = []
            for key, ratio_map in mbti_ratio.items():
                if not ratio_map: continue
                ratio_text = "、".join(f"{sub_key} {value:.0f}%" for sub_key, value in ratio_map.items())
                lines.append(f"{mbti_labels.get(key, key)}：{ratio_text}")
            return "\n".join(lines)

        def build_prompt(target_count):
            prompt_parts = [
                f"你是一位消费者洞察专家，请生成 {target_count} 个中国白酒消费者画像。",
                f"年龄需分布在 {age_min}-{age_max} 岁之间，每位画像的年龄为整数。",
                format_counts("性别配额", gender_quota),
                format_counts("城市配额", city_quota),
                format_counts("饮酒频率配额", drink_quota),
                format_counts("偏好香型配额", flavor_quota),
                "MBTI 倾向请尽量贴近以下比例：",
                format_mbti_ratio() or "（未提供额外约束，可自行合理设定）",
                "每名画像需包含以下字段，并使用简体中文：gender, age, city, profession, education, income, expected_price, drink_frequency, drinking_history, preferred_aroma, mbti。",
                "字段约束：",
                "1. education 取值范围：高中及以下、大专、本科、硕士、博士。",
                "2. income 取值范围：10万以下、10-20万、20-50万、50万以上，可适当拓展但需符合常理。",
                "3. expected_price 取值范围：100元以下、100-299元、300-999元、1000元以上。",
                "4. drink_frequency 必须来自配额中的类别。",
                "5. preferred_aroma 必须来自配额中的类别。",
                "6. drinking_history 为整数，范围 0-30，且不得超过年龄 - 18。",
                "7. MBTI 必须是四字母组合（如 ISTJ），请符合比例要求。",
                "8. profession、education、income 等信息需保持人物之间的差异与真实性。",
                f"输出格式：必须返回一个 JSON 对象，且仅包含一个键 `personas`，其值是长度为 {target_count} 的数组。不得包含额外说明、空行或 Markdown 符号。",
                "确保 JSON 严格符合 RFC8259，所有字符串使用双引号，严禁尾随逗号；如无相关信息，请填入最接近的合理值。",
                f"在返回结果之前，务必核对 personas 数组长度是否等于 {target_count}；若不满足必须重写并仅在条件满足时返回结果。"
            ]
            return "\n".join(part for part in prompt_parts if part)

        def build_example_payload():
            first_city = city_ratio_entries[0]['city'] if city_ratio_entries else "北京"
            first_drink_freq = list(drink_quota.keys())[0] if drink_quota else "每月1-2次"
            first_flavor = list(flavor_quota.keys())[0] if flavor_quota else "酱香型"
            first_gender = list(gender_quota.keys())[0] if gender_quota else "男"
            return {"personas": [{"gender": first_gender, "age": age_min, "city": first_city, "profession": "市场总监", "education": "本科", "income": "20-50万", "expected_price": "300-999元", "drink_frequency": first_drink_freq, "drinking_history": 6, "preferred_aroma": first_flavor, "mbti": "ENTJ"}]}

        def extract_personas(text):
            errors = []
            def try_load(payload):
                try:
                    data = json.loads(payload)
                    if isinstance(data, dict) and isinstance(data.get("personas"), list): return data["personas"]
                    if isinstance(data, list): return data
                except Exception as exc: errors.append(exc)
                return None
            parsed = try_load(text)
            if parsed is not None: return parsed
            candidate = text.strip()
            if candidate.startswith("```"):
                candidate = candidate.strip('`').split('\n', 1)[-1]
            start, end = candidate.find('['), candidate.rfind(']')
            if start != -1 and end != -1:
                array_slice = candidate[start:end + 1]
                parsed = try_load(f'{{"personas": {array_slice}}}')
                if parsed is not None: return parsed
                parsed = try_load(array_slice)
                if parsed is not None: return parsed
            parsed = try_load(candidate)
            if parsed is not None: return parsed
            sanitized = re.sub(r',\s*([\]}])', r'\1', candidate)
            if sanitized != candidate:
                parsed = try_load(sanitized)
                if parsed is not None: return parsed
                if sanitized.endswith(','):
                    parsed = try_load(sanitized.rstrip(','))
                    if parsed is not None: return parsed
            python_candidate = candidate.replace('null', 'None').replace('true', 'True').replace('false', 'False')
            try:
                data = ast.literal_eval(python_candidate)
                if isinstance(data, dict) and isinstance(data.get("personas"), list): return data["personas"]
                if isinstance(data, list): return data
            except Exception as exc: errors.append(exc)
            raise (errors[0] if errors else ValueError("无法解析返回内容"))

        def request_persona_batch(target_count):
            prompt = build_prompt(target_count)
            example_payload = build_example_payload()
            raw_text, last_error = "", None
            for _ in range(3):
                try:
                    response = client.chat.completions.create(model="gpt-4o-mini", temperature=0.2, max_tokens=4000, response_format={"type": "json_object"}, messages=[{"role": "system", "content": "你是一位严谨的消费者洞察专家，擅长按照配额生成真实可信的用户画像。输出必须严格遵循 JSON 规范。"}, {"role": "user", "content": f"{prompt}\n请严格参照以下示例返回格式：{json.dumps(example_payload, ensure_ascii=False)}"}])
                except Exception as exc:
                    last_error = exc
                    print(f"API call failed during persona batch generation: {exc}") 
                    continue
                if not response or not response.choices or not response.choices[0].message or not response.choices[0].message.content:
                    last_error = ValueError("API returned an empty response during persona batch generation.")
                    print(last_error) 
                    continue 
                raw_text = response.choices[0].message.content.strip()
                try: 
                    personas = extract_personas(raw_text)
                except Exception as exc:
                    last_error = exc
                    print(f"Failed to extract personas during batch generation: {exc}") 
                    print(f"Raw text was: {raw_text[:500]}") 
                    continue
                if isinstance(personas, list) and len(personas) == target_count: 
                    return True, personas, raw_text, None
                last_error = ValueError(f"Batch generation expected {target_count}, got {len(personas) if isinstance(personas, list) else 'unknown'}.")
                print(last_error) 
            return False, [], raw_text, last_error

        all_personas, remaining = [], count
        batch_size = 10 if count > 10 else count
        last_raw_text, last_error = "", None
        while remaining > 0:
            target_count = min(batch_size, remaining)
            success, batch_personas, raw_text, error = request_persona_batch(target_count)
            if not success:
                last_error, last_raw_text = error, raw_text
                break
            all_personas.extend(batch_personas)
            remaining = count - len(all_personas)
        if remaining > 0: return jsonify({"error": f"生成画像失败: {str(last_error)}" if last_error else "生成画像失败", "raw": (last_raw_text or "")[:400]}), 500
        personas = all_personas
        if not isinstance(personas, list) or len(personas) != count:
            return jsonify({"error": f"最终数量不匹配：期望 {count} 个，实际得到 {len(personas) if isinstance(personas, list) else '未知'} 个。", "raw": (last_raw_text or "")[:400]}), 500

        defaults = {"gender": "未指定", "age": age_min, "city": "未指定", "profession": "未指定职业", "education": "本科", "income": "10-20万", "expected_price": "300-999元", "drink_frequency": "每月1-2次", "drinking_history": 0, "preferred_aroma": "酱香型", "mbti": "ISTJ"}
        cleaned_personas = []
        for persona in personas:
            cleaned = {}
            for key, default_value in defaults.items():
                value = persona.get(key, default_value)
                if key in {'gender', 'city', 'profession', 'education', 'income', 'expected_price', 'drink_frequency', 'preferred_aroma'}:
                    cleaned[key] = str(value).strip() if pd.notna(value) else default_value
                elif key == 'age':
                    try: age_val = int(value)
                    except (TypeError, ValueError): age_val = age_min
                    cleaned[key] = max(age_min, min(age_max, age_val))
                elif key == 'drinking_history':
                    try: history = int(value)
                    except (TypeError, ValueError): history = defaults['drinking_history']
                    history = max(0, min(30, history))
                    cleaned[key] = min(history, cleaned.get('age', age_min) - 18 if cleaned.get('age', age_min) > 18 else 0)
                elif key == 'mbti':
                    mbti_val = str(value).strip().upper() if pd.notna(value) else default_value
                    if re.match(r"^[IE][NS][TF][JP]$", mbti_val):
                        cleaned[key] = mbti_val
                    else:
                        cleaned[key] = defaults['mbti'] 
            cleaned_personas.append(cleaned)

        summary = {"gender": Counter(p['gender'] for p in cleaned_personas), "city": Counter(p['city'] for p in cleaned_personas), "drink_frequency": Counter(p['drink_frequency'] for p in cleaned_personas), "preferred_aroma": Counter(p['preferred_aroma'] for p in cleaned_personas)}

        # 注意：此处的 file=None 是因为我们不在这个阶段保存文件到 *用户* 电脑
        # 真正的服务器归档发生在 /generate_report 调用的 long_running_analysis 中
        return jsonify({"personas": cleaned_personas, "file": None, "summary": {key: dict(value) for key, value in summary.items()}})
    
    except Exception as e:
        print(f"!!! Error in /generate_personas endpoint !!!")
        import traceback
        traceback.print_exc() 
        return jsonify({"error": f"生成画像时发生内部错误: {e}"}), 500


# --- Flask Routes (Report Generation) ---
@app.route('/generate_report', methods=['POST'])
def generate_report_start():
    job_id = hashlib.md5(str(time.time()).encode()).hexdigest()
    analysis_queues[job_id] = Queue()
    data = request.json
    thread = threading.Thread(
        target=long_running_analysis,
        args=(job_id, data.get('personas', []), data.get('productData', {}), data.get('persona_file'))
    )
    thread.start()
    return jsonify({"job_id": job_id})

@app.route('/stream/<job_id>')
def stream(job_id):
    def event_stream():
        q = analysis_queues.get(job_id)
        if not q: yield f"data: {json.dumps({'type': 'error', 'data': '无效的任务ID'})}\n\n"; return
        while True:
            try:
                message = q.get(timeout=240)
                yield f"data: {json.dumps(message)}\n\n"
                if message.get('type') == 'done': break
            except Exception:
                yield f"data: {json.dumps({'type': 'error', 'data': '数据流超时或任务已结束'})}\n\n"; break
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/')
def index():
    return render_template('index.html')

# --- 💡 新增：后台管理路由 ---

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin_login.html', error='账号或密码错误')
    
    # 如果已经登录，直接跳转到后台
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard'))
        
    return render_template('admin_login.html', error=None)

@app.route('/admin_logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    archives = []
    if ARCHIVE_DIR.exists():
        # 按名称（时间戳）倒序排序
        sorted_dirs = sorted(os.listdir(ARCHIVE_DIR), reverse=True)
        
        for report_id in sorted_dirs:
            report_dir = ARCHIVE_DIR / report_id
            if not report_dir.is_dir():
                continue
            
            # --- 💡 新增：解析时间戳为可读格式 ---
            display_time = report_id
            try:
                # 尝试按 YYYYMMDD_HHMMSS 格式解析
                dt = datetime.strptime(report_id, "%Y%m%d_%H%M%S")
                display_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                # 如果解析失败（比如文件夹名字不是这个格式），就用原始 ID
                pass
            # --- 结束新增 ---

            try:
                files = os.listdir(report_dir)
                
                # 优先查找带图表的报告
                summary_file = "summary_report_with_charts.html"
                if summary_file not in files:
                    summary_file = "summary_report.html" # 回退
                
                archives.append({
                    "id": report_id,
                    "display_time": display_time, # 💡 传递可读时间
                    "has_personas": "personas.json" in files,
                    "summary_file": summary_file if summary_file in files else None,
                    "has_individual": "individual_reports.html" in files
                })
            except Exception as e:
                print(f"扫描归档 {report_id} 出错: {e}")

    return render_template('admin_dashboard.html', archives=archives)

@app.route('/download_archive/<path:report_id>/<path:filename>')
def download_archive(report_id, filename):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    try:
        # 安全地拼接路径，防止路径遍历攻击
        safe_path = safe_join(str(ARCHIVE_DIR.resolve()), report_id)
        if safe_path is None or not os.path.isdir(safe_path):
             return "Invalid report ID", 404
             
        # 再次检查文件名
        safe_filename = safe_join(safe_path, filename)
        if safe_filename is None or not os.path.exists(safe_filename) or not safe_filename.startswith(safe_path):
            return "Invalid filename", 404

        return send_from_directory(
            directory=safe_path,
            path=filename,
            as_attachment=True
        )
    except Exception as e:
        print(f"下载文件失败: {e}")
        return "下载失败", 500

# --- 结束新增 ---


# --- Final startup logic ---
load_and_preprocess_data()
vectorize_database()

if __name__ == '__main__':
    print("\n--- MingYin Insight Engine (Admin Enabled) ---")
    print("真实数据已加载并向量化。应用已准备就绪。")
    print(f"归档目录位于: {ARCHIVE_DIR.resolve()}")
    print("请使用 Waitress 启动生产服务器。")
    print("推荐命令: waitress-serve --host=0.0.0.0 --port=5001 --threads=10 app:app")
    # 备注：本地调试时，可以使用 app.run(debug=True, port=5001)
    # app.run(debug=True, port=5001)