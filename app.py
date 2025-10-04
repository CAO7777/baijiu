# app.py (完整优化最终版)
import os
import random
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# --- 基本配置 ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- AI 和数据加载 ---
client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("API_URL"))
try:
    df_users = pd.read_excel("panel_data_sample.xlsx")
except FileNotFoundError:
    print("FATAL: panel_data_sample.xlsx not found. The application cannot start.")
    df_users = None

def get_matching_user_id(persona):
    if df_users is None or df_users.empty:
        return 'U001'
    matches = df_users.copy()
    age = int(persona.get('age', 0))
    matches = matches[
        (matches['性别'] == persona.get('gender')) &
        (matches['年龄'].between(age - 5, age + 5)) &
        (matches['MBTI/性格'] == persona.get('mbti'))
    ]
    if not matches.empty:
        return random.choice(matches['用户ID'].unique())
    else:
        return random.choice(df_users['用户ID'].unique())

def run_ai_analysis(personas, product_data):
    try:
        product_description = product_data.get('description')
        base64_image = product_data.get('image').split(',')[1]
        individual_reports = []

        for i, persona in enumerate(personas):
            matched_user_id = get_matching_user_id(persona)
            user_history_df = df_users[df_users['用户ID'] == matched_user_id]
            user_profile_from_db = user_history_df.iloc[0]
            
            # 准备用户画像和历史数据文本
            profession = persona.get('profession') or user_profile_from_db['职业']
            profile_text = (f"你是一位{user_profile_from_db['年龄']}岁的{user_profile_from_db['性别']}性，来自{user_profile_from_db['地区/城市']}，"
                          f"职业是“{profession}”，性格类型为{user_profile_from_db['MBTI/性格']}。")
            
            # --- 核心修正：将错误语法的代码改回简单、正确的 if 判断 ---
            optional_details = []
            if persona.get('education'):
                optional_details.append(f"你的教育程度是{persona['education']}。")
            if persona.get('income'):
                optional_details.append(f"你的年收入在{persona['income']}范围。")
            if persona.get('drink_frequency'):
                optional_details.append(f"你的饮酒频率是'{persona['drink_frequency']}'。")
            if persona.get('drinking_history'):
                optional_details.append(f"你已经有{persona['drinking_history']}年酒龄。")
            if persona.get('expected_price'):
                optional_details.append(f"你购买白酒的心理价位是{persona['expected_price']}。")
            if persona.get('preferred_aroma'):
                optional_details.append(f"你个人偏好的白酒香型是'{persona['preferred_aroma']}'。")
            # --- 修正结束 ---

            if optional_details:
                profile_text += " " + " ".join(optional_details)
            history_text = user_history_df[['日期', '白酒香型', '度数带', '价格区间', '主要用途']].to_string(index=False)
            real_user_prompt = f"{profile_text}\n\n以下是你过去的部分消费记录：\n{history_text}"

            # --- 结构化的个体分析 (COT) Prompt ---
            structured_individual_prompt = (
                f"背景：你将代入以下人物画像进行思考。\n"
                f"人物画像与消费记录：\n---\n{real_user_prompt}\n---\n\n"
                f"任务：请严格按照以下【思考链】的四个步骤，分析一款新产品，并最终做出购买决策。\n\n"
                f"【第一步：视觉分析】\n"
                f"首先，仅根据产品图片，分析其视觉呈现。在内心回答（无需输出此步骤的内心思考）：\n"
                f"包装设计、瓶身造型、品牌质感如何？\n"
                f"它给你的第一印象是高端、中端还是大众化？\n"
                f"它的风格是偏向传统经典，还是现代创新？\n"
                f"你认为它的目标人群是谁？适用于什么场景（例如：商务宴请、朋友聚会、日常独酌、节日送礼）？\n\n"
                f"【第二步：信息匹配】\n"
                f"接下来，结合产品文字描述：'{product_description}'，以及你在第一步的视觉分析，与你的人物画像进行匹配。在内心回答：\n"
                f"产品的香型、度数、价格区间（根据你的视觉判断）是否符合你的偏好和消费记录？\n"
                f"产品的外观风格是否符合你的个人审美和身份定位？\n"
                f"产品的定位（高端/中端/大众）是否符合你的心理价位和主要用途？\n\n"
                f"【第三步：场景构思】\n"
                f"基于以上分析，构思一到两个你可能会购买或使用这款产品的具体生活场景。\n\n"
                f"【第四步：最终决策】\n"
                f"综合以上所有思考，输出你的最终结论。你的输出必须严格遵守以下格式，以'【决策】'和'【理由】'作为开头。\n"
                f"在【理由】部分，请清晰地阐述你是如何根据你的思考链（视觉感受、信息匹配、场景构思）推导出最终决策的。\n\n"
                f"--- DO NOT a single word about the step-by-step thinking process in the final output. JUST output the final decision in the requested format. ---\n"
                f"贯穿全文的重要要求：最终输出中不能包含任何Markdown符号（如`*`、`#`、`-`），也不能署名或添加日期。"
            )

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user", "content": [
                        {"type": "text", "text": structured_individual_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=1000
            )
            report_text = response.choices[0].message.content
            individual_reports.append({"persona_id": i + 1, "report": report_text, "persona_details": persona})

        # --- 结构化的市场总结 Prompt  ---
        all_reports_text = "\n\n".join([f"--- 第{item['persona_id']}位模拟用户的独立报告 ---\n{item['report']}\n" for item in individual_reports])
        
        structured_summary_prompt = (
            f"你是一位顶级的市场研究总监，你的任务是基于以下 {len(individual_reports)} 份由不同模拟用户生成的独立反馈报告，撰写一份深刻、专业、结构化的综合市场分析报告。\n\n"
            f"【原始数据：独立反馈报告】\n{all_reports_text}\n\n"
            f"请严格按照以下大纲撰写你的综合分析报告，确保内容深刻、逻辑清晰、语言专业。报告中不能包含任何Markdown符号。\n\n"
            f"1. 核心洞察与购买意向**\n"
            f"首先，精确计算并明确展示总体购买率（购买人数/总人数）。\n"
            f"提炼出本次模拟中最关键、最核心的市场洞察是什么。\n\n"
            f"2. 关键购买驱动力分析**\n"
            f"深入总结吸引用户做出“购买”决策的核心因素。请从产品属性（如香型、口感描述）、包装设计（如外观、档次感）、品牌定位和价格感知等多个维度进行剖析。\n\n"
            f"3. 主要购买壁垒剖析**\n"
            f"同样，深入剖析导致用户做出“不购买”决策的关键障碍。分析这些障碍是源于产品自身、价格、包装，还是与目标用户的核心需求存在错位。\n\n"
            f"4. 综合结论与市场策略建议**\n"
            f"对产品的市场潜力给出一个简洁有力的综合结论。\n"
            f"基于以上所有分析，为该产品的市场策略提供1-2条具体的、可执行的建议（例如：调整营销话术、优化包装细节、主攻特定消费场景等）。"
            f"贯穿全文的重要要求：最终输出中不能包含任何Markdown符号（如`*`、`#`、`-`），也不能署名或添加日期。"
        )

        summary_response = client.chat.completions.create(
            model="gpt-4-turbo", 
            messages=[
                {"role": "user", "content": structured_summary_prompt}
            ],
            max_tokens=1500
        )
        summary_report = summary_response.choices[0].message.content
        return {"individual_reports": individual_reports, "summary_report": summary_report}

    except Exception as e:
        print(f"An error occurred during AI analysis: {e}")
        return {"error": str(e)}

# --- API Endpoint & Index Route (保持不变) ---
@app.route('/generate_report', methods=['POST'])
def generate_report_sync():
    data = request.json
    personas = data.get('personas', [])
    product_data = data.get('productData', {})
    final_result = run_ai_analysis(personas, product_data)
    if "error" in final_result:
        return jsonify(final_result), 500
    return jsonify(final_result)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
