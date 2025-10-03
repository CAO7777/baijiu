import os
import random
from flask import Flask, request, jsonify
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
    # 读取之前脚本生成的面板数据样本
    df_users = pd.read_excel("panel_data_sample.xlsx")
except FileNotFoundError:
    print("FATAL: panel_data_sample.xlsx not found. The application cannot start.")
    df_users = None

def get_matching_user_id(persona):
    """
    根据前端传入的画像，从数据库中找到最匹配的【用户ID】。
    """
    if df_users is None or df_users.empty:
        return 'U001' # 如果文件不存在，返回一个默认ID

    matches = df_users.copy()
    age = int(persona.get('age', 0))
    
    # 核心匹配逻辑
    matches = matches[
        (matches['性别'] == persona.get('gender')) &
        (matches['年龄'].between(age - 5, age + 5)) &
        (matches['MBTI/性格'] == persona.get('mbti'))
    ]
    
    if not matches.empty:
        # 从所有匹配的记录中，随机选一个用户ID
        matched_ids = matches['用户ID'].unique()
        return random.choice(matched_ids)
    else:
        # 如果没找到，从整个数据库中随机选一个用户ID作为后备
        all_ids = df_users['用户ID'].unique()
        return random.choice(all_ids)

def run_ai_analysis(personas, product_data):
    """
    包含了所有的AI分析逻辑。
    会动态构建包含用户可选信息的AI指令。
    """
    try:
        product_description = product_data.get('description')
        base64_image = product_data.get('image').split(',')[1]
        
        individual_reports = []
        
        for i, persona in enumerate(personas):
            # 1. 找到一个匹配的用户ID
            matched_user_id = get_matching_user_id(persona)
            
            # 2. 提取该用户的所有历史记录
            user_history_df = df_users[df_users['用户ID'] == matched_user_id]
            
            # 3. 从该用户的第一条记录中提取其不变的画像信息
            user_profile_from_db = user_history_df.iloc[0]
            
            # --- 核心修改部分：智能构建AI指令 ---

            # 4. 确定职业：优先使用用户在前端输入的职业，如果为空，则使用数据库中匹配到的职业
            profession = persona.get('profession') or user_profile_from_db['职业']

            # 5. 构建基础画像文本，并使用上面确定的职业
            profile_text = (f"你是一位{user_profile_from_db['年龄']}岁的{user_profile_from_db['性别']}性，来自{user_profile_from_db['地区/城市']}，"
                            f"职业是“{profession}”，性格类型为{user_profile_from_db['MBTI/性格']}。")
            
            # 6. 检查并添加用户在前端填写的【其他可选】信息
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

            # 7. 将可选信息合并到主指令中
            if optional_details:
                profile_text += " " + " ".join(optional_details)
            
            # 8. 添加历史消费记录
            history_text = user_history_df[['日期', '白酒香型', '度数带', '价格区间', '主要用途']].to_string(index=False)
            real_user_prompt = f"{profile_text}\n\n以下是你过去的部分消费记录：\n{history_text}"

            # --- AI调用部分 ---
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user", "content": [
                        {"type": "text", "text": (f"请基于以下这个非常详细的人物画像和他的历史消费记录进行思考：'{real_user_prompt}'。"
                                                  f"\n\n现在，请分析一款【新】的白酒产品，产品描述为：'{product_description}'。"
                                                  f"请你根据你的人物画像和消费习惯，判断是否会购买这款【新】产品？"
                                                  "以'【决策】'和'【理由】'作为开头，明确给出你的购买决策和详细理由。"
                                                  "重要：你的整个回答都必须严格使用简体中文。，不要**在任何地方出现，不能包含任何Markdown符号（如`*`、`#`、`-`），也不能署名或添加日期。")},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ], max_tokens=800)
            report_text = response.choices[0].message.content
            individual_reports.append({"persona_id": i + 1, "report": report_text, "persona_details": persona})

        # --- 汇总报告部分 ---
        all_reports_text = "\n\n".join([f"第{item['persona_id']}位模拟用户的报告：\n{item['report']}" for item in individual_reports])
        summary_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "你是一位资深的市场研究分析师，擅长洞察消费者行为并撰写专业的中文分析报告。你的所有输出都必须是简体中文。不能包含任何Markdown符号（如`*`、`#`、`-`），也不能署名或添加日期。"
                },
                {"role": "user", "content": (f"这里有{len(individual_reports)}位模拟用户对一款新白酒产品的反馈报告：\n\n---\n\n"
                                f"{all_reports_text}\n\n---\n\n"
                                "请你基于以上所有独立报告，撰写一份综合性的市场分析报告。报告需要包含以下几点：\n"
                                "1. 总体购买意向：明确计算并展示总体购买率（购买人数/总人数）。，不要**在任何地方出现\n"
                                "2. 主要购买驱动力：总结吸引用户购买的关键因素是什么（例如：口感、品牌、包装、价格等）。，不要**在任何地方出现\n"
                                "3. 主要购买障碍：总结导致用户不购买的关键因素是什么。，不要**在任何地方出现\n"
                                "4. 结论与建议：给出一个简洁的总结，并为产品的市场策略提供1-2条具体建议。，不要**在任何地出现\n"
                                "请使用专业的、结构化的中文报告格式进行撰写。，不要**在任何地方出现")
                }
            ], max_tokens=1500)
        summary_report = summary_response.choices[0].message.content

        return {"individual_reports": individual_reports, "summary_report": summary_report}

    except Exception as e:
        print(f"An error occurred during AI analysis: {e}")
        return {"error": str(e)}

# --- API Endpoint ---
@app.route('/generate_report', methods=['POST'])
def generate_report_sync():
    data = request.json
    personas = data.get('personas', [])
    product_data = data.get('productData', {})
    final_result = run_ai_analysis(personas, product_data)
    if "error" in final_result:
        return jsonify(final_result), 500
    return jsonify(final_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)