import pandas as pd
import random
from datetime import datetime, timedelta

# --- 1. 定义不变的用户画像 (User Profiles) ---
user_profiles = [
    {
        '用户ID': 'U001', '性别': '男', '年龄': 35, '职业': '科技/互联网从业者',
        '教育程度': '硕士', '收入区间': '20-30万', '地区/城市': '北京', 'MBTI/性格': 'INTJ',
        '饮酒频率': '每周', '饮酒历史（酒龄）': 10
    },
    {
        '用户ID': 'U002', '性别': '女', '年龄': 42, '职业': '金融/法律/咨询顾问',
        '教育程度': '本科', '收入区间': '30-50万', '地区/城市': '上海', 'MBTI/性格': 'ESTJ',
        '饮酒频率': '偶尔', '饮酒历史（酒龄）': 15
    },
    {
        '用户ID': 'U003', '性别': '男', '年龄': 28, '职业': '自由职业/创业者',
        '教育程度': '本科', '收入区间': '10-20万', '地区/城市': '成都', 'MBTI/性格': 'ENFP',
        '饮酒频率': '月1–2次', '饮酒历史（酒龄）': 5
    },
    {
        '用户ID': 'U004', '性别': '女', '年龄': 55, '职业': '公务员/事业单位',
        '教育程度': '本科', '收入区间': '20-30万', '地区/城市': '广州', 'MBTI/性格': 'ISFJ',
        '饮酒频率': '几乎每天', '饮酒历史（酒龄）': 30
    },
    {
        '用户ID': 'U005', '性别': '男', '年龄': 48, '职业': '企业主',
        '教育程度': '博士', '收入区间': '>50万', '地区/城市': '深圳', 'MBTI/性格': 'ENTJ',
        '饮酒频率': '每周', '饮酒历史（酒龄）': 25
    }
]

# --- 2. 定义每次购买时可能变化的属性选项 ---
aroma_types = ['酱香', '浓香', '清香', '兼香']
alcohol_bands = ['53+', '39-52', '≤38']
price_ranges = ['100-299元', '300-499元', '500-999元', '1000元以上']
purposes = ['自饮', '宴席', '送礼', '商务']
channels = ['商超', '餐饮', '电商', '直播']
holidays = ['无', '春节', '中秋', '618', '双11']

# --- 3. 生成10条面板数据 ---
data_rows = []
for _ in range(10):
    profile = random.choice(user_profiles).copy()
    
    purchase_date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
    profile['日期'] = purchase_date
    profile['白酒香型'] = random.choice(aroma_types)
    profile['度数带'] = random.choice(alcohol_bands)
    profile['价格区间'] = random.choice(price_ranges)
    profile['主要用途'] = random.choice(purposes)
    profile['购买渠道'] = random.choice(channels)
    profile['购买时间与节日'] = random.choice(holidays) if '6-18' in purchase_date or '11-11' in purchase_date else '无'
    
    data_rows.append(profile)

# --- 4. 创建DataFrame并保存到Excel ---
df = pd.DataFrame(data_rows)

column_order = [
    '用户ID', '日期', '性别', '年龄', '职业', '教育程度', '收入区间', '地区/城市',
    'MBTI/性格', '饮酒频率', '饮酒历史（酒龄）', '白酒香型', '度数带', '价格区间',
    '主要用途', '购买渠道', '购买时间与节日'
]
df = df[column_order]

# 保存到Excel文件
output_filename = 'panel_data_sample.xlsx'
df.to_excel(output_filename, index=False)

print(f"数据生成完毕！已保存为 {output_filename}")