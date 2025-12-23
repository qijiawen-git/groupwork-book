import pandas as pd
import numpy as np
import re
import json
from datetime import datetime


def clean_douban_data(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    print(f"原始数据形状: {df.shape}")
    print(f"原始数据列名: {df.columns.tolist()}")

    def clean_rating_count(text):
        if pd.isna(text):
            return 0
        numbers = re.findall(r'\d+', str(text))
        if numbers:
            return int(''.join(numbers))
        return 0

    df['评价人数'] = df['评价人数'].apply(clean_rating_count)

    def clean_price(text):
        if pd.isna(text):
            return None
        text = str(text).strip()

        if '元' in text:
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                return float(numbers[0])

        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            return float(numbers[-1])

        return None

    df['价格'] = df['价格'].apply(clean_price)

    def clean_publish_year(text):
        if pd.isna(text):
            return None
        text = str(text)
        years = re.findall(r'\d{4}', text)
        if years:
            return int(years[0])
        return None

    df['出版年份'] = df['出版时间'].apply(clean_publish_year)

    df['评分'] = pd.to_numeric(df['评分'], errors='coerce')

    def extract_main_author(author_text):
        if pd.isna(author_text):
            return "", ""

        author_text = str(author_text)

        if '著' in author_text or '编' in author_text:
            parts = re.split(r'[/]', author_text)
            author = parts[0].strip() if len(parts) > 0 else ""
            translator = parts[1].strip() if len(parts) > 1 else ""
            return author, translator

        parts = re.split(r'[/]', author_text)
        if len(parts) > 1:
            author = parts[0].strip()
            translator = ''.join(parts[1:]).strip()
        else:
            author = author_text.strip()
            translator = ""

        return author, translator

    author_translator = df['作者'].apply(extract_main_author)
    df['主要作者'] = author_translator.apply(lambda x: x[0])
    df['译者编者'] = author_translator.apply(lambda x: x[1])

    def get_rating_range(count):
        if count >= 1000000:
            return "100万+"
        elif count >= 500000:
            return "50-100万"
        elif count >= 100000:
            return "10-50万"
        elif count >= 50000:
            return "5-10万"
        elif count >= 10000:
            return "1-5万"
        else:
            return "1万以下"

    df['评价区间'] = df['评价人数'].apply(get_rating_range)

    def get_price_range(price):
        if pd.isna(price):
            return "未知"
        if price <= 20:
            return "0-20元"
        elif price <= 40:
            return "20-40元"
        elif price <= 60:
            return "40-60元"
        elif price <= 100:
            return "60-100元"
        else:
            return "100元以上"

    df['价格区间'] = df['价格'].apply(get_price_range)

    def get_decade(year):
        if pd.isna(year):
            return "未知"
        decade = (year // 10) * 10
        return f"{decade}年代"

    df['出版年代'] = df['出版年份'].apply(get_decade)

    def get_star_category(rating):
        if pd.isna(rating):
            return "暂无评分"
        if rating >= 9.0:
            return "9分以上神作"
        elif rating >= 8.5:
            return "8.5-9分优秀"
        elif rating >= 8.0:
            return "8-8.5分良好"
        else:
            return "8分以下"

    df['评分等级'] = df['评分'].apply(get_star_category)

    publisher_counts = df['出版社'].value_counts().to_dict()
    df['出版社热度'] = df['出版社'].apply(lambda x: publisher_counts.get(x, 0))

    author_counts = df['主要作者'].value_counts().to_dict()
    df['作者热度'] = df['主要作者'].apply(lambda x: author_counts.get(x, 0))

    df['热度指数'] = df.apply(
        lambda row: row['评分'] * np.log10(max(row['评价人数'], 1)),
        axis=1
    )

    def extract_keywords(intro):
        if pd.isna(intro):
            return []
        intro = str(intro)
        words = re.findall(r'[\u4e00-\u9fa5]{2,4}', intro)
        return list(set(words))[:3]

    df['简介关键词'] = df['简介'].apply(extract_keywords)

    cleaned_file = 'douban_top250_cleaned.csv'
    df.to_csv(cleaned_file, index=False, encoding='utf-8-sig')

    print(f"数据清洗完成！清洗后数据形状: {df.shape}")
    print(f"数据已保存至: {cleaned_file}")

    print("\n=== 数据概览 ===")
    print(f"评分范围: {df['评分'].min():.1f} - {df['评分'].max():.1f}")
    print(f"评价人数范围: {df['评价人数'].min()} - {df['评价人数'].max()}")
    print(f"价格范围: {df['价格'].min()} - {df['价格'].max()}元")
    print(f"出版年份范围: {df['出版年份'].min()} - {df['出版年份'].max()}")

    return df


df = clean_douban_data('douban_top250_books.csv')