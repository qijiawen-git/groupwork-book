import os
import json
import requests
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime


class DoubanAIAnalyzer:
    def __init__(self, analyzer, api_key=None):
        self.analyzer = analyzer
        # 使用你提供的API密钥
        self.api_key = 'sk-q60TnJMBgZRpNLTVBf8092Fe084241D098B703D7F1C5Ad7c'
        self.api_url = "https://api.edgefn.net/v1/chat/completions"
        self.cache = {}

        # 添加调试信息
        print(f"[AI服务初始化] API密钥状态: {'已设置' if self.api_key else '未设置'}")
        print(f"[AI服务初始化] 使用模型: Qwen3-Next-80B-A3B-Instruct")
        print(f"[AI服务初始化] 请求地址: {self.api_url}")

    def _prepare_data_context(self, question: str) -> Dict:
        """根据问题准备数据上下文"""
        df = self.analyzer.df
        context = {
            "question": question,
            "data_summary": self.analyzer.get_data_summary(),
            "sample_data": df.head(5).to_dict('records'),
            "data_shape": df.shape,
            "columns": df.columns.tolist()
        }

        # 根据问题类型添加特定数据
        question_lower = question.lower()

        if '评分' in question_lower and '最高' in question_lower:
            context['top_books'] = self.analyzer.get_top_books('rating', 10)
        elif '评价人数' in question_lower or '热门' in question_lower:
            context['top_books'] = self.analyzer.get_top_books('popularity', 10)
        elif '价格' in question_lower or '便宜' in question_lower:
            context['top_books'] = self.analyzer.get_top_books('value', 10)
        elif '出版社' in question_lower:
            context['publisher_analysis'] = self.analyzer.get_publisher_analysis()
        elif '作者' in question_lower:
            context['author_analysis'] = self.analyzer.get_author_analysis()
        elif '趋势' in question_lower or '年份' in question_lower:
            context['trend_analysis'] = self.analyzer.get_trend_analysis()

        return context

    def _build_prompt(self, question: str, context: Dict) -> str:
        """构建AI提示词"""
        data_summary = context['data_summary']
        stats = data_summary['statistics']

        prompt = f"""你是一个数据分析专家，正在分析豆瓣图书Top250的真实数据。

真实数据概况：
- 数据集包含 {stats['total_books']} 本豆瓣Top250图书
- 平均评分：{stats['avg_rating']} 分
- 总评价次数：{stats['total_ratings']:,} 次
- 出版年份范围：{stats['year_range']}
- 出版社数量：{stats['unique_publishers']} 家
- 作者数量：{stats['unique_authors']} 位

用户问题：{question}

请基于上述数据提供准确、有用的分析和建议，用中文回答。"""

        return prompt

    def analyze_with_ai(self, question: str) -> Dict:
        """使用AI分析问题"""
        cache_key = question.lower()

        # 检查缓存
        if cache_key in self.cache:
            print(f"[AI分析] 从缓存加载结果")
            return self.cache[cache_key]

        # 准备数据上下文
        context = self._prepare_data_context(question)

        try:
            # 尝试调用外部AI API
            print(f"[AI分析] 尝试调用外部API分析: {question}")
            prompt = self._build_prompt(question, context)
            ai_response = self._call_external_api(prompt)

            result = {
                "question": question,
                "ai_analysis": ai_response,
                "data_context": {
                    "summary": context['data_summary'],
                    "sample_size": len(self.analyzer.df),
                    "data_columns": context['columns']
                },
                "timestamp": datetime.now().isoformat(),
                "source": "external_api"
            }

            # 缓存结果
            self.cache[cache_key] = result
            print("[AI分析] 外部API调用成功")
            return result

        except Exception as api_error:
            print(f"[AI分析] 外部API调用失败: {api_error}")
            print("[AI分析] 使用本地数据生成动态分析")

            # API失败时，基于真实数据生成动态分析
            return self._generate_dynamic_analysis(question, context)

    def _call_external_api(self, prompt: str) -> str:
        """调用外部API（Qwen3模型）"""
        if not self.api_key:
            raise ValueError("未设置API密钥")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "Qwen3-Next-80B-A3B-Instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个严谨的数据分析师，基于用户提供的数据进行分析，不编造信息，用中文回答。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.9
        }

        try:
            # 增加超时时间到60秒，避免网络波动导致超时
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)

            # 检查HTTP状态码
            if response.status_code == 401:
                raise ValueError("API密钥无效或过期，请检查您的API密钥")
            elif response.status_code == 404:
                raise ValueError("API端点或模型不存在，请检查模型名称和API地址")
            elif response.status_code == 429:
                raise ValueError("请求频率超限，请稍后重试")
            elif response.status_code >= 500:
                raise ValueError(f"API服务器错误 ({response.status_code})，请稍后重试")

            response.raise_for_status()  # 其他HTTP错误

            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            elif "error" in result:
                raise ValueError(f"API返回错误: {result['error']}")
            else:
                raise ValueError("API返回格式不正确")

        except requests.exceptions.Timeout:
            raise Exception("API请求超时（60秒），请检查网络连接或稍后重试")
        except requests.exceptions.ConnectionError:
            raise Exception("无法连接到API服务器，请检查网络连接")
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求失败: {str(e)}"
            if hasattr(e, 'response') and e.response:
                try:
                    error_detail = e.response.json()
                    error_msg += f"\n错误详情: {error_detail}"
                except:
                    error_msg += f"\n响应内容: {e.response.text[:200]}"
            raise Exception(error_msg)

    def _generate_dynamic_analysis(self, question: str, context: Dict) -> Dict:
        """基于真实数据生成动态分析（备用方案）"""
        df = self.analyzer.df
        stats = context['data_summary']['statistics']

        # 根据问题类型生成不同的分析
        question_lower = question.lower()

        if '评分' in question_lower and '最高' in question_lower:
            analysis = self._generate_rating_analysis(df, stats)
        elif '评价人数' in question_lower or '热门' in question_lower:
            analysis = self._generate_popularity_analysis(df, stats)
        elif '价格' in question_lower or '性价比' in question_lower:
            analysis = self._generate_value_analysis(df, stats)
        elif '出版社' in question_lower:
            analysis = self._generate_publisher_analysis(df, stats)
        elif '作者' in question_lower:
            analysis = self._generate_author_analysis(df, stats)
        elif '趋势' in question_lower or '年份' in question_lower:
            analysis = self._generate_trend_analysis(df, stats)
        else:
            # 通用分析，基于当前数据动态生成
            analysis = self._generate_general_analysis(df, stats, question)

        return {
            "question": question,
            "ai_analysis": analysis,
            "data_context": {
                "summary": context['data_summary'],
                "sample_size": len(df),
                "data_columns": context['columns']
            },
            "timestamp": datetime.now().isoformat(),
            "source": "local_dynamic_analysis",
            "note": "基于本地真实数据的动态分析（外部API不可用时的备用方案）"
        }

    def _generate_rating_analysis(self, df: pd.DataFrame, stats: Dict) -> str:
        """动态生成评分分析"""
        top_rated = df.nlargest(5, '评分')

        analysis = f"基于豆瓣Top250数据，当前数据集中评分最高的书籍如下：\n\n"

        for i, (_, row) in enumerate(top_rated.iterrows(), 1):
            rating = row['评分']
            rating_level = "神作" if rating >= 9.0 else "优秀" if rating >= 8.5 else "良好"
            analysis += f"{i}. 《{row['书名']}》\n"
            analysis += f"   评分：{rating}分（{rating_level}）\n"
            analysis += f"   作者：{row.get('主要作者', '未知')}\n"
            analysis += f"   评价人数：{row.get('评价人数', 0):,}人\n\n"

        # 添加评分统计信息
        analysis += f"数据洞察：\n"
        analysis += f"- 数据集平均评分：{stats['avg_rating']}分\n"
        analysis += f"- 最高评分：{stats['max_rating']}分，最低评分：{stats['min_rating']}分\n"
        analysis += f"- 9分以上书籍：{len(df[df['评分'] >= 9.0])}本\n"
        analysis += f"- 8.5分以上书籍：{len(df[df['评分'] >= 8.5])}本\n\n"

        analysis += f"建议：高评分书籍通常具有较高的文学价值和读者认可度，是阅读的优选。"

        return analysis

    def _generate_popularity_analysis(self, df: pd.DataFrame, stats: Dict) -> str:
        """动态生成热度分析"""
        top_popular = df.nlargest(5, '评价人数')

        analysis = f"基于豆瓣Top250数据，当前数据集中最受欢迎的书籍如下：\n\n"

        for i, (_, row) in enumerate(top_popular.iterrows(), 1):
            rating_count = row['评价人数']
            analysis += f"{i}. 《{row['书名']}》\n"
            analysis += f"   评价人数：{rating_count:,}人\n"
            analysis += f"   评分：{row.get('评分', 0)}分\n"
            analysis += f"   作者：{row.get('主要作者', '未知')}\n\n"

        # 添加热度统计信息
        analysis += f"数据洞察：\n"
        analysis += f"- 数据集总评价次数：{stats['total_ratings']:,}次\n"
        analysis += f"- 平均每本书评价人数：{stats['total_ratings'] // stats['total_books']:,}人\n"
        analysis += f"- 最热门书籍评价人数是最冷门书籍的{top_popular.iloc[0]['评价人数'] / df['评价人数'].min():.0f}倍\n\n"

        analysis += f"建议：热门书籍通常具有广泛的影响力和讨论度，适合作为入门阅读选择。"

        return analysis

    def _generate_general_analysis(self, df: pd.DataFrame, stats: Dict, question: str) -> str:
        """动态生成通用分析"""
        # 获取最新的数据统计
        total_books = len(df)
        avg_rating = df['评分'].mean()
        total_ratings = df['评价人数'].sum()

        # 动态计算价格信息
        price_info = ""
        if '价格' in df.columns:
            valid_prices = df['价格'].dropna()
            if len(valid_prices) > 0:
                avg_price = valid_prices.mean()
                price_info = f"- 平均价格：{avg_price:.2f}元（基于{len(valid_prices)}本有价格数据的书籍）\n"

        # 动态计算出版社和作者信息
        publisher_info = ""
        if '出版社' in df.columns:
            top_publishers = df['出版社'].value_counts().head(3)
            if len(top_publishers) > 0:
                publisher_info = f"- 热门出版社：{', '.join([f'{pub}({count}本)' for pub, count in top_publishers.items()])}\n"

        author_info = ""
        if '主要作者' in df.columns:
            top_authors = df['主要作者'].value_counts().head(3)
            if len(top_authors) > 0:
                author_info = f"- 热门作者：{', '.join([f'{author}({count}部)' for author, count in top_authors.items()])}\n"

        # 构建动态分析
        analysis = f"针对你的问题「{question}」，基于豆瓣Top250的{total_books}本真实图书数据：\n\n"
        analysis += f"数据概览：\n"
        analysis += f"- 平均评分：{avg_rating:.2f}分\n"
        analysis += f"- 总评价次数：{total_ratings:,}次\n"
        analysis += f"- 出版年份范围：{stats['year_range']}\n"
        analysis += price_info
        analysis += publisher_info
        analysis += author_info
        analysis += f"\n数据集还包含以下信息可供分析：\n"

        # 动态列出可分析的维度
        available_columns = []
        column_mapping = {
            '评分': '书籍评分',
            '评价人数': '评价热度',
            '价格': '价格信息',
            '出版社': '出版社分布',
            '主要作者': '作者作品',
            '出版年份': '出版趋势',
            '价格区间': '价格分布',
            '评分等级': '评分等级分布'
        }

        for col in df.columns:
            if col in column_mapping:
                available_columns.append(column_mapping[col])

        if available_columns:
            analysis += "- " + "\n- ".join(available_columns) + "\n\n"

        analysis += f"你可以具体询问：\n"
        analysis += f"• 特定评分区间的书籍有哪些？\n"
        analysis += f"• 某位作者的所有作品及评分情况\n"
        analysis += f"• 价格与评分的关系分析\n"
        analysis += f"• 不同出版社的书籍质量对比\n"

        return analysis

    def batch_analyze(self, questions: List[str]) -> List[Dict]:
        """批量分析多个问题"""
        results = []
        for question in questions:
            result = self.analyze_with_ai(question)
            results.append(result)
        return results

    def get_analysis_history(self) -> List[Dict]:
        """获取分析历史"""
        return list(self.cache.values())