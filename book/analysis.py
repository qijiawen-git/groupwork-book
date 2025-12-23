import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any

warnings.filterwarnings('ignore')


class DoubanAnalyzer:
    def __init__(self, df_path='douban_top250_cleaned.csv'):
        try:
            self.df = pd.read_csv(df_path, encoding='utf-8-sig')
            print(f"成功加载数据，形状: {self.df.shape}")

            self._ensure_data_types()
            self.stats = self._calculate_basic_stats()
            self._generate_all_insights()

        except Exception as e:
            print(f"加载数据失败: {e}")
            raise

    def _ensure_data_types(self):
        self.df['评分'] = pd.to_numeric(self.df['评分'], errors='coerce')
        self.df['评价人数'] = pd.to_numeric(self.df['评价人数'], errors='coerce')
        self.df['价格'] = pd.to_numeric(self.df['价格'], errors='coerce')
        self.df['出版年份'] = pd.to_numeric(self.df['出版年份'], errors='coerce')

        str_columns = ['书名', '作者', '出版社', '出版时间', '简介',
                       '主要作者', '译者编者', '价格区间', '出版年代', '评分等级']
        for col in str_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).fillna('')

    def _to_native_type(self, value):
        if pd.isna(value):
            return None
        elif isinstance(value, (np.integer, np.int64, np.int32, np.int8)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
            return float(value)
        elif isinstance(value, pd.Timestamp):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            return value.to_dict()
        else:
            return value

    def _calculate_basic_stats(self) -> Dict[str, Any]:
        df = self.df

        stats = {
            'total_books': int(len(df)),
            'avg_rating': float(round(df['评分'].mean(), 2)) if not df['评分'].isna().all() else 0.0,
            'median_rating': float(round(df['评分'].median(), 2)) if not df['评分'].isna().all() else 0.0,
            'max_rating': float(round(df['评分'].max(), 2)) if not df['评分'].isna().all() else 0.0,
            'min_rating': float(round(df['评分'].min(), 2)) if not df['评分'].isna().all() else 0.0,
            'avg_price': float(round(df['价格'].mean(), 2)) if not df['价格'].isna().all() else None,
            'total_ratings': int(df['评价人数'].sum()),
            'unique_authors': int(df['主要作者'].nunique()),
            'unique_publishers': int(df['出版社'].nunique()),
            'year_range': f"{int(df['出版年份'].min()) if not df['出版年份'].isna().all() else 0}-{int(df['出版年份'].max()) if not df['出版年份'].isna().all() else 0}",
            'data_quality': {
                'missing_prices': int(df['价格'].isna().sum()),
                'missing_years': int(df['出版年份'].isna().sum()),
                'missing_ratings': int(df['评分'].isna().sum())
            }
        }

        if '出版社' in df.columns:
            top_publishers = df['出版社'].value_counts().head(5)
            stats['top_publishers'] = {
                str(pub): int(count) for pub, count in top_publishers.items()
            }
        else:
            stats['top_publishers'] = {}

        if '主要作者' in df.columns:
            top_authors = df['主要作者'].value_counts().head(5)
            stats['top_authors'] = {
                str(author): int(count) for author, count in top_authors.items()
            }
        else:
            stats['top_authors'] = {}

        return stats

    def _generate_all_insights(self):
        df = self.df

        rating_dist = {}
        if '评分等级' in df.columns:
            for rating_level, count in df['评分等级'].value_counts().items():
                rating_dist[str(rating_level)] = int(count)

        high_rated = df[df['评分'] >= 9.0] if '评分' in df.columns else pd.DataFrame()
        high_rated_count = int(len(high_rated))

        price_analysis = []
        if '价格区间' in df.columns:
            for price_range, group in df.groupby('价格区间'):
                price_analysis.append({
                    'price_range': str(price_range),
                    'book_count': int(len(group)),
                    'avg_rating': float(group['评分'].mean() if '评分' in group.columns else 0),
                    'avg_ratings_count': float(group['评价人数'].mean() if '评价人数' in group.columns else 0)
                })

        year_analysis = []
        if '出版年份' in df.columns:
            for year, group in df.groupby('出版年份'):
                if not pd.isna(year):
                    year_analysis.append({
                        'year': int(year),
                        'book_count': int(len(group)),
                        'avg_rating': float(group['评分'].mean() if '评分' in group.columns else 0)
                    })
            year_analysis = sorted(year_analysis, key=lambda x: x['year'])[-20:]

        if '评分' in df.columns and '价格' in df.columns:
            cheap_high_quality = df[(df['评分'] >= 9.0) & (df['价格'] <= 30)]
            value_books_count = int(len(cheap_high_quality))
        else:
            value_books_count = 0

        self.insights = {
            'rating_distribution': rating_dist,
            'high_rated_count': high_rated_count,
            'price_analysis': price_analysis,
            'year_analysis': year_analysis,
            'value_books_count': value_books_count,
            'total_insights': []
        }

        self._generate_text_insights()

    def _generate_text_insights(self):
        insights = []

        insights.append(f"豆瓣Top250包含{self.stats['total_books']}本书籍")
        insights.append(f"平均评分: {self.stats['avg_rating']}分，中位数: {self.stats['median_rating']}分")

        total_ratings = self.stats['total_ratings']
        if total_ratings >= 1000000:
            formatted = f"{total_ratings / 1000000:.1f}百万"
        elif total_ratings >= 10000:
            formatted = f"{total_ratings / 10000:.1f}万"
        else:
            formatted = f"{total_ratings:,}"
        insights.append(f"总评价次数: {formatted}次")

        insights.append(f"9分以上的神作有{self.insights['high_rated_count']}本")

        if self.stats['avg_price']:
            insights.append(f"平均价格: {self.stats['avg_price']}元")
        else:
            insights.append("平均价格: 数据缺失")

        if self.stats['top_authors']:
            top_authors = list(self.stats['top_authors'].items())[:3]
            author_str = [f"{author}({count}本)" for author, count in top_authors]
            insights.append(f"作品最多的作者: {', '.join(author_str)}")

        if self.stats['top_publishers']:
            top_pubs = list(self.stats['top_publishers'].items())[:3]
            pub_str = [f"{pub}({count}本)" for pub, count in top_pubs]
            insights.append(f"出版最多的出版社: {', '.join(pub_str)}")

        self.insights['text_insights'] = insights

    def get_data_summary(self) -> Dict[str, Any]:
        try:
            if not hasattr(self, 'insights'):
                self._generate_all_insights()

            if 'text_insights' not in self.insights:
                self._generate_text_insights()

            return {
                'statistics': self.stats,
                'insights': self.insights.get('text_insights', []),
                'data_quality': self.stats.get('data_quality', {})
            }
        except Exception as e:
            print(f"获取数据摘要时出错: {e}")
            return {
                'statistics': self.stats,
                'insights': ["数据摘要生成中..."],
                'data_quality': {}
            }

    def get_top_books(self, category='rating', limit=10) -> List[Dict[str, Any]]:
        df = self.df.copy()

        if category == 'rating':
            result = df.nlargest(limit, '评分')[['书名', '评分', '评价人数', '主要作者', '出版社']]
        elif category == 'popularity':
            result = df.nlargest(limit, '评价人数')[['书名', '评价人数', '评分', '主要作者', '出版社']]
        elif category == 'value':
            df_filtered = df[df['价格'].notna()]
            if not df_filtered.empty:
                df_filtered['性价比'] = df_filtered['评分'] / df_filtered['价格']
                result = df_filtered.nlargest(limit, '性价比')[['书名', '评分', '价格', '主要作者', '出版社']]
            else:
                result = pd.DataFrame()
        elif category == 'recent':
            result = df.nlargest(limit, '出版年份')[['书名', '出版年份', '评分', '评价人数', '主要作者']]
        else:
            result = df.head(limit)

        books = []
        if not result.empty:
            for _, row in result.iterrows():
                book_dict = {}
                for col in result.columns:
                    book_dict[col] = self._to_native_type(row[col])
                books.append(book_dict)

        return books

    def get_publisher_analysis(self, limit=15) -> Dict[str, Any]:
        if '出版社' not in self.df.columns:
            return {
                'top_by_count': [],
                'top_by_rating': [],
                'summary': {'total_publishers': 0, 'avg_books_per_publisher': 0}
            }

        publisher_stats = self.df.groupby('出版社').agg({
            '书名': 'count',
            '评分': 'mean',
            '评价人数': 'mean',
            '价格': 'mean'
        }).round(2).reset_index()

        publisher_stats.columns = ['出版社', '出版数量', '平均评分', '平均评价人数', '平均价格']

        top_by_count = []
        for _, row in publisher_stats.sort_values('出版数量', ascending=False).head(limit).iterrows():
            top_by_count.append({
                '出版社': str(row['出版社']),
                '出版数量': int(row['出版数量']),
                '平均评分': float(row['平均评分']),
                '平均评价人数': float(row['平均评价人数']),
                '平均价格': float(row['平均价格']) if not pd.isna(row['平均价格']) else None
            })

        top_by_rating = []
        for _, row in publisher_stats.sort_values('平均评分', ascending=False).head(limit).iterrows():
            top_by_rating.append({
                '出版社': str(row['出版社']),
                '出版数量': int(row['出版数量']),
                '平均评分': float(row['平均评分']),
                '平均评价人数': float(row['平均评价人数']),
                '平均价格': float(row['平均价格']) if not pd.isna(row['平均价格']) else None
            })

        return {
            'top_by_count': top_by_count,
            'top_by_rating': top_by_rating,
            'summary': {
                'total_publishers': int(publisher_stats.shape[0]),
                'avg_books_per_publisher': float(round(publisher_stats['出版数量'].mean(), 1))
            }
        }

    def get_author_analysis(self, limit=15) -> Dict[str, Any]:
        if '主要作者' not in self.df.columns:
            return {
                'top_by_count': [],
                'top_by_popularity': [],
                'summary': {'total_authors': 0, 'avg_works_per_author': 0}
            }

        author_stats = self.df.groupby('主要作者').agg({
            '书名': 'count',
            '评分': 'mean',
            '评价人数': 'sum'
        }).round(2).reset_index()

        author_stats.columns = ['作者', '作品数量', '平均评分', '总评价人数']

        top_by_count = []
        for _, row in author_stats.sort_values('作品数量', ascending=False).head(limit).iterrows():
            top_by_count.append({
                '作者': str(row['作者']),
                '作品数量': int(row['作品数量']),
                '平均评分': float(row['平均评分']),
                '总评价人数': int(row['总评价人数'])
            })

        top_by_popularity = []
        for _, row in author_stats.sort_values('总评价人数', ascending=False).head(limit).iterrows():
            top_by_popularity.append({
                '作者': str(row['作者']),
                '作品数量': int(row['作品数量']),
                '平均评分': float(row['平均评分']),
                '总评价人数': int(row['总评价人数'])
            })

        return {
            'top_by_count': top_by_count,
            'top_by_popularity': top_by_popularity,
            'summary': {
                'total_authors': int(author_stats.shape[0]),
                'avg_works_per_author': float(round(author_stats['作品数量'].mean(), 1))
            }
        }

    def get_trend_analysis(self) -> Dict[str, Any]:
        year_trend = []
        if '出版年份' in self.df.columns:
            year_groups = self.df.groupby('出版年份')
            for year, group in year_groups:
                if not pd.isna(year) and int(year) >= 1900:
                    year_trend.append({
                        'year': int(year),
                        'book_count': int(len(group)),
                        'avg_rating': float(group['评分'].mean() if '评分' in group.columns else 0),
                        'avg_ratings_count': float(group['评价人数'].mean() if '评价人数' in group.columns else 0)
                    })
            year_trend = sorted(year_trend, key=lambda x: x['year'])

        price_trend = []
        if '价格区间' in self.df.columns:
            price_groups = self.df.groupby('价格区间')
            for price_range, group in price_groups:
                price_trend.append({
                    'price_range': str(price_range),
                    'book_count': int(len(group)),
                    'avg_rating': float(group['评分'].mean() if '评分' in group.columns else 0)
                })

        return {
            'year_trend': year_trend,
            'price_trend': price_trend
        }

    # 在 analysis.py 的 generate_visualization_data 方法中添加热门作者图表数据
    def generate_visualization_data(self) -> Dict[str, Any]:
        df = self.df

        # 评分分布
        rating_bins = np.arange(7.5, 10.1, 0.2)
        rating_counts = []
        for i in range(len(rating_bins) - 1):
            count = len(df[(df['评分'] >= rating_bins[i]) & (df['评分'] < rating_bins[i + 1])])
            rating_counts.append({
                'range': f'{float(rating_bins[i]):.1f}-{float(rating_bins[i + 1]):.1f}',
                'count': int(count)
            })

        # 价格分布
        price_dist = []
        if '价格区间' in df.columns:
            for price_range, count in df['价格区间'].value_counts().items():
                price_dist.append({
                    'price_range': str(price_range),
                    'count': int(count)
                })

        # 热门出版社
        top_publishers = []
        if '出版社' in df.columns:
            publisher_counts = df['出版社'].value_counts().head(10)
            for publisher, count in publisher_counts.items():
                publisher_books = df[df['出版社'] == publisher]
                avg_rating = publisher_books['评分'].mean() if not publisher_books.empty else 0
                top_publishers.append({
                    'publisher': str(publisher),
                    'count': int(count),
                    'avg_rating': float(round(avg_rating, 2))
                })

        # 热门作者 - 修复此部分
        top_authors = []
        if '主要作者' in df.columns:
            # 过滤空作者
            valid_authors = df[df['主要作者'].notna() & (df['主要作者'].str.strip() != '')]
            if not valid_authors.empty:
                author_counts = valid_authors['主要作者'].value_counts().head(10)
                for author, count in author_counts.items():
                    author_books = df[df['主要作者'] == author]
                    avg_rating = author_books['评分'].mean() if not author_books.empty else 0
                    total_ratings = author_books['评价人数'].sum() if not author_books.empty else 0
                    top_authors.append({
                        'author': str(author),
                        'count': int(count),
                        'avg_rating': float(round(avg_rating, 2)),
                        'total_ratings': int(total_ratings)
                    })

        # 出版年份趋势
        year_trend = []
        if '出版年份' in df.columns:
            year_counts = df['出版年份'].value_counts().sort_index()
            for year, count in year_counts.items():
                if not pd.isna(year):
                    year_trend.append({
                        'year': int(year),
                        'count': int(count)
                    })

        # 评分 vs 评价人数散点图数据
        scatter_data = []
        scatter_df = df[['评分', '评价人数', '书名', '主要作者', '价格']].copy()
        scatter_df = scatter_df.dropna(subset=['评分', '评价人数'])

        # 限制数据量
        if len(scatter_df) > 100:
            scatter_df = scatter_df.sample(100, random_state=42)

        for _, row in scatter_df.iterrows():
            scatter_data.append({
                '评分': float(row['评分']),
                '评价人数': int(row['评价人数']),
                '书名': str(row['书名']),
                '作者': str(row['主要作者']) if not pd.isna(row['主要作者']) else '未知',
                '价格': float(row['价格']) if not pd.isna(row['价格']) else None
            })

        # 评分与价格关系散点图
        price_scatter_data = []
        price_scatter_df = df[['评分', '价格', '书名', '主要作者']].copy()
        price_scatter_df = price_scatter_df.dropna(subset=['评分', '价格'])

        if len(price_scatter_df) > 100:
            price_scatter_df = price_scatter_df.sample(100, random_state=42)

        for _, row in price_scatter_df.iterrows():
            price_scatter_data.append({
                '评分': float(row['评分']),
                '价格': float(row['价格']),
                '书名': str(row['书名']),
                '作者': str(row['主要作者']) if not pd.isna(row['主要作者']) else '未知'
            })

        return {
            'rating_distribution': rating_counts,
            'price_distribution': price_dist,
            'top_publishers': top_publishers,
            'top_authors': top_authors,
            'year_trend': year_trend,
            'rating_vs_ratings': scatter_data,  # 评分 vs 评价人数
            'rating_vs_price': price_scatter_data,  # 评分 vs 价格
            'data_summary': {
                'total_books': len(df),
                'avg_rating': float(round(df['评分'].mean(), 2)),
                'total_authors': df['主要作者'].nunique() if '主要作者' in df.columns else 0,
                'total_publishers': df['出版社'].nunique() if '出版社' in df.columns else 0
            }
        }
    def analyze_custom_query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        df = self.df.copy()
        filters = {}

        if 'min_rating' in query_params:
            min_rating = float(query_params['min_rating'])
            df = df[df['评分'] >= min_rating]
            filters['min_rating'] = min_rating

        if 'max_price' in query_params:
            max_price = float(query_params['max_price'])
            df = df[df['价格'] <= max_price]
            filters['max_price'] = max_price

        if 'publisher' in query_params:
            publisher = query_params['publisher']
            df = df[df['出版社'].str.contains(publisher, na=False)]
            filters['publisher'] = publisher

        if 'author' in query_params:
            author = query_params['author']
            df = df[df['主要作者'].str.contains(author, na=False)]
            filters['author'] = author

        if 'sort_by' in query_params:
            sort_by = query_params['sort_by']
            ascending = query_params.get('ascending', 'false').lower() == 'true'
            df = df.sort_values(sort_by, ascending=ascending)

        limit = int(query_params.get('limit', 20))

        books = []
        for _, row in df.head(limit).iterrows():
            book_dict = {}
            for col in df.columns:
                book_dict[col] = self._to_native_type(row[col])
            books.append(book_dict)

        return {
            'filters_applied': filters,
            'result_count': int(len(df)),
            'books': books,
            'summary': {
                'avg_rating': float(round(df['评分'].mean(), 2)) if len(df) > 0 else 0.0,
                'avg_price': float(round(df['价格'].mean(), 2)) if len(df) > 0 else 0.0
            }
        }