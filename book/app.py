import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import pandas as pd
import json
import traceback
import os
from datetime import datetime

from analysis import DoubanAnalyzer
from ai_service import DoubanAIAnalyzer

app = Flask(__name__, static_folder='.')
CORS(app)

analyzer = None
ai_analyzer = None

def init_analyzers():
    global analyzer, ai_analyzer

    try:
        print("正在初始化数据分析器...")
        analyzer = DoubanAnalyzer('douban_top250_cleaned.csv')
        print("数据分析器初始化完成")

        ai_analyzer = DoubanAIAnalyzer(analyzer)
        print("AI分析器初始化完成")
        return True

    except Exception as e:
        print(f"初始化失败: {e}")
        traceback.print_exc()
        return False

init_analyzers()

@app.route('/')
def index():
    return send_from_directory('.', 'dashboard.html')

@app.route('/<path:filename>')
def static_files(filename):
    if filename.endswith('.html') or filename.endswith('.js') or filename.endswith('.css'):
        try:
            return send_from_directory('.', filename)
        except:
            return "文件不存在", 404
    return send_from_directory('.', filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'success': True,
        'status': 'healthy',
        'service': 'douban-analysis-api',
        'timestamp': datetime.now().isoformat(),
        'data_loaded': analyzer is not None,
        'version': '1.0.0'
    })

@app.route('/api/data/summary', methods=['GET'])
def get_data_summary():
    try:
        if analyzer is None:
            return jsonify({
                'success': False,
                'error': '数据分析器未初始化'
            }), 500

        summary = analyzer.get_data_summary()
        return jsonify({
            'success': True,
            'data': summary,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"获取数据摘要错误: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/data/top-books', methods=['GET'])
def get_top_books():
    try:
        if analyzer is None:
            return jsonify({'success': False, 'error': '数据分析器未初始化'}), 500

        category = request.args.get('category', 'rating')
        limit = int(request.args.get('limit', 10))

        books = analyzer.get_top_books(category, limit)

        return jsonify({
            'success': True,
            'category': category,
            'limit': limit,
            'count': len(books),
            'books': books,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/data/publishers', methods=['GET'])
def get_publisher_analysis():
    try:
        if analyzer is None:
            return jsonify({'success': False, 'error': '数据分析器未初始化'}), 500

        limit = int(request.args.get('limit', 15))
        analysis = analyzer.get_publisher_analysis(limit)

        return jsonify({
            'success': True,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/data/authors', methods=['GET'])
def get_author_analysis():
    try:
        if analyzer is None:
            return jsonify({'success': False, 'error': '数据分析器未初始化'}), 500

        limit = int(request.args.get('limit', 15))
        analysis = analyzer.get_author_analysis(limit)

        return jsonify({
            'success': True,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/data/trends', methods=['GET'])
def get_trend_analysis():
    try:
        if analyzer is None:
            return jsonify({'success': False, 'error': '数据分析器未初始化'}), 500

        trends = analyzer.get_trend_analysis()

        return jsonify({
            'success': True,
            'trends': trends,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/data/visualization', methods=['GET'])
def get_visualization_data():
    try:
        if analyzer is None:
            return jsonify({'success': False, 'error': '数据分析器未初始化'}), 500

        viz_data = analyzer.generate_visualization_data()

        return jsonify({
            'success': True,
            'visualizations': viz_data,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# 在 app.py 中修改 search_books 函数
@app.route('/api/data/search', methods=['GET'])
def search_books():
    try:
        if analyzer is None:
            return jsonify({'success': False, 'error': '数据分析器未初始化'}), 500

        query = request.args.get('q', '')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 20))
        sort_by = request.args.get('sort_by', '评分')  # 添加排序参数
        ascending = request.args.get('ascending', 'false').lower() == 'true'

        # 获取数据
        results = analyzer.df.copy()

        # 搜索逻辑
        if query:
            # 构建搜索条件
            search_conditions = []

            # 搜索书名
            if '书名' in results.columns:
                mask_title = results['书名'].str.contains(query, case=False, na=False)
                search_conditions.append(mask_title)

            # 搜索作者
            if '主要作者' in results.columns:
                mask_author = results['主要作者'].str.contains(query, case=False, na=False)
                search_conditions.append(mask_author)

            # 搜索出版社
            if '出版社' in results.columns:
                mask_publisher = results['出版社'].str.contains(query, case=False, na=False)
                search_conditions.append(mask_publisher)

            # 搜索简介
            if '简介' in results.columns:
                mask_intro = results['简介'].str.contains(query, case=False, na=False)
                search_conditions.append(mask_intro)

            # 合并所有条件
            if search_conditions:
                combined_mask = search_conditions[0]
                for condition in search_conditions[1:]:
                    combined_mask = combined_mask | condition
                results = results[combined_mask]

        # 排序逻辑
        if sort_by in results.columns:
            results = results.sort_values(by=sort_by, ascending=ascending, na_position='last')

        # 分页逻辑
        total = len(results)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        # 转换为字典记录
        paginated_results = []
        if total > 0:
            paginated_data = results.iloc[start_idx:end_idx]
            paginated_results = []
            for _, row in paginated_data.iterrows():
                book_dict = {}
                for col in results.columns:
                    value = row[col]
                    # 转换特殊类型
                    if pd.isna(value):
                        book_dict[col] = None
                    elif isinstance(value, (np.integer, np.int64)):
                        book_dict[col] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        book_dict[col] = float(value)
                    else:
                        book_dict[col] = str(value)
                paginated_results.append(book_dict)

        return jsonify({
            'success': True,
            'query': query,
            'page': page,
            'page_size': page_size,
            'total': total,
            'total_pages': max(1, (total + page_size - 1) // page_size),
            'books': paginated_results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"搜索书籍错误: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500
@app.route('/api/ai/analyze', methods=['POST'])
def analyze_with_ai():
    try:
        if ai_analyzer is None:
            return jsonify({
                'success': False,
                'error': 'AI分析器未初始化'
            }), 500

        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '未提供数据'}), 400

        question = data.get('question', '')
        if not question:
            return jsonify({'success': False, 'error': '未提供问题'}), 400

        result = ai_analyzer.analyze_with_ai(question)

        return jsonify({
            'success': True,
            'analysis': result,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    try:
        format_type = request.args.get('format', 'cleaned')

        if format_type == 'raw':
            file_path = 'douban_top250_books.csv'
            filename = 'douban_top250_raw.csv'
        else:
            file_path = 'douban_top250_cleaned.csv'
            filename = 'douban_top250_cleaned.csv'

        return send_file(
            file_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export/json', methods=['GET'])
def export_json():
    try:
        if analyzer is None:
            return jsonify({'success': False, 'error': '数据分析器未初始化'}), 500

        df = analyzer.df
        json_data = df.to_dict('records')

        return jsonify({
            'success': True,
            'count': len(json_data),
            'data': json_data,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/debug/columns', methods=['GET'])
def debug_columns():
    try:
        if analyzer is None:
            return jsonify({'success': False, 'error': '数据分析器未初始化'}), 500

        columns = analyzer.df.columns.tolist()
        sample = analyzer.df.head(3).to_dict('records')

        return jsonify({
            'success': True,
            'columns': columns,
            'sample': sample,
            'shape': analyzer.df.shape,
            'row_count': len(analyzer.df)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/debug/stats', methods=['GET'])
def debug_stats():
    try:
        if analyzer is None:
            return jsonify({'success': False, 'error': '数据分析器未初始化'}), 500

        return jsonify({
            'success': True,
            'stats': analyzer.stats,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': '请求的资源不存在',
        'path': request.path
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': '服务器内部错误',
        'message': str(error) if str(error) else '未知错误'
    }), 500

if __name__ == '__main__':
    if analyzer is not None:
        print("=" * 60)
        print("豆瓣图书Top250分析系统启动成功!")
        print(f"仪表板地址: http://localhost:5000")
        print(f"API服务地址: http://localhost:5000/api")
        print("数据概览:")
        print(f"   总书籍数: {analyzer.stats['total_books']}")
        print(f"   平均评分: {analyzer.stats['avg_rating']}")
        print(f"   总评价次数: {analyzer.stats['total_ratings']:,}")
        print(f"   出版社数量: {analyzer.stats['unique_publishers']}")
        print(f"   作者数量: {analyzer.stats['unique_authors']}")
        print("=" * 60)

        app.run(
            debug=True,
            host='127.0.0.1',
            port=5000,
            threaded=True
        )
    else:
        print("系统启动失败，请检查数据文件是否存在!")
        print("请确保有以下文件：")
        print("  - douban_top250_cleaned.csv")
        print("  - douban_top250_books.csv")