import requests
from bs4 import BeautifulSoup
import time
import csv
def fetch_douban_top250():
    """
    爬取豆瓣图书Top250信息
    """
    base_url = "https://book.douban.com/top250"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    all_books = []
    # 豆瓣Top250共有10页，每页25本书
    for page in range(10):
        try:
            # 构建每一页的URL
            if page == 0:
                url = f"{base_url}?icn=index-book250-all"
            else:
                start = page * 25
                url = f"{base_url}?start={start}"

            print(f"正在爬取第 {page + 1} 页: {url}")
            # 发送请求
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # 检查请求是否成功
            response.encoding = 'utf-8'
            # 解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            # 找到所有的书籍条目
            book_items = soup.find_all('tr', class_='item')
            for item in book_items:
                book_info = extract_book_info(item)
                if book_info:
                    all_books.append(book_info)

            # 避免请求过快，添加延迟
            time.sleep(2)

        except requests.RequestException as e:
            print(f"爬取第 {page + 1} 页时出错: {e}")
            continue
    return all_books

def extract_book_info(item):
    """
    从单个书籍条目中提取信息
    """
    try:
        # 1. 书名
        title_elem = item.find('div', class_='pl2').find('a')
        title = title_elem.get('title', '').strip() if title_elem.get('title') else title_elem.text.strip()
        # 2. 作者、出版社、出版时间、价格信息
        pl_elem = item.find('p', class_='pl')
        if pl_elem:
            pl_text = pl_elem.text.strip()
            # 分割字符串获取详细信息
            parts = pl_text.split(' / ')
            author = parts[0] if len(parts) > 0 else ''
            publisher = parts[1] if len(parts) > 1 else ''
            publish_date = parts[2] if len(parts) > 2 else ''
            price = parts[3] if len(parts) > 3 else ''
        else:
            author = publisher = publish_date = price = ''

        # 3. 评分
        rating_elem = item.find('span', class_='rating_nums')
        rating = rating_elem.text.strip() if rating_elem else ''

        # 4. 评价人数
        rating_people_elem = item.find('span', class_='pl')
        rating_people = ''
        if rating_people_elem:
            rating_people_text = rating_people_elem.text.strip()
            # 提取数字部分
            rating_people = rating_people_text.replace('(', '').replace(')', '').replace('人评价', '')
        intro_elem = item.find('span', class_='inq')
        intro = intro_elem.text.strip() if intro_elem else ''
        # 6. 书籍链接（可选）
        link_elem = item.find('div', class_='pl2').find('a')
        book_link = link_elem.get('href', '') if link_elem else ''
        return {
            '书名': title,
            '作者': author,
            '出版社': publisher,
            '出版时间': publish_date,
            '价格': price,
            '评分': rating,
            '评价人数': rating_people,
            '简介': intro,
            '书籍链接': book_link
        }

    except Exception as e:
        print(f"提取书籍信息时出错: {e}")
        return None


def save_to_csv(books, filename='douban_top250_books.csv'):
    """
    将数据保存到CSV文件
    """
    if not books:
        print("没有数据可保存")
        return

    # 定义CSV文件的列名
    fieldnames = ['书名', '作者', '出版社', '出版时间', '价格', '评分', '评价人数', '简介', '书籍链接']

    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for book in books:
                writer.writerow(book)
        print(f"数据已保存到 {filename}，共 {len(books)} 条记录")
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")


def display_sample(books, num=5):
    """
    显示样本数据
    """
    print(f"\n=== 前 {num} 本书籍信息样本 ===")
    for i, book in enumerate(books[:num], 1):
        print(f"\n{i}. 《{book['书名']}》")
        print(f"   作者: {book['作者']}")
        print(f"   出版社: {book['出版社']}")
        print(f"   出版时间: {book['出版时间']}")
        print(f"   价格: {book['价格']}")
        print(f"   评分: {book['评分']}")
        print(f"   评价人数: {book['评价人数']}")
        print(f"   简介: {book['简介']}")


def main():
    """
    主函数
    """
    print("开始爬取豆瓣图书Top250...")

    # 爬取数据
    books = fetch_douban_top250()

    if books:
        print(f"\n爬取完成，共获取 {len(books)} 本书籍信息")

        # 显示样本
        display_sample(books)

        # 保存到CSV
        save_to_csv(books)
    else:
        print("未能爬取到任何数据")
if __name__ == "__main__":
    main()