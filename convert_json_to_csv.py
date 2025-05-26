import json
import csv
import os
import re

def parse_query(query_str):
    # 提取表名
    tables = []
    table_pattern = r'FROM\s+([^;]+?)(?:\s+WHERE|\s*;)'
    table_matches = re.findall(table_pattern, query_str, re.IGNORECASE)
    if table_matches:
        tables = [t.strip() for t in table_matches[0].split(',')]
    
    # 提取连接条件
    joins = []
    join_pattern = r'(\w+\.\w+\s*=\s*\w+\.\w+)'
    joins = re.findall(join_pattern, query_str)
    
    # 提取谓词条件
    predicates = []
    where_pattern = r'WHERE\s+(.+?);'
    where_matches = re.findall(where_pattern, query_str, re.IGNORECASE)
    if where_matches:
        # 分割AND条件
        conditions = [p.strip() for p in where_matches[0].split('AND')]
        for condition in conditions:
            # 使用正则表达式匹配列名、操作符和值
            match = re.match(r'(\w+\.\w+)\s*([<>=!]+)\s*(\d+)', condition)
            if match:
                col, op, val = match.groups()
                # 将条件转换为逗号分隔的格式
                predicates.append(f"{col},{op},{val}")
    
    return tables, joins, predicates

def parse_explain_result(explain_str, use_plan_rows=False):
    try:
        explain_data = json.loads(explain_str)
        plan = explain_data.get('QUERY PLAN', [{}])[0].get('Plan', {})
        if use_plan_rows:
            return str(plan.get('Plan Rows', 0))
        else:
            return str(plan.get('Actual Rows', 0))
    except:
        return "0"

def convert_json_to_csv(json_file, csv_file, use_plan_rows=False):
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 创建CSV文件
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='#')
        
        # 处理每个查询
        for query_data in data:
            # 解析查询
            tables, joins, predicates = parse_query(query_data['query'])
            
            # 解析执行计划获取基数
            cardinality = parse_explain_result(query_data['explain_result'], use_plan_rows)
            
            # 转换为字符串
            tables_str = ','.join(tables)
            joins_str = ','.join(joins)
            predicates_str = ','.join(predicates)
            
            # 写入CSV
            writer.writerow([tables_str, joins_str, predicates_str, cardinality])

def main():
    # 确保data目录存在
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 转换训练数据（使用Actual Rows）
    convert_json_to_csv('data/train_data.json', 'data/train.csv', use_plan_rows=False)
    print("已转换训练数据")
    
    # 转换测试数据（使用Plan Rows）
    convert_json_to_csv('data/test_data.json', 'data/test.csv', use_plan_rows=True)
    print("已转换测试数据")

if __name__ == '__main__':
    main() 