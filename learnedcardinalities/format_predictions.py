import csv

input_file = 'results/predictions_test.csv'
output_file = 'results/predictions_test_formatted.csv'

with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    writer.writerow(['Query ID', 'Predicted Cardinality'])
    for idx, row in enumerate(reader):
        # 去除中括号
        pred = row[0].replace('[','').replace(']','')
        writer.writerow([idx, pred])

print(f"格式化完成，结果已保存到 {output_file}")