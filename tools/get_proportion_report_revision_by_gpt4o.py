import os
import json
import sys
from glob import glob

def calculate_revision_ratio():
    if len(sys.argv) != 2:
        print("使用方法: python script.py <文件夹路径>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    json_files = glob(os.path.join(folder_path, '*.json'))
    
    valid_files = 0
    needs_revision = 0
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查必要字段存在性
            if 'revised_med_report' not in data:
                raise KeyError('revised_med_report')
                
            report = data['revised_med_report']
            if 'need revision' not in report:
                raise KeyError('need revision')
            
            # 统计有效文件
            valid_files += 1
            # print(report['need revision'])
            if report['need revision'] is True:
                needs_revision += 1
                
        except KeyError as e:
            print(f"文件 {os.path.basename(file_path)} 缺少关键字段: {e}")
        except json.JSONDecodeError:
            print(f"文件 {os.path.basename(file_path)} 不是有效的JSON格式")
        except Exception as e:
            print(f"处理 {os.path.basename(file_path)} 时发生错误: {str(e)}")
    
    if valid_files == 0:
        print("\n没有找到有效可处理的JSON文件")
        return
    
    ratio = needs_revision / valid_files
    print("\n统计结果:")
    print(f"需要修订的文件数: {needs_revision}")
    print(f"有效文件总数:     {valid_files}")
    print(f"需要修订的比例:   {ratio:.2%}")

if __name__ == "__main__":
    calculate_revision_ratio()