import json
import os
import copy

def fix_round_numbering(json_data):
    """修复JSON文件中对话轮次的编号"""
    conversations = json_data["conversations"]
    fixed_conversations = copy.deepcopy(conversations)
    
    # 重新计算轮次编号
    current_round = 1
    role_count = {}  # 记录每轮中角色出现次数
    
    for i, entry in enumerate(fixed_conversations):
        role = entry["role"]
        
        # 如果是新轮次的开始（当前角色在本轮已经出现过）
        if role in role_count:
            current_round += 1
            role_count = {role: 1}  # 重置角色计数，并计入当前角色
        else:
            role_count[role] = 1
        
        # 更新轮次编号
        entry["round"] = current_round
    
    # 更新JSON数据
    result = copy.deepcopy(json_data)
    result["conversations"] = fixed_conversations
    return result

def process_json_files(directory):
    """处理目录中的所有JSON文件"""
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"错误: 目录不存在 {directory}")
        return
    
    # 计数器
    total_files = 0
    modified_files = 0
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            total_files += 1
            
            try:
                # 读取JSON文件
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 修复轮次编号
                fixed_data = fix_round_numbering(data)
                
                # 检查是否有修改
                if fixed_data != data:
                    modified_files += 1
                    
                    # 创建备份
                    backup_path = filepath + '.bak'
                    os.rename(filepath, backup_path)
                    
                    # 保存修复后的JSON文件
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(fixed_data, f, indent=4, ensure_ascii=False)
                    
                    print(f"修复: {filename}")
                else:
                    print(f"无需修复: {filename}")
                    
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    
    # 打印汇总信息
    print(f"\n处理完成! 总文件数: {total_files}, 修复的文件数: {modified_files}")

if __name__ == "__main__":
    # 处理指定目录中的所有JSON文件
    directory_path = "/hpc2hdd/home/yfan546/workplace/xray_teeth/unlabeled_data/multi-turn"
    process_json_files(directory_path)
