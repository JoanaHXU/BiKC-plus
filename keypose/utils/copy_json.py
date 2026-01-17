import json

input_path = '/home/xuhang/TASE/kp_dataset/aloha2_pants/json_sg/episode_0.json'
output_dir = '/home/xuhang/TASE/kp_dataset/aloha2_pants/json_sg/'
num_episodes = 50

# 读取 episode_0.json 文件
with open(input_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 将数据保存为 episode_1.json

for idx in range(1, num_episodes):
    output_path = f'{output_dir}episode_{idx}.json'
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

print(f"成功将 episode_0.json 的内容复制到 {num_episodes} 个新文件中")