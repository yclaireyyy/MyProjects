# 构造棋盘
BOARD = [['jk','2s','3s','4s','5s','6s','7s','8s','9s','jk'],
         ['6c','5c','4c','3c','2c','ah','kh','qh','th','ts'],
         ['7c','as','2d','3d','4d','5d','6d','7d','9h','qs'],
         ['8c','ks','6c','5c','4c','3c','2c','8d','8h','ks'],
         ['9c','qs','7c','6h','5h','4h','ah','9d','7h','as'],
         ['tc','ts','8c','7h','2h','3h','kh','td','6h','2d'],
         ['qc','9s','9c','8h','9h','th','qh','qd','5h','3d'],
         ['kc','8s','tc','qc','kc','ac','ad','kd','4h','4d'],
         ['ac','7s','6s','5s','4s','3s','2s','2h','3h','5d'],
         ['jk','ad','kd','qd','td','9d','8d','7d','6d','jk']]

# 生成映射
from collections import defaultdict
card_to_coords = defaultdict(list)

for row in range(10):
    for col in range(10):
        card = BOARD[row][col]
        card_to_coords[card].append((row, col))

# 转换为普通字典，准备展示
card_to_coords = dict(card_to_coords)
card_to_coords_sorted = dict(sorted(card_to_coords.items()))
print(card_to_coords_sorted)
