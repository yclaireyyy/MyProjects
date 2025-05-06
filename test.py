from Sequence.sequence_model import SequenceGameRule
from copy import deepcopy

def test_sequence(chips, coords, player_colour):
    # 构建一个正常初始化的游戏状态（包含两个玩家）
    rule = SequenceGameRule(2)
    state = rule.initialGameState()

    # 获取一名玩家并强制设定颜色
    fake_player = state.agents[0]
    fake_player.colour = player_colour
    fake_player.opp_colour = 'b' if player_colour == 'r' else 'r'
    fake_player.seq_colour = 'X' if player_colour == 'r' else 'O'
    fake_player.opp_seq_colour = 'O' if player_colour == 'r' else 'X'

    # 模拟落子
    test_chips = deepcopy(chips)
    r, c = coords
    test_chips[r][c] = player_colour

    # 执行检测
    result, seq_type = rule.checkSeq(test_chips, fake_player, coords)

    print(f"落子位置：{coords}")
    print(f"玩家颜色：{player_colour}")
    if result:
        print(f"得分类型：{seq_type}")
        print(f"获得分数：{result['num_seq']}")
        print(f"触发方向：{result['orientation']}")
        print(f"触发坐标：{result['coords']}")
        return result['num_seq']
    else:
        print("未形成任何有效的sequence。")
        return 0

# 示例：构造一个即将完成横向五连的场景
chips = [['_' for _ in range(10)] for _ in range(10)]
for i in range(9):
    if i < 5:
        chips[5][i] = 'X'  # 第5行，列0~3为红色棋子
    if i > 5:
        chips[5][i] = 'r'  # 第5行，列0~3为红色棋子
print(chips[5])

test_sequence(chips, (5, 5), 'r')  # 在第5行第4列落子，预期触发 sequence 得分
