import math
import numpy as np
import torch.nn.functional as F

import argparse
import torch
import torch.optim as optim
from environment import TetrisEnv  # 사용자가 만든 env
from model import AlphaZeroPolicyValueNet
from torch.utils.tensorboard import SummaryWriter

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # numpy array
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.N = 0  # 방문 횟수
        self.W = 0.0  # 누적 가치
        self.Q = 0.0  # 평균 가치
        self.P = 0.0  # policy prior

    def is_leaf(self):
        return len(self.children)==0


def run_mcts(root_node, env, model, non_huristic=False, simulations=20, max_depth=2, c_puct=1.0, device='cpu'):
    """
    MCTS 스켈레톤:
    - 각 시뮬레이션마다 Selection/Expansion/Simulation/Backprop
    - leaf 평가 시 model로 (policy, value) 예측
    - 액션 마스킹 적용
    """
    import copy
    for _ in range(simulations):
        node = root_node
        depth = 0

        # env 복제
        sim_env = copy.deepcopy(env)
        
        if env.non_huristic:
            sim_env.non_huristic = True
        
        # 1) Selection
        while (not node.is_leaf()) and (depth < max_depth):
            best_score = -999999
            best_a = None
            best_child = None

            # 자식 노드가 없으면 => break
            if not node.children:
                break

            # 유효한 액션만 고려
            valid_moves = sim_env.get_valid_moves(sim_env.current_piece['type'])
            valid_actions = []
            for rot, x in valid_moves:
                action_id = rot * 10 + x
                if action_id < 40:
                    valid_actions.append(action_id)
            if sim_env.can_hold:
                valid_actions.append(40)

            for a in valid_actions:
                if a in node.children:
                    child = node.children[a]
                    # PUCT
                    U = c_puct * child.P * math.sqrt(node.N+1)/(child.N+1)
                    score = child.Q + U
                    if score > best_score:
                        best_score = score
                        best_a = a
                        best_child = child

            # sim_env step
            if best_a is None:
                break

            _, reward_, done_, _, _ = sim_env.step(best_a)
            node = best_child
            depth += 1
            if done_:
                break
        
        # 2) Expansion
        if node.is_leaf() and (depth < max_depth) and (not sim_env.done):
            # 유효한 액션 마스크 생성
            valid_moves = sim_env.get_valid_moves(sim_env.current_piece['type'])
            action_mask = np.zeros(41, dtype=np.float32)
            for rot, x in valid_moves:
                action_id = rot * 10 + x
                if action_id < 40:
                    action_mask[action_id] = 1.0
            if sim_env.can_hold:
                action_mask[40] = 1.0

            # model inference with masking
            import torch
            state_t = torch.FloatTensor(node.state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, value = model(state_t)
                # 마스크 적용 (크기를 맞춰줌)
                logits = logits.cpu().numpy()[0]  # shape: (41,)
                logits = logits - (1 - action_mask) * 1e9  # 무효 액션에 큰 음수
                policy = F.softmax(torch.FloatTensor(logits), dim=0).numpy()
                v = value.item()

            # 유효한 액션에 대해서만 자식 노드 생성
            for a in np.where(action_mask > 0)[0]:
                sim_env_copy = copy.deepcopy(sim_env)
                next_state, _, _, _, _ = sim_env_copy.step(a)
                child = MCTSNode(state=next_state, parent=node)
                child.P = policy[a]
                node.children[a] = child
            leaf_value = v
        else:
            # leaf_value
            import torch
            state_t = torch.FloatTensor(node.state).unsqueeze(0).to(device)
            with torch.no_grad():
                _, value = model(state_t)
                leaf_value = value.item()

        # 3) Backprop
        bnode = node
        ret = leaf_value
        while bnode is not None:
            bnode.N += 1
            bnode.W += ret
            bnode.Q = bnode.W / bnode.N
            bnode = bnode.parent

    # 4) 루트 노드에서 정책 분포 계산 (유효한 액션만)
    valid_moves = env.get_valid_moves(env.current_piece['type'])
    valid_actions = []
    for rot, x in valid_moves:
        action_id = rot * 10 + x
        if action_id < 40:
            valid_actions.append(action_id)
    if env.can_hold:
        valid_actions.append(40)

    if not valid_actions:
        return [0], [1.0]  # 유효한 액션이 없으면 기본값 반환

    visits = [root_node.children[a].N if a in root_node.children else 0 
             for a in valid_actions]
    sumv = sum(visits)
    if sumv < 1e-9:
        pi = [1.0/len(valid_actions)] * len(valid_actions)
    else:
        pi = [v/sumv for v in visits]

    return valid_actions, pi

############################
# 4) MCTS 학습 루프
############################

def train_mcts(env, model, optimizer, num_episodes=1000, mcts_sims=20, max_depth=2,
               device='cpu', save_interval=100, log_dir='runs/mcts_tetris', non_huristic=False):
    writer = SummaryWriter(log_dir)
    model.to(device)
    
    # non_huristic 설정
    env.non_huristic = non_huristic

    all_scores=[]
    for ep in range(num_episodes):
        state = env.reset()
        done=False
        episode_data=[]
        total_reward=0

        while not done:
            # 루트 노드 생성
            root = MCTSNode(state=state)
            # MCTS 실행
            actions, pi = run_mcts(root, env, model,
                                   simulations=mcts_sims,
                                   max_depth=max_depth,
                                   device=device)
            # actions: list of valid action idx
            # pi: same length list
            # 확률적으로 액션 선택
            a_chosen = np.random.choice(actions, p=pi)
            
            # step
            next_s, r, done, _, _ = env.step(a_chosen)
            total_reward+=r

            # (state, pi_full, reward)
            # -> pi_full: num_actions=40
            pi_full = np.zeros(model.num_actions, dtype=np.float32)
            for act_i, prob_i in zip(actions, pi):
                if act_i<model.num_actions:
                    pi_full[act_i]=prob_i

            episode_data.append((state.copy(), pi_full, 0.0))  # value=추후 final로

            state=next_s

        # 에피소드 끝
        all_scores.append(total_reward)
        
        # final_value = np.tanh(total_reward/100.0)
        # 모든 시점에 대해 동일한 최종 value를 주는 경우
        final_value = max(-1.0, min(1.0, total_reward/1000.0))

        # 학습용 텐서 준비
        states=[]
        target_pis=[]
        target_vs=[]
        for (s, pi_vec, _) in episode_data:
            states.append(s)
            target_pis.append(pi_vec)
            target_vs.append(final_value)
        
        states_t = torch.FloatTensor(states).to(device)
        pis_t = torch.FloatTensor(target_pis).to(device)
        vs_t = torch.FloatTensor(target_vs).unsqueeze(1).to(device)

        # 순전파
        policy_logits, values = model(states_t)
        log_probs = F.log_softmax(policy_logits, dim=1)
        # policy loss (cross-entropy)
        policy_loss = -torch.mean(torch.sum(pis_t * log_probs, dim=1))
        # value loss (mse)
        value_loss = F.mse_loss(values, vs_t)

        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 로깅
        writer.add_scalar('Train/Score', total_reward, ep)
        writer.add_scalar('Train/Loss', loss.item(), ep)
        writer.add_scalar('Train/PolicyLoss', policy_loss.item(), ep)
        writer.add_scalar('Train/ValueLoss', value_loss.item(), ep)

        if (ep+1)%10==0:
            print(f"Episode {ep+1}/{num_episodes} | Score={total_reward:.1f} | Loss={loss.item():.4f}")
        if (ep+1)%save_interval==0:
            print(f"Saving model at episode {ep+1} at {log_dir}/model_ep{ep+1}.pth")
            torch.save(model.state_dict(), f"{log_dir}/model_ep{ep+1}.pth")

    writer.close()
    return all_scores


############################
# 메인 (argparse)
############################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--non_huristic', action='store_true', help='Huristic 제거')
    parser.add_argument('--sims', type=int, default=20)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save-interval', type=int, default=100)
    args = parser.parse_args()

    # 환경 & 모델
    env = TetrisEnv(non_huristic=args.non_huristic)
    model = AlphaZeroPolicyValueNet(obs_dim=214, num_actions=41)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 학습
    train_mcts(env, model, optimizer,
               num_episodes=args.episodes,
               mcts_sims=args.sims,
               max_depth=args.depth,
               device=args.device,
               save_interval=args.save_interval,
               log_dir='runs/mcts_tetris',
               non_huristic=args.non_huristic)

if __name__ == "__main__":
    main()