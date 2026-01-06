import math
import torch
import torch.nn as nn
import simulator

from . import param as p
from .data_loader import DatasetLoader

from utils.plot import Plot
from utils.paths import RESULTS_LOG_PATH
from simulator._cpu_ import policy

class CasualTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.rtg_embedding = nn.Linear(1, p.D_MODEL)
        self.state_embedding = nn.Linear(p.STATE_DIM, p.D_MODEL)
        self.action_embedding = nn.Embedding(p.NUM_ACTIONS, p.D_MODEL)

        self.action_positional_encoding = nn.Embedding(p.NUM_ACTIONS, p.D_MODEL)
        self.timestep_positional_encoding = nn.Embedding(p.MAX_TIMESTEPS, p.D_MODEL)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=p.D_MODEL,
            nhead=p.N_HEADS,
            dim_feedforward=4*p.D_MODEL,
            activation='relu',
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=p.N_LAYERS
        )

        self.ln = nn.LayerNorm(p.D_MODEL)
        self.action_head = nn.Linear(p.D_MODEL, p.NUM_ACTIONS)
        pass

    def forward(self, rtgs, states, actions, timesteps):
        """
        rtgs      : (B, K, 1)
        states    : (B, K, state_dim)
        actions   : (B, K)
        timesteps : (B, K)
        """

        device = rtgs.device
        rtg_tok = self.rtg_embedding(rtgs)                 # (B, K, d)
        state_tok = self.state_embedding(states)           # (B, K, d)
        # Clamp actions to valid embedding index range to avoid IndexError
        actions = actions.clamp(min=0, max=p.NUM_ACTIONS - 1)
        action_tok = self.action_embedding(actions)        # (B, K, d)

        tokens = torch.stack(
            (rtg_tok, state_tok, action_tok),
            dim=2
        ).reshape(p.BATCH_SIZE, 3 * p.SEQ_LEN, -1)                # (B, 3K, d)

        time_tok = self.timestep_positional_encoding(timesteps)    # (B, K, d)
        time_tok = time_tok.repeat_interleave(3, dim=1)
        tokens = tokens + time_tok

        mask = torch.triu(torch.ones(3 * p.SEQ_LEN, 3 * p.SEQ_LEN, device=device), diagonal=1).bool()
        h = self.transformer(tokens, mask=mask)        # (B, 3K, d)

        state_positions = torch.arange(1, 3 * p.SEQ_LEN, 3, device=device)
        h_state = h[:, state_positions]                 # (B, K, d)
        h_state = self.ln(h_state)

        logits = self.action_head(h_state)              # (B, K, num_actions)
        return logits
    
def train(_model_, path, count) -> CasualTransformer:
    print(f'Initiated training...')
    model, optimizer = _model_['model'], _model_['optimizer']
    data_loader = DatasetLoader(path, count)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    total_steps = getattr(p, "TRAIN_STEPS", 1000)
    with open(RESULTS_LOG_PATH, 'w') as log_file:
        log_file.write('Training Log:\n')
    for step in range(total_steps):
        batch = data_loader.sample_batch()

        rtgs = torch.from_numpy(batch["rtgs"]).float().unsqueeze(-1).to(device)        # (B, K, 1)
        states = torch.from_numpy(batch["states"]).float().to(device)                 # (B, K, state_dim)
        actions = torch.from_numpy(batch["actions"]).long().to(device).clamp(0, p.NUM_ACTIONS - 1)  # (B, K)
        timesteps = torch.arange(p.SEQ_LEN, device=device).unsqueeze(0).repeat(p.BATCH_SIZE, 1)  # (B, K)

        # Gradient accumulation: accumulate over ACCUM_STEPS before updating
        accum_steps = getattr(p, "ACCUM_STEPS", 1)
        if (step % accum_steps) == 0:
            optimizer.zero_grad(set_to_none=True)

        logits = model(rtgs, states, actions, timesteps)       # (B, K, num_actions)
        loss = criterion(logits.view(-1, p.NUM_ACTIONS), actions.view(-1))
        (loss / accum_steps).backward()

        if ((step + 1) % accum_steps == 0) or (step + 1 == total_steps):
            torch.nn.utils.clip_grad_norm_(model.parameters(), p.GRAD_CLIP)
            optimizer.step()

        with open(RESULTS_LOG_PATH, 'a') as log_file:
            # Log unscaled loss for comparability
            log_file.write(f"step {step+1}/{total_steps} loss={loss.item():.4f}\n")
        if (step + 1) % 10 == 0:
            print(f"step {step+1}/{total_steps} loss={loss.item():.4f}")
    print(f'Training completed.')
    return {'model': model, 'optimizer': optimizer}

def test(_model_, path, count) -> float:
    print(f'Initiated testing...')
    model, _ = _model_['model'], _model_['optimizer']
    data_loader = DatasetLoader(path, count)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_loss = 0.0
    total_batches = 0
    
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            try:
                batch = data_loader.sample_batch()
                rtgs = torch.from_numpy(batch["rtgs"]).float().unsqueeze(-1).to(device)        # (B, K, 1)
                states = torch.from_numpy(batch["states"]).float().to(device)                 # (B, K, state_dim)
                actions = torch.from_numpy(batch["actions"]).long().to(device).clamp(0, p.NUM_ACTIONS - 1)  # (B, K)
                timesteps = torch.arange(p.SEQ_LEN, device=device).unsqueeze(0).repeat(p.BATCH_SIZE, 1)  # (B, K)

                logits = model(rtgs, states, actions, timesteps)                       # (B, K, num_actions)
                loss = criterion(logits.view(-1, p.NUM_ACTIONS), actions.view(-1))
                
                total_loss += loss.item()
                total_batches += 1
            except ValueError:
                break
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    print(f'Testing completed.')
    print(f'Average Loss: {avg_loss:.4f}')
    with open(RESULTS_LOG_PATH, 'a') as log_file:
        log_file.write(f'Average Test Loss: {avg_loss:.4f}\n')
    return avg_loss

def run(_model_):
    plot = Plot()
    print(f'Initiated run...')
    model = _model_['model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()

    env_transformer = simulator.make_env(True)
    env_fcfs = simulator.make_env(False)
    env_sjf = simulator.make_env(True)
    env_rr = simulator.make_env(True)
    env_srtf = simulator.make_env(True)

    env_transformer.reset(seed=42)
    env_fcfs.reset(seed=42)
    env_sjf.reset(seed=42)
    env_rr.reset(seed=42)
    env_srtf.reset(seed=42)

    K = p.SEQ_LEN
    B = p.BATCH_SIZE

    states_buf = torch.zeros((B, K, p.STATE_DIM), dtype=torch.float32, device=device)
    actions_buf = torch.zeros((B, K), dtype=torch.long, device=device)
    rtgs_buf = torch.zeros((B, K, 1), dtype=torch.float32, device=device)
    timesteps = torch.arange(K, device=device).unsqueeze(0).repeat(B, 1)

    t = 0
    rtg_curr = 10000.0
    done, done_1, done_2, done_3, done_4, done_5 = False, False, False, False, False, False

    print(f'Timesteps for transformer:{env_transformer.episode_length}')
    print(f'Timesteps for FCFS:{env_fcfs.episode_length}')
    print(f'Timesteps for SJF:{env_sjf.episode_length}')
    print(f'Timesteps for RR:{env_rr.episode_length}')
    print(f'Timesteps for SRTF:{env_srtf.episode_length}')

    with open(RESULTS_LOG_PATH, 'a') as log_file:
        log_file.write('Run Log:\n')
        log_file.write(f'Timesteps for transformer:{env_transformer.episode_length}\n')
        log_file.write(f'Timesteps for FCFS:{env_fcfs.episode_length}\n')
        log_file.write(f'Timesteps for SJF:{env_sjf.episode_length}\n')
        log_file.write(f'Timesteps for RR:{env_rr.episode_length}\n')
        log_file.write(f'Timesteps for SRTF:{env_srtf.episode_length}\n')
    while not done:
        if done_1 is False:
            state = env_transformer.get_state()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)

            if t >= K:
                states_buf[:, :-1] = states_buf[:, 1:]
                actions_buf[:, :-1] = actions_buf[:, 1:]
                rtgs_buf[:, :-1] = rtgs_buf[:, 1:]
                idx = K - 1
            else:
                idx = t

            states_buf[0, idx, :] = state_tensor
            rtgs_buf[0, idx, 0] = rtg_curr

            with torch.no_grad():
                logits = model(rtgs_buf, states_buf, actions_buf, timesteps)
                step_logits = logits[0, idx, :]

                # Sanitize logits to avoid NaN/Inf and improve numerical stability
                step_logits = torch.nan_to_num(step_logits, nan=0.0, posinf=0.0, neginf=0.0)
                step_logits = step_logits - step_logits.max()  # softmax stability

                probs = torch.softmax(step_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0)

                # Fallback to uniform distribution if probs become invalid
                probs_sum = probs.sum()
                if (probs_sum <= 0) or (not torch.isfinite(probs_sum)):
                    probs = torch.full_like(probs, 1.0 / p.NUM_ACTIONS)

                action = torch.distributions.Categorical(probs=probs).sample().item()

            _, reward, done_1 = env_transformer.step(action)
            if done_1:
                print(f'Episode for Transformer finished after {t+1} timesteps.')
                with open(RESULTS_LOG_PATH, 'a') as log_file:
                    log_file.write(f'Episode for Transformer finished after {t+1} timesteps.\n')
            actions_buf[0, idx] = int(action)
            rtg_curr = rtg_curr - float(reward)
        
        if done_2 is False:
            action_fcfs = policy.act(env_fcfs, 'FCFS')
            _, reward_fcfs, done_2 = env_fcfs.step(action_fcfs)
            if done_2:
                print(f'Episode for FCFS finished after {t+1} timesteps.')
                with open(RESULTS_LOG_PATH, 'a') as log_file:
                    log_file.write(f'Episode for FCFS finished after {t+1} timesteps.\n')

        if done_3 is False:
            action_sjf = policy.act(env_sjf, 'SJF')
            _, reward_sjf, done_3 = env_sjf.step(action_sjf)
            if done_3:
                print(f'Episode for SJF finished after {t+1} timesteps.')
                with open(RESULTS_LOG_PATH, 'a') as log_file:
                    log_file.write(f'Episode for SJF finished after {t+1} timesteps.\n')

        if done_4 is False:
            action_rr = policy.act(env_rr, 'RR')
            _, reward_rr, done_4 = env_rr.step(action_rr)
            if done_4:
                print(f'Episode for RR finished after {t+1} timesteps.')
                with open(RESULTS_LOG_PATH, 'a') as log_file:
                    log_file.write(f'Episode for RR finished after {t+1} timesteps.\n')

        if done_5 is False:
            action_srtf = policy.act(env_srtf, 'SRTF')
            _, reward_srtf, done_5 = env_srtf.step(action_srtf)
            if done_5:
                print(f'Episode for SRTF finished after {t+1} timesteps.')
                with open(RESULTS_LOG_PATH, 'a') as log_file:
                    log_file.write(f'Episode for SRTF finished after {t+1} timesteps.\n')
        
        plot.log_rewards(reward, reward_fcfs, reward_sjf, reward_rr, reward_srtf)
        done = done_1 and done_2 and done_3 and done_4 and done_5
        if done:
            print(f'All episodes finished after {t+1} timesteps.')
            with open(RESULTS_LOG_PATH, 'a') as log_file:
                log_file.write(f'All episodes finished after {t+1} timesteps.\n')

        if t % 10 == 0:
            with open(RESULTS_LOG_PATH, 'a') as log_file:
                log_file.write(f'Timestep {t}: Transformer: {reward}, FCFS: {reward_fcfs}, SJF: {reward_sjf}, RR: {reward_rr}, SRTF: {reward_srtf}\n')
        t += 1

        if t % 50 == 0:
            print(f'{t} timesteps completed.')
    print(f'Run completed.')
    plot.plot_rewards()
    pass

def get_instance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CasualTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p.LEARNING_RATE, weight_decay=getattr(p, 'WEIGHT_DECAY', 0.0))
    return {'model': model, 'optimizer': optimizer}

