import pathlib
import numpy as np

def extract_first_episode_per_env(data_dir: str = "./logs-expert/data"):
    dir_path = pathlib.Path(data_dir)
    npz_files = list(dir_path.glob("*.npz"))
    
    if not npz_files:
        print(f"No .npz files found in {data_dir}.")
        return

    print("Extracting the FIRST episode of each 128 parallel environments (Optimized).")

    for file_path in npz_files:
        print(f"\nLoading {file_path.name} into memory (This takes ~30 seconds)...")
        # Load fully into memory dict to avoid repetitive zip decompression overhead
        with np.load(file_path) as npz:
            data = {k: npz[k] for k in npz.files}
            
        dones = data['done']
        num_steps, num_envs = dones.shape
        
        episodes_obs = []
        episodes_action = []
        episodes_solved = []
        episodes_length = []
        episodes_return = []
        
        print("Slicing 128 episodes...")
        for env_idx in range(num_envs):
            done_indices = np.where(dones[:, env_idx])[0]
            if len(done_indices) > 0:
                end_idx = done_indices[0]
                episodes_obs.append(data['obs'][0:end_idx+1, env_idx])
                episodes_action.append(data['action'][0:end_idx+1, env_idx])
                episodes_solved.append(data['solved'][end_idx, env_idx])
                episodes_return.append(data['return_'][end_idx, env_idx])
                episodes_length.append(end_idx + 1)
                
        if len(episodes_length) == 0:
            print(f"Skipping {file_path.name}: No completed episodes found.")
            continue
            
        print("Padding and saving...")
        max_len = max(episodes_length)
        padded_obs = np.zeros((len(episodes_length), max_len, *data['obs'].shape[2:]), dtype=data['obs'].dtype)
        padded_act = np.zeros((len(episodes_length), max_len, *data['action'].shape[2:]), dtype=data['action'].dtype)
        
        for i in range(len(episodes_length)):
            seq_len = episodes_length[i]
            padded_obs[i, :seq_len] = episodes_obs[i]
            padded_act[i, :seq_len] = episodes_action[i]
            
        sliced_data = {
            'obs': padded_obs,
            'action': padded_act,
            'solved': np.array(episodes_solved),
            'length': np.array(episodes_length),
            'return_': np.array(episodes_return)
        }
        
        # Save to a temporary file, then overwrite, to ensure safety
        tmp_file = file_path.with_suffix('.tmp.npz')
        np.savez(tmp_file, **sliced_data)
        tmp_file.replace(file_path)
        
        print(f"  -> Saved {file_path.name} ({len(episodes_length)} perfectly matched episodes).")

if __name__ == "__main__":
    extract_first_episode_per_env()
