import argparse
import gym
from gym.utils.save_video import save_video
import numpy as np
import torch
import time
from lib.model import ActorCritic


ENV_ID  = "Humanoid-v4"
HIDDEN_SIZE = 64

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-d", "--deterministic", default=False, action="store_true", help="enable deterministic actions")
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    # Autodetect CUDA
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    env = gym.make(args.env, render_mode="rgb_array_list")
    print("OKAY")
    num_inputs  = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    model = ActorCritic(num_inputs, num_outputs, HIDDEN_SIZE).to(device)
    #model.load_state_dict(torch.load(args.model))
    model = torch.load(args.model)

    step_starting_index = 0

    episode_index = 0


    #Training: sampling actions semi-randomly from the prob dist output by the network, so we get exploration
    # Testing: deterministic not random
    for i in range(20): #numner of videos
        state = env.reset()[0]
        done = False
        total_steps = 0
        total_reward = 0
        while total_steps<=256:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, _ = model(state)
            #continous action space instead of sampling based on the mean and stdf, we use means
            #deterministic action space we would take the arg max of probaliblies
            action = dist.mean.detach().cpu().numpy()[0] if args.deterministic \
                else dist.sample().cpu().numpy()[0]
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward
            total_steps += 1
        save_video(
           env.render(),
           "videos",
           fps=env.metadata["render_fps"],
           step_starting_index=step_starting_index,
           episode_index=episode_index
        )
        step_starting_index = step_starting_index + 1
        episode_index += 1
        env.reset()

        print("In %d steps we got %.3f reward" % (total_steps, total_reward))
    env.close()
