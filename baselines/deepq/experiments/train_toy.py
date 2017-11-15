import gym
import argparse
from baselines import deepq
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--exploration_fraction', type=float, default=0.2)
    boolean_flag(parser, 'prioritized_replay', default=False)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--space_size', type=int, default=8)
    boolean_flag(parser, 'hindsight', default=False)
    args = parser.parse_args()
    return args

def main(args):
    env = gym.make("HindsightToy-v0")
    env.set_space_size(args.space_size)
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.mlp([256], layer_norm=False)
    act = deepq.toy_learn(
        env,
        q_func=model,
        lr=args.lr,
        buffer_size=args.buffer_size,
        exploration_fraction=args.exploration_fraction,
        batch_size=args.batch_size,
        gamma=args.gamma,
        prioritized_replay=args.prioritized_replay,
        hindsight=args.hindsight
    )
    print("Saving model to HindsightToy.pkl")
    act.save("HindsightToy.pkl")

if __name__ == '__main__':
    args = parse_args()
    main(args)
