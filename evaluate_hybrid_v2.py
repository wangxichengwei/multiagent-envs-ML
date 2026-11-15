"""Standalone deterministic testing without starting training."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import time
from contextlib import contextmanager
import torch
import train_hybrid_v2 as experiment
from hybrid_v2 import PPOV2 as HybridPPO
from train_hybrid_v2 import OUT, make_env, scenarios, evaluate, write_json

@contextmanager
def visible_evaluation(env, fps=10., episode_pause=.6, caption='Controller evaluation'):
    """Restore normal flips for manual viewing without altering batch experiments."""
    original_state=experiment.image_state
    original_reset=env.reset
    next_frame=None
    episode=0

    def wait_until(deadline):
        while time.monotonic()<deadline:
            for viewer in env.viewers:
                if viewer is not None:
                    if getattr(viewer.window,'has_exit',False): raise KeyboardInterrupt
                    viewer.window.dispatch_events()
            time.sleep(min(.02,max(0.,deadline-time.monotonic())))

    def reset(*args,**kwargs):
        nonlocal episode,next_frame
        if episode: wait_until(time.monotonic()+episode_pause)
        episode+=1
        next_frame=None
        return original_reset(*args,**kwargs)

    def image_state(current_env,tracker,action=None):
        nonlocal next_frame
        if next_frame is not None: wait_until(next_frame)
        for viewer in current_env.viewers:
            if viewer is not None:
                if getattr(viewer.window,'has_exit',False): raise KeyboardInterrupt
                viewer.window.switch_to()
        # Unlike the batch helper, this never replaces window.flip with a no-op.
        result=experiment.train.get_state(current_env,tracker,action)
        for viewer in current_env.viewers:
            if viewer is not None:
                viewer.window.set_vsync(False)
                viewer.window.set_caption(f'{caption} | Episode {episode} | {fps:g} FPS')
        next_frame=time.monotonic()+1/fps
        return result

    env.reset=reset
    experiment.image_state=image_state
    try:
        yield
    finally:
        experiment.image_state=original_state
        env.reset=original_reset

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['base','hybrid','reference'],default='base')
    parser.add_argument('--checkpoint',type=Path,default=OUT/'best.pt')
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--seed',type=int,default=1014)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fast',action='store_true',help='Batch mode without visible animation')
    parser.add_argument('--fps',type=float,default=10.,help='Visible frames per second (default: 10)')
    args=parser.parse_args()
    if args.episodes<1: parser.error('--episodes must be positive')
    if not 0<args.fps<=120: parser.error('--fps must be in (0,120]')
    torch.set_num_threads(1)
    agent=None;model_hash=None
    if args.method=='hybrid':
        agent=HybridPPO();agent.load(args.checkpoint)
        model_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    output=args.output or OUT/'manual_tests'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    cases=scenarios(args.seed,args.episodes)
    write_json(output/'scenarios.json',dict(seed=args.seed,scenarios=cases))
    write_json(output/'config.json',dict(method=args.method,checkpoint=str(args.checkpoint) if agent else None,
        checkpoint_sha256=model_hash,episodes=args.episodes,seed=args.seed,
        visible=not args.fast,fps=None if args.fast else args.fps))
    env=make_env()
    try:
        if args.fast:
            evaluate(env,cases,output/'episodes.jsonl',agent,reference=args.method=='reference')
        else:
            print(f'Visible evaluation: {args.episodes} episodes, {args.fps:g} FPS',flush=True)
            with visible_evaluation(env,args.fps,caption=f'{args.method} evaluation'):
                evaluate(env,cases,output/'episodes.jsonl',agent,reference=args.method=='reference')
    except KeyboardInterrupt:
        print('Evaluation interrupted; completed episodes remain in',output,flush=True)
        return
    finally:
        for viewer in env.viewers:
            if viewer is not None and not getattr(viewer.window,'has_exit',False):
                viewer.close()
    print('Results:',output,flush=True)

if __name__=='__main__':main()
