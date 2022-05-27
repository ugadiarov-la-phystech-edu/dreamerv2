from dreamerv2.grafter.grafter.utils.human_player.play_wrapper import PlayWrapper
from dreamerv2.grafter.grafter.wrapper import GrafterWrapper

if __name__ == "__main__":

    env = GrafterWrapper(30, 30)
    env = PlayWrapper(env, seed=100)
    env.play(fps=3)
