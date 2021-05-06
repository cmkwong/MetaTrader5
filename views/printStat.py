
def print_dict(dict):
    print("\n~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")
    for key, value in dict.items():
        print("{}:\t{:.5f}".format(key, value))

def loss_status(writer, loss, episode, mode='train'):
    """
    :param writer: SummaryWriter from pyTorch
    :param loss: float
    :param episode: int
    :param mode: string "train" / "test"
    """
    writer.add_scalar("{}-episode_loss".format(mode), loss, episode)
    print("{}. {} loss: {:.6f}".format(episode, mode, loss))