from losses.losses import get_loss, get_loss_nik, get_loss_reparametrization_trick
from models.simple_unet import ApproximateScore


def loss_model_bel(mode):
    loss_function = lambda *args: get_loss_nik(*args, mode=mode)
    nn_model = ApproximateScore()
    name = f"BEL {mode}"
    return loss_function, nn_model, name


def loss_model_gaussian():
    loss_function = get_loss
    nn_model = ApproximateScore()
    name = "Ours"
    return loss_function, nn_model, name


def loss_model_reparam():
    loss_function = get_loss_reparametrization_trick
    nn_model = ApproximateScore()
    name = "Reparametrization Trick"
    return loss_function, nn_model, name


def loss_model_no_train():
    loss = lambda *args: 0.0
    nn_model = ApproximateScore()
    name = "Untrained"
    return loss, nn_model, name
