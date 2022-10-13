from model.MSPRL import MSPRL


def build_net(model_name, image_channel):
    if model_name == "MSPRL":
        return MSPRL(image_channel)
