import numpy as np
import pidnet
import torch
import torch.nn.functional as F

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image


def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location="cpu")
    if "state_dict" in pretrained_dict:
        pretrained_dict = pretrained_dict["state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {
        k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)
    }
    msg = "Loaded {} parameters!".format(len(pretrained_dict))
    print("Attention!!!")
    print(msg)
    print("Over!!!")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    return model


name = "pidnet-l"
num_classes = 5
pretrained = "../pretrained_model/pidnet/pidnet_large_boatsim.pt"

model = pidnet.get_pred_model(name, num_classes)
# model = load_pretrained(model, pretrained).cuda()
model = load_pretrained(model, pretrained).cpu()
model.eval()


def get_semantic_label(image):
    image = input_transform(image)
    image = image.transpose((2, 0, 1)).copy()
    # image = torch.from_numpy(image).unsqueeze(0).cuda()
    image = torch.from_numpy(image).unsqueeze(0).cpu()
    pred = model(image)
    pred = F.interpolate(pred, size=image.size()[-2:], mode="bilinear", align_corners=True)
    pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    return pred
