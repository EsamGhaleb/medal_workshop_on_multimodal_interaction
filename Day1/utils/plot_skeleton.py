import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from celluloid import Camera

# mpl.use('Agg')

import matplotlib.animation as animation


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
def visualize_only_skeleton(joints, frames, location, result, parts, pause, given_title, file_path, args, filename="fig.png"):
    joints_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9],
                    [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [14, 16], [0, 15], [15, 17]]
    C, T, V, M = location.shape
    # ani =   animation.FuncAnimation(fig, update_plot, frames=range(len(sequence)),
    #                               fargs=(sequence, scat))

    # plt.show()

    from celluloid import Camera

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis("off")
    plt.gca().invert_yaxis()

    camera = Camera(fig)
    for frame in range(0, joints.shape[0]):
        ax.title.set_text(given_title.format(frame))
        x = location[0, frame, :, 0]
        y = location[1, frame, :, 0]
    
        c = []
        for v in range(V):
            r = result[frame // 4, v, 0]
            g = 0
            b = 1 - r
            c.append([r, g, b])
            # ntu starts with 1
        c = np.array(c)
        for part in parts:
            c_part = np.max(c[part], 0)
            c[part] = c_part
        ax.scatter(x, y, marker='o', c=c, s=36)
        # ax.scatter(joints[frame, :, 0], joints[frame, :, 1])
        if frame > len(frames)-1:
            break
        # print(frame)
        # print(len(frames))
        # ax.imshow(frames[frame])
        for edge in joints_edges:
            edges_size = np.mean(c[edge])
            ax.plot((joints[frame, edge[0], 0], joints[frame, edge[1], 0]),
                    (joints[frame, edge[0], 1], joints[frame, edge[1], 1]),
                    '-o', c=np.array(np.mean(c[edge], 0)), linewidth=3 * edges_size, markersize=2)
        # plt.show()
        # plt.close()
        camera.snap()

    animation = camera.animate(interval=200, blit=True)
    # plt.show()
    # plt.close()
    animation.save('results/{}/skeleton_'.format(args.branch)+file_path+'.mp4', dpi=100,
                   savefig_kwargs={
        'frameon': False,
        'pad_inches': 0
    })
    plt.close()
    return animation


def visualize_skeleton(joints, frames, location, result, parts, pause, given_title, file_path, args, filename="fig.png"):
    joints_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9],
                    [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [14, 16], [0, 15], [15, 17]]
    C, T, V, M = location.shape
    # ani =   animation.FuncAnimation(fig, update_plot, frames=range(len(sequence)),
    #                               fargs=(sequence, scat))

    # plt.show()

    from celluloid import Camera

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis("off")
    plt.gca().invert_yaxis()

    camera = Camera(fig)
    for frame in range(0, joints.shape[0]):
        ax.title.set_text(given_title.format(frame))
        x = location[0, frame, :, 0]
        y = location[1, frame, :, 0]
    
        c = []
        for v in range(V):
            r = result[frame // 4, v, 0]
            g = 0
            b = 1 - r
            c.append([r, g, b])
            # ntu starts with 1
        c = np.array(c)
        for part in parts:
            c_part = np.max(c[part], 0)
            c[part] = c_part
        ax.scatter(x, y, marker='o', c=c, s=36)
        # ax.scatter(joints[frame, :, 0], joints[frame, :, 1])
        if frame > len(frames)-1:
            break
        # print(frame)
        # print(len(frames))
        ax.imshow(frames[frame])
        for edge in joints_edges:
            edges_size = np.mean(c[edge])
            ax.plot((joints[frame, edge[0], 0], joints[frame, edge[1], 0]),
                    (joints[frame, edge[0], 1], joints[frame, edge[1], 1]),
                    '-o', c=np.array(np.mean(c[edge], 0)), linewidth=3 * edges_size, markersize=2)
        # plt.show()
        # plt.close()
        camera.snap()

    animation = camera.animate(interval=200, blit=True)
    # plt.show()
    # plt.close()
    animation.save('results/{}/'.format(args.branch)+file_path+'.mp4', dpi=100,
                   savefig_kwargs={
        'frameon': False,
        'pad_inches': 0
    })
    plt.close()
    return animation


def visualize_skeleton_openpose_18(joints, frames, file_path, filename="fig.png"):
    joints_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9],
                    [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [14, 16], [0, 15], [15, 17]]
    joints = joints.transpose(1, 2, 0)
    joints[joints[:, :, 2] < 0.1] = np.nan
    joints[np.isnan(joints[:, :, 2])] = np.nan

    # ani =   animation.FuncAnimation(fig, update_plot, frames=range(len(sequence)),
    #                               fargs=(sequence, scat))

    # plt.show()

    # from celluloid import Camera

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.gca().invert_yaxis()

    camera = Camera(fig)
    for frame in range(0, joints.shape[0]):

        ax.scatter(joints[frame, :, 0], joints[frame, :, 1])
        if frame >= len(frames):
            break
        ax.imshow(frames[frame])
        for edge in joints_edges:
            ax.plot((joints[frame, edge[0], 0], joints[frame, edge[1], 0]),
                    (joints[frame, edge[0], 1], joints[frame, edge[1], 1]))
        # plt.show()
        # plt.close()
        camera.snap()

    animation = camera.animate(interval=30)
    # plt.show()
    plt.close()
    animation.save('figs/'+file_path+'.gif')

    return animation

def make_barplot(y, c, label):
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.02f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    x = np.arange(len(c))  # the label locations

    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x, y, width, label=label)

    autolabel(rects1)

    plt.xticks(rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(c)
    plt.tight_layout()
    plt.close()

    return fig


import cv2

features_blobs = None


def hook_feature(module, input, output):
    global features_blobs
    features_blobs = np.squeeze(output.data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def setup_cam(model):
    features_names = ['layer4']  # this is the last conv layer of the resnet
    for name in features_names:
        model.module.base_model._modules.get(name).register_forward_hook(hook_feature)