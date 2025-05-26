from typing import List, Union

from lightning.pytorch.callbacks import Callback

from gestures_forms_sim import measure_sim_gestures


class OnlineEvalCallback(Callback):
    """
    Callback for online evaluation of the model on different tasks during training.

    Args:
        poses: poses from data feeder
        mirrored_poses: mirrored poses from data feeder
        every_n_epochs: list/int of how often the callback should be executed per task
        tasks: names of the tasks, each task is implemented as a class method `_{task}`
    """
    def __init__(
            self,
            poses,
            mirrored_poses,
            audio_dict=None,
            every_n_epochs: Union[int, List[int]] = 5,
            tasks: List[str] = ["retrieval", "similarity"],
            skeleton_backbone: str = "stgcn",
            modalities: List[str] = ["skeleton", "semantic", "speech"]
    ):
        super().__init__()
        self.epoch = 0
        self.modalities = modalities
        self.poses = poses
        self.mirrored_poses = mirrored_poses
        self.skeleton_backbone = skeleton_backbone
        self.audio_dict = audio_dict

        self.tasks = tasks
        self.every_n_epochs = every_n_epochs if isinstance(every_n_epochs, list) else [every_n_epochs] * len(tasks)

    def on_validation_epoch_end(self, trainer, pl_module):
        for i, task in enumerate(self.tasks):
            if self.epoch % self.every_n_epochs[i] == 0:
                print(f"Epoch {self.epoch}. Online evaluation on {task} task.")
                method = getattr(self, f"_{task}", None)
                if callable(method):
                    method(pl_module)
        self.epoch += 1

       
    def _similarity(self, pl_module):
        correlation, difference, _ = measure_sim_gestures(pl_module.model, self.poses, self.mirrored_poses, skeleton_backbone=self.skeleton_backbone, modalities=self.modalities, audio_dict=self.audio_dict)
        self.log('val/correlation', correlation)
        self.log('val/difference', difference)
