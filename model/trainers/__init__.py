from model.trainers.GTA.GTA_mix_trainer import GTAMixTrainer
from model.trainers.COCO2Wiki.COCO2Wiki_trainer import COCO2WikiTrainer
from model.trainers.Dehaze.Dehaze_trainer import DehazeTrainer
from model.trainers.Dehaze.Dehaze_pair_trainer import DehazeTrainer_pair
from model.trainers.Density_estimation.De_trainer import DeTrainer
from model.trainers.Density_estimation.Glow_trainer import GlowTrainer
from model.trainers.s_resolution.sr_trainer import SR_Trainer

__all__ = ['GTAMixTrainer','COCO2WikiTrainer','DehazeTrainer','DehazeTrainer_pair','DeTrainer','GlowTrainer','SR_Trainer']