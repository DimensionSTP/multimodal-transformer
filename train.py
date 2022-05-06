import dotenv
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities.distributed import rank_zero_info

from src.setup import SetUp

dotenv.load_dotenv(override=True)
OmegaConf.register_new_resolver('eval', lambda x: eval(x))

def train(config: DictConfig):
    
    if 'seed' in config:
        seed_everything(config.seed)
            
    setting = SetUp(config)
    
    train_loader = setting.get_train_loader()
    val_loader = setting.get_val_loader()
    
    
    pl_module = setting.get_pl_module()
    callbacks = setting.get_callbacks()
    logger = setting.get_wandb_logger()
            
    trainer: Trainer = instantiate(config.trainer, 
                                   callbacks=callbacks, 
                                   logger=logger,  
                                   _convert_='partial')
        
    trainer.fit(model=pl_module, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)
    
    optimized_metric = config.get('optimized_metric')
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
    
@hydra.main(config_path='configs/', config_name='train.yaml')
def main(config):
    rank_zero_info(OmegaConf.to_yaml(config))
    return train(config)

if __name__ == '__main__':
    main()