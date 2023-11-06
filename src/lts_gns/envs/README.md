# How To Add a New Environment

## 1. Create a new environment class
Create new folder in `src/envs/` and add a new file `my_env_processor.py` with the following content:

```python
class MyEnvDataLoaderProcessor(AbstractDataLoaderProcessor):

    ###########################################
    ####### Interfaces for data loading #######
    ###########################################

    def _get_rollout_length(self, raw_task: ValueDict) -> int:
        """
        Returns the rollout length of the task. This is the number of timesteps in the rollout.
        We have a -1 here because the initial for loop went to rollout_length -2 here and -1 for all other tasks
        Args:
            raw_task:

        Returns:

        """
        ...

    def _load_raw_data(self, split: str) -> List[ValueDict]:
        ...

    def _select_and_normalize_attributes(self, raw_task: ValueDict) -> ValueDict:
        ...

    def _build_data_dict(self, raw_task: ValueDict, timestep: int) -> ValueDict:
        ...

    def _build_graph(self, data_dict: ValueDict) -> Data:
        ...

    ###########################################
    ####### Functions for the processor #######
    ###########################################

    def get_integrate_predictions_fn(self) -> Callable[[torch.Tensor, Batch], Dict[str, torch.Tensor]]:
        ...

    def get_update_batch_fn(self) -> Callable[[Batch, Dict[str, torch.Tensor]], Batch]:
        ...
```
You can have a look at `sofa_envs` as an example.

## 2. Add your environment to the `gns_environemnt_factory`
Add the name of your environemnt to the `gns_environemnt_factory.py` file in the case statement.

## 3. Add your environment parameters to the default config
Open the config `config.yml` and scroll down to `envs`. Add your environment parameters under a new dictionary with the
name of your environment. You can have a look at the `sofa_envs` as an example.

## 4. (Optional) Compute the statistics for your environment
If you want to use the normalization, you have to compute the mean and standard deviation of the **velocities** of your 
environment. You can take a look at `scripts/compute_env_statistics/main.py` for an example. Add them in the `statistics`
dictionary in your env dictionary of the `config.yml` file.
