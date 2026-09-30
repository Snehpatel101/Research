# API reference

The public surface, generated from the docstrings. Everything else under `src/`
is internal and may change without notice.

| You want to | Use | Page |
|---|---|---|
| Configure a run | `ExperimentConfig` and its sections | [ExperimentConfig](config.md) (field tables: [Configuration](../configuration.md)) |
| Train, backtest, bundle and deploy | `MLFactory(cfg).run()` → `ExperimentResult` | [MLFactory](factory.md) |
| Serve a deployed run | `load_deploy_artifact`, `load_bundle` | [Deploy and bundles](inference.md) |
| Serve several bundles together | `UniversalInferencePipeline` | [UniversalInferencePipeline](universal-pipeline.md) |
| Backtest your own signals | `Backtester`, `BacktestConfig` | [Backtesting](backtesting.md) |

```python
from src.config.experiment import ExperimentConfig
from src.factory import MLFactory, ExperimentResult
from src.inference import (
    load_deploy_artifact, load_bundle, validate_deploy_artifact,
    ModelBundle, EnsembleBundle, RegimeBundle, MetaLabelingBundle,
    UniversalInferencePipeline,
)
from src.inference.backtesting import Backtester, BacktestConfig
```
