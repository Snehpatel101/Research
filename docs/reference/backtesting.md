# Backtesting

`MLFactory` backtests the deployed strategy's out-of-sample signals when
`evaluation.run_backtest` is set, wiring in the label's barriers and cost term.
Use the classes directly to backtest your own signals. The execution-timing
contract (fills at bar *i + 1* by default) is explained in
[Concepts](../concepts.md#backtest-execution-timing).

::: src.inference.backtesting.backtest
    options:
      show_root_heading: false
      members: false

::: src.inference.backtesting.Backtester
    options:
      members: [run]

::: src.inference.backtesting.BacktestConfig
    options:
      members: false

::: src.inference.backtesting.BacktestResult
    options:
      members: [summary, print_summary]

::: src.inference.backtesting.ExecutionModel

::: src.inference.backtesting.TransactionCosts
    options:
      members: false
