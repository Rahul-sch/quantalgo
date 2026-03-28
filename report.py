"""
Results Reporter - Generates JSON + Markdown reports.
"""
import os
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_results(all_results: dict, best_combos: list):
    """Save full results to JSON and summary to Markdown."""
    ensure_results_dir()

    # JSON (strip equity curves and trade lists for size)
    json_data = {}
    for key, res in all_results.items():
        json_data[key] = {k: v for k, v in res.items() if k not in ("equity_curve", "trades")}
        json_data[key]["trade_count"] = len(res.get("trades", []))

    json_path = os.path.join(RESULTS_DIR, "backtest_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    # Markdown
    md_path = os.path.join(RESULTS_DIR, "backtest_report.md")
    with open(md_path, "w") as f:
        f.write("# Goldbach/ICT Backtesting Report\n\n")
        f.write("> ⚠️ Paper trading only. Past performance ≠ future results.\n\n")

        # Best combos
        f.write("## 🏆 Top Strategy+Pair Combos\n\n")
        f.write("| Rank | Strategy | Pair | Win Rate | Total Return | Profit Factor | Sharpe | Trades |\n")
        f.write("|------|----------|------|----------|-------------|---------------|--------|--------|\n")
        for i, combo in enumerate(best_combos[:10], 1):
            f.write(f"| {i} | {combo['strategy']} | {combo['symbol']} | "
                    f"{combo['win_rate']}% | {combo['total_return_pct']}% | "
                    f"{combo['profit_factor']} | {combo['sharpe_ratio']} | "
                    f"{combo['total_trades']} |\n")

        f.write("\n## 📊 All Results\n\n")
        for key, res in sorted(all_results.items()):
            if res["total_trades"] == 0:
                continue
            f.write(f"### {key}\n")
            f.write(f"- Trades: {res['total_trades']} | Wins: {res['wins']} | Losses: {res['losses']}\n")
            f.write(f"- Win Rate: {res['win_rate']}%\n")
            f.write(f"- Total P&L: ${res['total_pnl']} ({res['total_return_pct']}%)\n")
            f.write(f"- Avg Win: ${res['avg_win']} | Avg Loss: ${res['avg_loss']}\n")
            f.write(f"- Profit Factor: {res['profit_factor']}\n")
            f.write(f"- Max Drawdown: {res['max_drawdown_pct']}%\n")
            f.write(f"- Sharpe Ratio: {res['sharpe_ratio']}\n\n")

    print(f"\nResults saved to:\n  {json_path}\n  {md_path}")
    return json_path, md_path
