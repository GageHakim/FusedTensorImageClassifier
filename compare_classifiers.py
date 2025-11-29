import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from evaluate_baseline_classifier import evaluate_baseline
from evaluate_improved_baseline_classifier import evaluate_improved_baseline
from evaluate_fused_classifier import evaluate_fused
from evaluate_mean_fused_classifier import evaluate_mean_fused


def run_evaluation(console, title, eval_function):
    """Runs an evaluation function and handles exceptions."""
    console.print(f"[bold blue]Step: Evaluating {title}...[/bold blue]")
    try:
        metrics = eval_function()
        console.print(f"[green]{title} evaluation complete.[/green]\n")
        return metrics
    except Exception as e:
        console.print(f"[bold red]Error during {title} evaluation: {e}[/bold red]\n")
        return None


def create_comparison_table(title, column_name, metrics_data, key, format_spec):
    """Creates a Rich table for a specific metric."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Classifier", style="dim", width=25)
    table.add_column(column_name, justify="right")

    for name, metrics in metrics_data:
        if metrics and key in metrics:
            value = format(metrics[key], format_spec)
            table.add_row(name, value)
        else:
            table.add_row(name, "[red]N/A[/red]")
    return table


def create_detailed_table(title, metrics_data, keys_and_labels):
    """Creates a Rich table for detailed metrics like timing or FPS."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=35)
    for name, _ in metrics_data:
        table.add_column(name, justify="right")

    for key, label in keys_and_labels:
        row_data = [label]
        for _, metrics in metrics_data:
            if metrics and key in metrics:
                # Assuming all these metrics are float, format them
                value = f"{metrics[key]:.2f}"
                row_data.append(value)
            else:
                row_data.append("[dim]N/A[/dim]")
        table.add_row(*row_data)

    return table


def warmup_device(device, console):
    """Performs a warm-up routine to stabilize device clock speeds."""
    console.print("[bold yellow]Warming up device for stable benchmarking...[/bold yellow]")
    # Create a dummy tensor on the target device
    dummy_input = torch.randn(10, 10, device=device)
    
    # Perform a series of operations to "heat up" the device
    for _ in range(100):
        dummy_input = dummy_input @ dummy_input
        
    # Synchronize to ensure operations are complete
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
        
    console.print("[green]Warm-up complete.[/green]\n")


def main():
    """
    Runs evaluations for all classifiers and prints a comparative summary.
    """
    console = Console()
    console.print("[bold yellow]Starting Classifier Comparison Engine...[/bold yellow]")

    # --- Hardware Info ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS (Apple Silicon)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    console.print(f"[bold]Running on device: {device_name}[/bold]\n")

    # --- Warm-up ---
    warmup_device(device, console)

    # --- Classifier Definitions ---
    classifiers_to_evaluate = [
        ("Baseline Classifier", evaluate_baseline),
        ("Improved Baseline", evaluate_improved_baseline),
        ("Fused Classifier", evaluate_fused),
        ("Mean Fused Classifier", evaluate_mean_fused),
    ]

    # --- Run Evaluations ---
    all_metrics = []
    for name, func in classifiers_to_evaluate:
        metrics = run_evaluation(console, name, func)
        all_metrics.append((name, metrics))

    # --- Display Explanations ---
    console.print(Panel(
        "[bold]Baseline Classifier:[/bold] A simple CNN trained from scratch on the target dataset.\n"
        "[bold]Improved Baseline:[/bold] Uses a powerful, pre-trained [i]Minnen2018-Mean[/i] model as a frozen feature extractor. A small custom ResNet then classifies these features. This tests the quality of the pre-trained features directly.\n"
        "[bold]Fused Classifier:[/bold] A more complex model that fuses the main latent tensor with a 'texture' signal (the bit-rate) from the compression model before classification.\n"
        "[bold]Mean Fused Classifier:[/bold] Fuses the main latent tensor with the 'uncertainty' signal (scales from the entropy model) before classification.",
        title="[yellow]Model Descriptions[/yellow]",
        expand=False
    ))


    # --- Display Results ---
    console.print("\n[bold yellow]Comparison Summary:[/bold yellow]")

    # 1. Accuracy Table
    accuracy_table = create_comparison_table(
        "Overall Accuracy", "Accuracy (%)", all_metrics, 'accuracy', '.2f'
    )
    console.print(accuracy_table)

    # 2. Inference Speed Table
    timing_keys = [
        ('avg_extraction_time_ms', "Feature Extraction / Fusion (ms/image)"),
        ('avg_classification_time_ms', "Classification (ms/image)"),
        ('avg_total_time_ms', "[bold]Total Inference (ms/image)[/bold]"),
    ]
    timing_table = create_detailed_table(
        "Inference Speed", all_metrics, timing_keys
    )
    console.print(timing_table)

    # 3. Throughput Table
    fps_keys = [
        ('fps_extraction', "Feature Extraction / Fusion (FPS)"),
        ('fps_classification', "Classification (FPS)"),
        ('fps_total', "[bold]Total Throughput (FPS)[/bold]"),
    ]
    fps_table = create_detailed_table(
        "Throughput", all_metrics, fps_keys
    )
    console.print(fps_table)

    console.print("\n[bold]Note:[/bold] Rerunning may yield slightly different performance results due to system load.")
    console.print("Confusion matrices have been saved as PNG files in the root directory.")


if __name__ == '__main__':
    main()
