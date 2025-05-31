from metrics.collect_metrics import load_metrics
from normalization.boxcox_transform import normalize_data
from modeling.nonlinear_regression import build_models
from evaluation.quality_check import evaluate_quality

if __name__ == "__main__":
    raw_data = load_metrics("data/projects_metrics.csv")
    normalized_data, lambdas = normalize_data(raw_data)
    models = build_models(normalized_data)
    results = evaluate_quality(raw_data, normalized_data, models, lambdas)

    for res in results:
        print(f"App: {res['name']}, Quality: {res['quality']}")