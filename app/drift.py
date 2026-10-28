import numpy as np


MODERATE_PSI_THRESHOLD = 0.10
SIGNIFICANT_PSI_THRESHOLD = 0.25


def calculate_psi(current_values, bin_edges, expected_proportions):
    counts, _ = np.histogram(current_values, bins=bin_edges)
    actual_proportions = counts / counts.sum()

    epsilon = 1e-6
    actual_proportions = np.clip(actual_proportions, epsilon, None)
    expected_proportions = np.clip(
        expected_proportions,
        epsilon,
        None,
    )

    return float(
        np.sum(
            (actual_proportions - expected_proportions)
            * np.log(actual_proportions / expected_proportions)
        )
    )


def classify_psi(psi):
    if psi >= SIGNIFICANT_PSI_THRESHOLD:
        return "significant_drift"

    if psi >= MODERATE_PSI_THRESHOLD:
        return "moderate_drift"

    return "no_significant_drift"


def assess_drift(input_frame, reference_distribution):
    feature_results = []

    for feature_name, reference in reference_distribution.items():
        psi = calculate_psi(
            input_frame[feature_name].to_numpy(dtype=float),
            reference["bin_edges"],
            reference["proportions"],
        )

        feature_results.append(
            {
                "feature_name": feature_name,
                "psi": round(psi, 6),
                "status": classify_psi(psi),
            }
        )

    statuses = {feature["status"] for feature in feature_results}

    if "significant_drift" in statuses:
        overall_status = "significant_drift"
    elif "moderate_drift" in statuses:
        overall_status = "moderate_drift"
    else:
        overall_status = "no_significant_drift"

    return {
        "sample_count": len(input_frame),
        "overall_status": overall_status,
        "moderate_threshold": MODERATE_PSI_THRESHOLD,
        "significant_threshold": SIGNIFICANT_PSI_THRESHOLD,
        "features": feature_results,
    }