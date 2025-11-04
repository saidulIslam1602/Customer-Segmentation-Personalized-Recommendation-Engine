"""
Advanced A/B Testing Framework for Business Intelligence
Provides statistical testing, experiment design, and performance analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import os
import warnings

warnings.filterwarnings("ignore")


class ABTestingFramework:
    """
    Comprehensive A/B Testing Framework for Business Intelligence

    Features:
    - Statistical significance testing
    - Power analysis and sample size calculation
    - Multi-variant testing (A/B/C/D...)
    - Sequential testing and early stopping
    - Business metrics tracking
    - Experiment lifecycle management
    - Automated reporting and insights
    """

    def __init__(self, results_dir="reports/ab_tests"):
        """Initialize A/B testing framework"""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.experiments = {}
        self.active_experiments = []
        self.experiment_history = []

    def design_experiment(
        self,
        experiment_name,
        variants,
        success_metric,
        minimum_effect_size=0.05,
        power=0.8,
        alpha=0.05,
    ):
        """Design a new A/B test experiment"""
        print(f"🧪 Designing experiment: {experiment_name}")

        # Calculate required sample size
        sample_size = self._calculate_sample_size(minimum_effect_size, power, alpha)

        experiment = {
            "name": experiment_name,
            "variants": variants,
            "success_metric": success_metric,
            "minimum_effect_size": minimum_effect_size,
            "power": power,
            "alpha": alpha,
            "required_sample_size": sample_size,
            "start_date": datetime.now(),
            "status": "designed",
            "results": {},
            "participants": {variant: [] for variant in variants},
        }

        self.experiments[experiment_name] = experiment

        print(f"   ✅ Experiment designed")
        print(f"   📊 Variants: {variants}")
        print(f"   🎯 Success metric: {success_metric}")
        print(f"   📈 Required sample size per variant: {sample_size}")
        print(f"   ⚡ Minimum detectable effect: {minimum_effect_size:.1%}")

        return experiment

    def _calculate_sample_size(self, effect_size, power=0.8, alpha=0.05):
        """Calculate required sample size for experiment"""
        # Using Cohen's formula for two-sample t-test
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Assuming equal variance and sample sizes
        sample_size = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(sample_size))

    def start_experiment(self, experiment_name):
        """Start an A/B test experiment"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        experiment = self.experiments[experiment_name]
        experiment["status"] = "running"
        experiment["actual_start_date"] = datetime.now()

        self.active_experiments.append(experiment_name)

        print(f"🚀 Started experiment: {experiment_name}")
        print(f"   📅 Start date: {experiment['actual_start_date']}")

        return experiment

    def assign_variant(self, experiment_name, user_id, method="random"):
        """Assign a user to a variant in an experiment"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        experiment = self.experiments[experiment_name]
        variants = experiment["variants"]

        if method == "random":
            # Simple random assignment
            variant = np.random.choice(variants)
        elif method == "balanced":
            # Balanced assignment to keep group sizes equal
            current_sizes = {v: len(experiment["participants"][v]) for v in variants}
            min_size = min(current_sizes.values())
            min_variants = [v for v, size in current_sizes.items() if size == min_size]
            variant = np.random.choice(min_variants)
        else:
            raise ValueError(f"Unknown assignment method: {method}")

        # Record assignment
        experiment["participants"][variant].append(
            {"user_id": user_id, "assigned_date": datetime.now(), "variant": variant}
        )

        return variant

    def record_outcome(self, experiment_name, user_id, outcome_value):
        """Record an outcome for a user in an experiment"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        experiment = self.experiments[experiment_name]

        # Find user's variant
        user_variant = None
        for variant, participants in experiment["participants"].items():
            for participant in participants:
                if participant["user_id"] == user_id:
                    participant["outcome"] = outcome_value
                    participant["outcome_date"] = datetime.now()
                    user_variant = variant
                    break
            if user_variant:
                break

        if not user_variant:
            print(f"⚠️  User {user_id} not found in experiment {experiment_name}")
            return False

        return True

    def analyze_experiment(self, experiment_name, early_stopping=True):
        """Analyze experiment results and statistical significance"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        experiment = self.experiments[experiment_name]

        print(f"📊 Analyzing experiment: {experiment_name}")

        # Collect outcomes by variant
        variant_outcomes = {}
        for variant, participants in experiment["participants"].items():
            outcomes = [p["outcome"] for p in participants if "outcome" in p]
            variant_outcomes[variant] = outcomes

        # Check if we have enough data
        min_sample_size = experiment["required_sample_size"]
        sufficient_data = all(
            len(outcomes) >= min_sample_size for outcomes in variant_outcomes.values()
        )

        if not sufficient_data and not early_stopping:
            print("⚠️  Insufficient data for analysis")
            return None

        # Statistical analysis
        results = self._perform_statistical_tests(variant_outcomes, experiment)

        # Business impact analysis
        business_impact = self._calculate_business_impact(variant_outcomes, experiment)

        # Combine results
        analysis_results = {
            "experiment_name": experiment_name,
            "analysis_date": datetime.now(),
            "statistical_results": results,
            "business_impact": business_impact,
            "sample_sizes": {
                v: len(outcomes) for v, outcomes in variant_outcomes.items()
            },
            "sufficient_data": sufficient_data,
            "recommendation": self._get_recommendation(results, business_impact),
        }

        experiment["results"] = analysis_results

        # Print summary
        self._print_analysis_summary(analysis_results)

        return analysis_results

    def _perform_statistical_tests(self, variant_outcomes, experiment):
        """Perform statistical significance tests"""
        results = {}
        variants = list(variant_outcomes.keys())

        if len(variants) == 2:
            # Two-sample t-test
            group_a, group_b = variants
            outcomes_a = variant_outcomes[group_a]
            outcomes_b = variant_outcomes[group_b]

            if len(outcomes_a) > 1 and len(outcomes_b) > 1:
                t_stat, p_value = stats.ttest_ind(outcomes_a, outcomes_b)

                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (
                        (len(outcomes_a) - 1) * np.var(outcomes_a, ddof=1)
                        + (len(outcomes_b) - 1) * np.var(outcomes_b, ddof=1)
                    )
                    / (len(outcomes_a) + len(outcomes_b) - 2)
                )

                cohens_d = (np.mean(outcomes_a) - np.mean(outcomes_b)) / pooled_std

                results = {
                    "test_type": "t-test",
                    "statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < experiment["alpha"],
                    "effect_size": cohens_d,
                    "confidence_interval": self._calculate_confidence_interval(
                        outcomes_a, outcomes_b, experiment["alpha"]
                    ),
                }

        elif len(variants) > 2:
            # ANOVA for multiple variants
            outcome_groups = [variant_outcomes[v] for v in variants]
            f_stat, p_value = stats.f_oneway(*outcome_groups)

            results = {
                "test_type": "ANOVA",
                "statistic": f_stat,
                "p_value": p_value,
                "significant": p_value < experiment["alpha"],
            }

            # Post-hoc pairwise comparisons if significant
            if results["significant"]:
                pairwise_results = {}
                for i, variant_a in enumerate(variants):
                    for j, variant_b in enumerate(variants[i + 1 :], i + 1):
                        t_stat, p_val = stats.ttest_ind(
                            variant_outcomes[variant_a], variant_outcomes[variant_b]
                        )
                        pairwise_results[f"{variant_a}_vs_{variant_b}"] = {
                            "p_value": p_val,
                            "significant": p_val < experiment["alpha"],
                        }

                results["pairwise_comparisons"] = pairwise_results

        return results

    def _calculate_confidence_interval(self, group_a, group_b, alpha):
        """Calculate confidence interval for difference in means"""
        mean_a, mean_b = np.mean(group_a), np.mean(group_b)
        var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
        n_a, n_b = len(group_a), len(group_b)

        # Standard error of difference
        se_diff = np.sqrt(var_a / n_a + var_b / n_b)

        # Degrees of freedom (Welch's formula)
        df = (var_a / n_a + var_b / n_b) ** 2 / (
            (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        )

        # Critical value
        t_critical = stats.t.ppf(1 - alpha / 2, df)

        # Confidence interval
        diff = mean_a - mean_b
        margin_error = t_critical * se_diff

        return {
            "difference": diff,
            "lower_bound": diff - margin_error,
            "upper_bound": diff + margin_error,
            "confidence_level": 1 - alpha,
        }

    def _calculate_business_impact(self, variant_outcomes, experiment):
        """Calculate business impact metrics"""
        business_impact = {}

        for variant, outcomes in variant_outcomes.items():
            if len(outcomes) > 0:
                business_impact[variant] = {
                    "mean": np.mean(outcomes),
                    "median": np.median(outcomes),
                    "std": np.std(outcomes),
                    "sample_size": len(outcomes),
                    "total_value": np.sum(outcomes),
                }

        # Calculate relative improvements
        if len(variant_outcomes) == 2:
            variants = list(variant_outcomes.keys())
            control, treatment = variants[0], variants[1]

            if control in business_impact and treatment in business_impact:
                control_mean = business_impact[control]["mean"]
                treatment_mean = business_impact[treatment]["mean"]

                if control_mean != 0:
                    relative_improvement = (
                        treatment_mean - control_mean
                    ) / control_mean
                    business_impact["relative_improvement"] = relative_improvement
                    business_impact["absolute_improvement"] = (
                        treatment_mean - control_mean
                    )

        return business_impact

    def _get_recommendation(self, statistical_results, business_impact):
        """Generate recommendation based on results"""
        if not statistical_results.get("significant", False):
            return {
                "decision": "no_change",
                "reason": "No statistically significant difference detected",
                "confidence": "high",
            }

        # For two-variant tests
        if "relative_improvement" in business_impact:
            improvement = business_impact["relative_improvement"]

            if improvement > 0.05:  # 5% improvement threshold
                return {
                    "decision": "implement_treatment",
                    "reason": f"Significant improvement of {improvement:.1%} detected",
                    "confidence": "high",
                }
            elif improvement > 0:
                return {
                    "decision": "consider_implementation",
                    "reason": f"Small but significant improvement of {improvement:.1%}",
                    "confidence": "medium",
                }
            else:
                return {
                    "decision": "no_change",
                    "reason": f"Treatment performs worse by {abs(improvement):.1%}",
                    "confidence": "high",
                }

        return {
            "decision": "further_analysis_needed",
            "reason": "Complex multi-variant results require deeper analysis",
            "confidence": "low",
        }

    def _print_analysis_summary(self, results):
        """Print a summary of analysis results"""
        print(f"\n📋 EXPERIMENT ANALYSIS SUMMARY")
        print(f"=" * 50)

        print(f"🧪 Experiment: {results['experiment_name']}")
        print(f"📅 Analysis Date: {results['analysis_date']}")

        # Sample sizes
        print(f"\n📊 Sample Sizes:")
        for variant, size in results["sample_sizes"].items():
            print(f"   {variant}: {size:,} participants")

        # Statistical results
        stat_results = results["statistical_results"]
        print(f"\n🔬 Statistical Results:")
        print(f"   Test Type: {stat_results.get('test_type', 'N/A')}")
        print(f"   P-value: {stat_results.get('p_value', 0):.6f}")
        print(f"   Significant: {'✅' if stat_results.get('significant') else '❌'}")

        if "effect_size" in stat_results:
            print(f"   Effect Size (Cohen's d): {stat_results['effect_size']:.4f}")

        # Business impact
        business = results["business_impact"]
        print(f"\n💰 Business Impact:")
        for variant, metrics in business.items():
            if isinstance(metrics, dict) and "mean" in metrics:
                print(
                    f"   {variant}: Mean = {metrics['mean']:.4f}, Total = {metrics['total_value']:.2f}"
                )

        if "relative_improvement" in business:
            improvement = business["relative_improvement"]
            print(f"   📈 Relative Improvement: {improvement:.2%}")

        # Recommendation
        rec = results["recommendation"]
        print(f"\n🎯 Recommendation:")
        print(f"   Decision: {rec['decision'].replace('_', ' ').title()}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Confidence: {rec['confidence'].title()}")

    def stop_experiment(self, experiment_name, reason="completed"):
        """Stop a running experiment"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        experiment = self.experiments[experiment_name]
        experiment["status"] = "stopped"
        experiment["end_date"] = datetime.now()
        experiment["stop_reason"] = reason

        if experiment_name in self.active_experiments:
            self.active_experiments.remove(experiment_name)

        self.experiment_history.append(experiment_name)

        print(f"🛑 Stopped experiment: {experiment_name}")
        print(f"   Reason: {reason}")

        return experiment

    def get_experiment_summary(self):
        """Get summary of all experiments"""
        summary = {
            "total_experiments": len(self.experiments),
            "active_experiments": len(self.active_experiments),
            "completed_experiments": len(self.experiment_history),
            "experiments_by_status": {},
        }

        # Count by status
        for exp in self.experiments.values():
            status = exp["status"]
            summary["experiments_by_status"][status] = (
                summary["experiments_by_status"].get(status, 0) + 1
            )

        return summary

    def save_experiment_results(self, experiment_name):
        """Save experiment results to file"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        experiment = self.experiments[experiment_name]

        # Save to JSON
        filename = f"{self.results_dir}/{experiment_name}_results.json"
        with open(filename, "w") as f:
            json.dump(experiment, f, indent=2, default=str)

        print(f"💾 Experiment results saved: {filename}")

        return filename


def main():
    """Demo of A/B testing framework"""
    print("🧪 A/B TESTING FRAMEWORK DEMO")
    print("=" * 50)

    # Initialize framework
    ab_test = ABTestingFramework()

    # Design experiment
    experiment = ab_test.design_experiment(
        experiment_name="recommendation_algorithm_test",
        variants=["control", "enhanced"],
        success_metric="click_through_rate",
        minimum_effect_size=0.05,
    )

    # Start experiment
    ab_test.start_experiment("recommendation_algorithm_test")

    # Simulate some data
    np.random.seed(42)

    # Assign users and record outcomes
    for user_id in range(1000):
        variant = ab_test.assign_variant(
            "recommendation_algorithm_test", f"user_{user_id}"
        )

        # Simulate outcomes (enhanced variant performs better)
        if variant == "control":
            outcome = np.random.normal(0.1, 0.05)  # 10% CTR
        else:
            outcome = np.random.normal(0.12, 0.05)  # 12% CTR

        ab_test.record_outcome(
            "recommendation_algorithm_test", f"user_{user_id}", max(0, outcome)
        )

    # Analyze results
    results = ab_test.analyze_experiment("recommendation_algorithm_test")

    # Save results
    ab_test.save_experiment_results("recommendation_algorithm_test")

    print(f"\n✅ A/B Testing Framework Demo Completed!")


if __name__ == "__main__":
    main()
