import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Gestione dei voti
def resolve_voting(votes):
    """
    Risolve il voto per le 3 classi originarie:
    - 1 vincitore: restituisce la classe (0, 1 o 2)
    - 2 o più vincitori: restituisce la categoria 'tie'
    """
    counts = Counter(votes)
    max_count = max(counts.values())
    winners = [cls for cls, count in counts.items() if count == max_count]

    if len(winners) == 1:
        return winners[0]
    return 'tie'

def get_feature_intervals(rf_model, n_features):
    """
    Raccoglie tutte le soglie di tutti gli alberi e crea gli intervalli
    (le micro-regioni) per ciascuna feature.
    """
    thresholds = {f: set() for f in range(n_features)}

    for estimator in rf_model.estimators_:
        tree = estimator.tree_
        for i in range(tree.node_count):
            if tree.children_left[i] != tree.children_right[i]:
                feature = tree.feature[i]
                threshold = tree.threshold[i]
                thresholds[feature].add(threshold)

    intervals = {}
    for f in range(n_features):
        sorted_th = sorted(list(thresholds[f]))
        f_intervals = []
        prev = -np.inf
        for th in sorted_th:
            f_intervals.append((prev, th))
            prev = th
        f_intervals.append((prev, np.inf))
        intervals[f] = f_intervals

    return intervals

def prob_in_interval(lower, upper, mu, sigma):
    """Calcola la probabilità gaussiana in un intervallo tramite erf."""
    def cdf(x):
        if np.isneginf(x): return 0.0
        if np.isposinf(x): return 1.0
        return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))
    return max(0.0, cdf(upper) - cdf(lower))


class RFRegionEvaluator:
    """
    Gestisce il pre-calcolo delle micro-regioni della Random Forest.
    Valuta la struttura degli alberi UNA SOLA VOLTA.
    """
    def __init__(self, rf_model, n_features):
        self.rf_model = rf_model
        self.n_features = n_features
        # Dizionario: Classe -> Lista di combinazioni di indici delle regioni
        self.class_to_regions = {0: [], 1: [], 2: [], 'tie': []}
        self.intervals_list = []
        self._precompute_regions()

    def _precompute_regions(self):
        intervals = get_feature_intervals(self.rf_model, self.n_features)
        self.intervals_list = [intervals[f] for f in range(self.n_features)]

        # Genera tutte le combinazioni possibili di indici per le regioni
        all_indices = list(itertools.product(*(range(len(ivls)) for ivls in self.intervals_list)))
        
        # Vettorizzazione: Prepariamo tutti i test points contemporaneamente
        test_points = np.zeros((len(all_indices), self.n_features))
        
        for row_idx, indices in enumerate(all_indices):
            for f in range(self.n_features):
                lower, upper = self.intervals_list[f][indices[f]]
                if np.isneginf(lower):
                    test_points[row_idx, f] = upper - 1.0
                elif np.isposinf(upper):
                    test_points[row_idx, f] = lower + 1.0
                else:
                    test_points[row_idx, f] = (lower + upper) / 2.0

        # Calcolo BATCH delle predizioni (Estremamente veloce)
        tree_predictions = np.array([tree.predict(test_points) for tree in self.rf_model.estimators_])

        # Assegnazione delle categorie alle regioni
        for row_idx, indices in enumerate(all_indices):
            votes = tree_predictions[:, row_idx]
            winning_class = resolve_voting(votes)
            
            if winning_class not in self.class_to_regions:
                self.class_to_regions[winning_class] = []
            self.class_to_regions[winning_class].append(indices)

    def compute_probs(self, feature_val, feature_sigma):
        """
        Calcola analiticamente la probabilità di classificazione basandosi
        sulle regioni pre-calcolate.
        """
        # 1. Calcola probabilità 1D per ogni intervallo
        interval_probs = []
        for f in range(self.n_features):
            f_probs = []
            for lower, upper in self.intervals_list[f]:
                f_probs.append(prob_in_interval(lower, upper, feature_val[f], feature_sigma[f]))
            interval_probs.append(f_probs)

        # 2. Somma le probabilità calcolando solo il prodotto
        class_probs = {k: 0.0 for k in self.class_to_regions.keys()}
        
        for cls, regions in self.class_to_regions.items():
            for indices in regions:
                p_region = 1.0
                for f in range(self.n_features):
                    p_region *= interval_probs[f][indices[f]]
                    if p_region == 0.0:  # Early exit se la probabilità si azzera
                        break
                class_probs[cls] += p_region

        return class_probs


def generate_monte_carlo_vs_analytical_table(evaluator, feature_val, feature_sigma):
    """
    Genera una tabella di confronto con 6 decimali per le 4 categorie.
    """
    categories = [0, 1, 2, 'tie']

    # Usa l'evaluator pre-calcolato anziché ricalcolare tutto
    analytical_probs_dict = evaluator.compute_probs(feature_val, feature_sigma)
    p_analitiche = [analytical_probs_dict.get(c, 0.0) for c in categories]

    m_values = [10**2, 10**3, 10**4, 10**5, 10**6]
    n_features = len(feature_val)

    print(f"{'M':<8} | {'Monte Carlo (0, 1, 2, tie)':<38} | {'Analitico (0, 1, 2, tie)':<38} | {'Δ [%]':<10}")
    print("-" * 102)

    for M in m_values:
        np.random.seed(42)
        noisy_samples = np.random.normal(loc=feature_val, scale=feature_sigma, size=(M, n_features))
        tree_predictions = np.array([tree.predict(noisy_samples) for tree in evaluator.rf_model.estimators_])

        mc_predictions = []
        for i in range(M):
            votes = tree_predictions[:, i]
            winning_class = resolve_voting(votes)
            mc_predictions.append(winning_class)

        counts_mc = Counter(mc_predictions)
        p_mc = [counts_mc.get(c, 0) / M for c in categories]

        delta = math.sqrt(sum((p_mc[i] - p_analitiche[i])**2 for i in range(len(categories)))) * 100

        mc_str = f"{p_mc[0]:.6f}  {p_mc[1]:.6f}  {p_mc[2]:.6f}  {p_mc[3]:.6f}"
        an_str = f"{p_analitiche[0]:.6f}  {p_analitiche[1]:.6f}  {p_analitiche[2]:.6f}  {p_analitiche[3]:.6f}"
        print(f"10^{int(math.log10(M)):<5} | {mc_str:<38} | {an_str:<38} | {delta:.4f}")


if __name__ == "__main__":
    n_trees = 7
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    rf.fit(X_train, y_train)

    test_feature_value = [5.5, 3, 4.95, 0.8]
    std_dev = 0.2
    test_feature_std_dev = [std_dev, std_dev, std_dev, std_dev]

    # ------ INIZIALIZZAZIONE DELL'EVALUATOR ------
    # Questa operazione calcola le microregioni una tantum
    print("Inizializzazione delle regioni analitiche (Attendere...)")
    rf_evaluator = RFRegionEvaluator(rf, n_features=4)
    print("Inizializzazione completata!\n")

    print("\n------------------------------ Validazione del Framework vs Monte Carlo ------------------------------\n")
    generate_monte_carlo_vs_analytical_table(rf_evaluator, test_feature_value, test_feature_std_dev)

    feature_std_dev = [std_dev, std_dev, std_dev, std_dev]
    feature_value = [5, 3.4, 2.45, 0.7]

    probs = rf_evaluator.compute_probs(feature_value, feature_std_dev)

    print("\n\nProbabilità analitica per tutte le categorie (3 classi + tie):")
    for cls in [0, 1, 2, 'tie']:
        print(f"Categoria '{cls}': {probs.get(cls, 0.0):.6f}")

    print("\nGenerazione grafici...")

    # A. GRAFICO CONFRONTO SIGMA PER RANDOM FOREST
    print("\nInizio calcolo per il grafico probabilità vs petal width per diversi sigma (Random Forest)...")
    sigma_values = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    petal_widths_sigma = np.linspace(0.2, 1.2, 100)

    plt.figure(figsize=(10, 6))
    for sig in sigma_values:
        f_sigma_sig = [sig] * 4
        probs_rf_sig_curve = []
        for pw in petal_widths_sigma:
            f_val = [5, 3.4, 2.45, pw]
            prob = rf_evaluator.compute_probs(f_val, f_sigma_sig)[0]
            probs_rf_sig_curve.append(prob)
        plt.plot(petal_widths_sigma, probs_rf_sig_curve, label=f'σ={sig:.2f} cm')

    plt.xlabel('Petal width [cm]')
    plt.ylabel('Probability of correct classification (Setosa)')
    plt.title('Random Forest: Probability versus petal width for different σ')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # B. GRAFICO CONFRONTO SIGMA PER RANDOM FOREST (CLASSE VIRGINICA)
    print("Inizio calcolo per il grafico probabilità vs petal width (Classe Virginica)...")
    colors = ['navy', 'darkviolet', 'mediumvioletred', 'lightcoral', 'sandybrown', 'yellow']
    petal_widths_virginica = np.linspace(0.2, 2.1, 100)

    plt.figure(figsize=(8, 6))
    for sig, color in zip(sigma_values, colors):
        f_sigma_sig = [sig] * 4
        probs_rf_sig_virginica = []
        for pw in petal_widths_virginica:
            f_val = [5.0, 3.4, 4.95, pw]
            prob = rf_evaluator.compute_probs(f_val, f_sigma_sig)[2]
            probs_rf_sig_virginica.append(prob)
        plt.plot(petal_widths_virginica, probs_rf_sig_virginica,
                 label=f'σ={sig:.2f} [cm]', color=color)

    plt.xlabel('Petal width [cm]')
    plt.ylabel('Probability of correct classification (Virginica)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='-', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # C. GRAFICO CATEGORIA PAREGGIO IN FUNZIONE DI SIGMA
    print("Inizio calcolo per il grafico probabilità pareggio vs Sigma...")

    sigmas_ties = np.linspace(0.01, 1.0, 100)
    probs_disagreement = []

    f_val_tie = [5.5, 3, 4.95, 0.8]

    for sig in sigmas_ties:
        f_sigma_tie = [sig] * 4
        probs_tie = rf_evaluator.compute_probs(f_val_tie, f_sigma_tie)
        probs_disagreement.append(probs_tie.get('tie', 0.0))

    plt.figure(figsize=(10, 6))
    plt.plot(sigmas_ties, probs_disagreement, label="tie (non maggioranza)", color='red', linewidth=2.5)

    plt.xlabel('Standard Deviation σ [cm]')
    plt.ylabel('Tie Probability')
    plt.title('Probabilità di pareggio al variare di σ')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='-', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("Grafico per la categoria di pareggio vs σ generato con successo!")
