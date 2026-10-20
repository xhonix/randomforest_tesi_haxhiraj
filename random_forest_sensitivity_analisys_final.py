import numpy as np
import math
import joblib
import itertools
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def get_feature_intervals(rf_model, n_features):
    """
    Raccoglie tutte le soglie di tutti gli alberi e crea gli intervalli
    (le micro-regioni) per ciascuna feature.
    """
    thresholds = {f: set() for f in range(n_features)}
    
    # Estrai tutte le soglie da tutti gli stimatori
    for estimator in rf_model.estimators_:
        tree = estimator.tree_
        for i in range(tree.node_count):
            if tree.children_left[i] != tree.children_right[i]: # Se non è una foglia
                feature = tree.feature[i]
                threshold = tree.threshold[i]
                thresholds[feature].add(threshold)
    
    intervals = {}
    for f in range(n_features):
        sorted_th = sorted(list(thresholds[f]))
        # Crea intervalli [-inf, t1], (t1, t2], ..., (tn, inf]
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

def compute_rf_hard_voting_probs(rf_model, feature_val, feature_sigma):
    """
    Calcola analiticamente la probabilità di classificazione per una Random Forest.
    """
    n_features = len(feature_val)
    n_classes = rf_model.n_classes_
    
    # 1. Ottieni la griglia multidimensionale
    intervals = get_feature_intervals(rf_model, n_features)
    intervals_list = [intervals[f] for f in range(n_features)]
    
    class_probs = np.zeros(n_classes)
    
    # 2. Pre-calcola le probabilità per ogni singolo segmento per ottimizzare i tempi
    interval_probs = []
    for f in range(n_features):
        f_probs = []
        for lower, upper in intervals_list[f]:
            f_probs.append(prob_in_interval(lower, upper, feature_val[f], feature_sigma[f]))
        interval_probs.append(f_probs)
        
    # 3. Itera su tutte le micro-regioni possibili (prodotto cartesiano)
    for indices in itertools.product(*(range(len(ivls)) for ivls in intervals_list)):
        
        # Probabilità che il dato cada in QUESTA specifica micro-regione
        p_region = 1.0
        for f in range(n_features):
            p_region *= interval_probs[f][indices[f]]
            
        # Se la probabilità è vicina allo zero, saltiamo per efficienza
        if p_region < 1e-12: 
            continue
            
        # 4. Troviamo un "punto di test" al centro della regione per valutare il voto
        test_point = np.zeros(n_features)
        for f in range(n_features):
            lower, upper = intervals_list[f][indices[f]]
            if np.isneginf(lower):
                test_point[f] = upper - 1.0
            elif np.isposinf(upper):
                test_point[f] = lower + 1.0
            else:
                test_point[f] = (lower + upper) / 2.0
                
        # 5. HARD VOTING: Interroghiamo i 4 alberi sul punto di test
        test_point_reshaped = test_point.reshape(1, -1)
        votes = [int(estimator.predict(test_point_reshaped)[0]) for estimator in rf_model.estimators_]
        
        # Risoluzione del voto (con spareggio per la classe dall'indice più basso)
        counts = Counter(votes)
        max_count = max(counts.values())
        winners = [cls for cls, count in counts.items() if count == max_count]
        winning_class = min(winners) # Tie-breaker
        
        # 6. Assegna la probabilità della regione alla classe vincente
        class_probs[winning_class] += p_region
        
    return {cls: class_probs[cls] for cls in range(n_classes)}

if __name__ == "__main__":
    # Addestramento della Random Forest
    n_trees = 7s
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    rf.fit(X_train, y_train)

    # Definizione delle deviazioni standard (rumore gaussiano per ogni sensore)
    std_dev = 0.2
    feature_std_dev = [std_dev, std_dev, std_dev, std_dev]
    
    # Campione teorico da testare (es. una Setosa)
    feature_value = [5, 3, 2.45, 0.7]
    #feature_value = [0, 0, 4.7, 0.7]
    # Calcolo dei risultati analitici
    probs = compute_rf_hard_voting_probs(rf, feature_value, feature_std_dev)
    
    # --- Stampa Probabilità Analitica ---
    print("Probabilità analitica di corretta classificazione per classe:")
    for cls in range(rf.n_classes_):
        print(f"Classe {cls}: {probs.get(cls, 0.0)}")

    # Impostazioni Monte Carlo
    M = 100000  # Numero di iterazioni (M=10^5)
    n_features = len(feature_value)
    np.random.seed(42) # Fissiamo il seed per la riproducibilità del rumore

    # Generazione dei campioni rumorosi
    noisy_samples = np.random.normal(loc=feature_value, scale=feature_std_dev, size=(M, n_features))

    # Predizione in batch su tutti i 100.000 campioni usando l'HARD VOTING
    # Estraiamo le predizioni individuali di ciascun albero (forma: n_alberi x M)
    tree_predictions = np.array([tree.predict(noisy_samples) for tree in rf.estimators_])

    mc_predictions = []
    # Iteriamo sui campioni per applicare il voto di maggioranza esatto
    for i in range(M):
        votes = tree_predictions[:, i]
        counts = Counter(votes)
        max_count = max(counts.values())
        
        # Troviamo i vincitori e applichiamo lo spareggio (indice minore)
        winners = [cls for cls, count in counts.items() if count == max_count]
        winning_class = min(winners)
        mc_predictions.append(winning_class)

    # 6. Calcolo delle probabilità empiriche
    counts_mc = Counter(mc_predictions)
    mc_probs = {cls: counts_mc.get(cls, 0) / M for cls in range(rf.n_classes_)}

    # --- Stampa Probabilità Monte Carlo ---
    print(f"\nProbabilità stimate con Monte Carlo Hard Voting (M={M}):")
    for cls in range(rf.n_classes_):
        print(f"Classe {cls}: {mc_probs.get(cls, 0.0)}")
    
    
    print("\nCaricamento modello Decision Tree e generazione grafici...")

    # 1. Caricamento del modello Decision Tree pre-addestrato
    try:
        dt_model = joblib.load('tree_model.joblib')
        dt_tree = dt_model.tree_
    except FileNotFoundError:
        print("Errore: Il file 'tree_model.joblib' non è stato trovato. Impossibile procedere col confronto.")
        exit()

    
    def get_leaf_conditions_and_class(tree):
        leaf_info = {}
        def recurse(node_id, path):
            left = tree.children_left[node_id]
            right = tree.children_right[node_id]
            if left == right:
                class_index = np.argmax(tree.value[node_id])
                leaf_info[node_id] = {
                    "conditions": path.copy(),
                    "class": int(class_index)
                }
                return
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            recurse(left, path + [(feature, threshold, -1)])
            recurse(right, path + [(feature, threshold, +1)])
        recurse(0, [])
        return leaf_info

    def compute_class_correct_probs_erf(leaf_info, feature_theoretical_val, feature_theoretical_sigma):
        n_classes = max(info["class"] for info in leaf_info.values()) + 1
        leaf_probs = {}
        def prob_feature_in_interval(f, lower, upper):
            mu = feature_theoretical_val[f]
            sigma = feature_theoretical_sigma[f]
            def cdf(x):
                if np.isneginf(x):
                    return 0.0
                return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))
            return max(0.0, cdf(upper) - cdf(lower))

        for leaf_id, info in leaf_info.items():
            intervals = {}
            for (f, thr, s) in info["conditions"]:
                if f not in intervals:
                    intervals[f] = [-np.inf, np.inf]
                if s == -1:
                    intervals[f][1] = min(intervals[f][1], thr)
                else:
                    intervals[f][0] = max(intervals[f][0], np.nextafter(thr, np.inf))
            probs = np.zeros(n_classes)
            for cls in range(n_classes):
                p = 1.0
                for f in intervals:
                    lower, upper = intervals[f]
                    p *= prob_feature_in_interval(f, lower, upper)
                probs[cls] = p
            leaf_probs[leaf_id] = probs

        class_correct_prob = np.zeros(n_classes)
        for leaf_id, probs in leaf_probs.items():
            pred_cls = leaf_info[leaf_id]["class"]
            for cls in range(n_classes):
                if pred_cls == cls:
                    class_correct_prob[cls] += probs[cls]
        return {cls: class_correct_prob[cls] for cls in range(n_classes)}

    # Estraiamo i percorsi logici del singolo albero appena caricato
    dt_leaf_info = get_leaf_conditions_and_class(dt_tree)



    # A. GRAFICO CONFRONTO SIGMA PER RANDOM FOREST

    print("\nInizio calcolo per il grafico probabilità vs petal width per diversi sigma (Random Forest)...")
    
    # Valori di sigma mostrati nell'immagine di riferimento
    sigma_values = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    
    # Range della Petal width (asse x) da 0.2 a 1.2 cm
    petal_widths_sigma = np.linspace(0.2, 1.2, 100)
    
    plt.figure(figsize=(10, 6))
    
    # Calcolo delle probabilità per ciascun valore di sigma
    for sig in sigma_values:
        f_sigma_sig = [sig] * 4
        probs_rf_sig_curve = []
        
        for pw in petal_widths_sigma:
            #f_val = [5, 3.4, 2.45, pw]
            f_val = [5, 3.4, 4.95, pw] 
            # Estraiamo la probabilità analitica per la classe Setosa (indice 0)
            prob = compute_rf_hard_voting_probs(rf, f_val, f_sigma_sig)[0]
            probs_rf_sig_curve.append(prob)
            
        plt.plot(petal_widths_sigma, probs_rf_sig_curve, label=f'σ={sig:.2f} cm')

    # Dettagli di formattazione del grafico
    plt.xlabel('Petal width [cm]')
    plt.ylabel('Probability of correct classification (Setosa)')
    plt.title('Random Forest: Probability versus petal width for different σ')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("Grafico per i diversi sigma generato con successo!")


    # B. GRAFICO CONFRONTO SIGMA PER RANDOM FOREST (CLASSE VIRGINICA)

    print("\nInizio calcolo per il grafico probabilità vs petal width (Classe Virginica)...")
    
    # Valori di sigma mostrati nell'immagine di riferimento
    sigma_values = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    
    colors = ['navy', 'darkviolet', 'mediumvioletred', 'lightcoral', 'sandybrown', 'yellow']
    
    # Range della Petal width (asse x) da circa 0.2 a 2.1 cm
    petal_widths_virginica = np.linspace(0.2, 2.1, 100)
    
    plt.figure(figsize=(8, 6))
    
    # Calcolo delle probabilità per ciascun valore di sigma
    for sig, color in zip(sigma_values, colors):
        f_sigma_sig = [sig] * 4
        probs_rf_sig_virginica = []
        
        for pw in petal_widths_virginica:
            # Fissiamo Sepal Length e Sepal Width ai valori di default (es. 5.0, 3.4),
            # Petal Length = 4.95 cm, e Petal Width variabile
            f_val = [5.0, 3.4, 4.95, pw] 
            
            # Estraiamo la probabilità analitica per la classe Virginica (indice 2)
            prob = compute_rf_hard_voting_probs(rf, f_val, f_sigma_sig)[2]
            probs_rf_sig_virginica.append(prob)
            
        plt.plot(petal_widths_virginica, probs_rf_sig_virginica, 
                 label=f'σ={sig:.2f} [cm]', color=color)

    # Dettagli di formattazione del grafico
    plt.xlabel('Petal width [cm]')
    plt.ylabel('Probability of correct classification (Virginica)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='-', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    print("Grafico per la classe Virginica generato con successo!")
