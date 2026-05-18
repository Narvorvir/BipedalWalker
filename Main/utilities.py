import pandas as pd
import os
from datetime import datetime

def sauvegarder_tentative(nom_fichier, parametres, recompenses):
    """
    Sauvegarde les paramètres et les résultats dans une nouvelle feuille Excel.
    """
    

    parametres['date_essai'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    df_params = pd.DataFrame([parametres])

    df_recompenses = pd.DataFrame({
        'Épisode': range(1, len(recompenses) + 1),
        'Récompense': recompenses
    })

    nom_feuille = f"Essai_{datetime.now().strftime('%Hh%M_%S')}"

    if not os.path.exists(nom_fichier):
        # Si le fichier n'existe pas, on le crée
        with pd.ExcelWriter(nom_fichier, engine='openpyxl') as writer:
            df_params.to_excel(writer, sheet_name=nom_feuille, index=False, startrow=0)
            df_recompenses.to_excel(writer, sheet_name=nom_feuille, index=False, startrow=4)
    else:
        # Si le fichier existe, on ajoute l'onglet (mode 'a' pour append)
        with pd.ExcelWriter(nom_fichier, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            df_params.to_excel(writer, sheet_name=nom_feuille, index=False, startrow=0)
            df_recompenses.to_excel(writer, sheet_name=nom_feuille, index=False, startrow=4)

    print(f"Sauvegarde réussie dans l'onglet: {nom_feuille}")

# --- EXEMPLE D'UTILISATION ---

# Vos paramètres d'environnement
mes_params = {
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon': 0.1,
    'env_name': 'CartPole-v1'
}

# Vos résultats (ex: 10 épisodes)
mes_recompenses = [10, 15, 22, 40, 50, 48, 60, 75, 90, 100]

# Appel de la fonction
sauvegarder_tentative("resultats_rl.xlsx", mes_params, mes_recompenses)