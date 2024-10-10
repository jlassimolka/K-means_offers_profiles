import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Charger les données des fichiers CSV
alumni_data = pd.read_csv('C:\\Users\\Molka\\Desktop\\molkadata\\alumni.csv')
offre_data = pd.read_csv('C:\\Users\\Molka\\Desktop\\molkadata\\offre.csv')

# Appliquer l'encodage one-hot sur la colonne 'domain' des alumni
alumni_encoded = pd.get_dummies(alumni_data['domain'])

# Appliquer l'encodage one-hot sur la colonne 'candidate' des offres
offre_encoded = pd.get_dummies(offre_data['candidate'])

# Effectuer le clustering avec k-means sur les données encodées des alumni
kmeans_alumni = KMeans(n_clusters=4, random_state=0).fit(alumni_encoded)

# Effectuer le clustering avec k-means sur les données encodées des offres
kmeans_offre = KMeans(n_clusters=4, random_state=0).fit(offre_encoded)

# Ajouter les étiquettes de cluster au dataframe alumni
alumni_data['cluster_label'] = kmeans_alumni.labels_

# Ajouter les étiquettes de cluster au dataframe offre
offre_data['cluster_label'] = kmeans_offre.labels_

def find_matching_offres(alumni_id):
    # Trouver le cluster auquel l'alumni appartient
    alumni_cluster = alumni_data.loc[alumni_data['id'] == alumni_id, 'cluster_label'].values[0]
    
    # Trouver les offres correspondantes au cluster de l'alumni
    matching_offres = offre_data[offre_data['cluster_label'] == alumni_cluster]
    
    return matching_offres

# Demander à l'utilisateur de saisir l'ID de l'alumni
alumni_id = int(input("Veuillez entrer l'ID de l'alumni : "))

# Trouver et afficher les offres correspondantes
matching_offres = find_matching_offres(alumni_id)
print("Offres correspondantes pour l'alumni avec l'ID", alumni_id, ":")
print(matching_offres)



# Tracer la courbe
plt.figure(figsize=(10, 6))
offre_count_by_domain = offre_data['candidate'].value_counts()
plt.bar(offre_count_by_domain.index, offre_count_by_domain.values, color='skyblue')
plt.xlabel('Domaine')
plt.ylabel("Nombre d'offres")
plt.title("Nombre d'offres par domaine")
plt.xticks(rotation=45)
plt.tight_layout()

# Afficher les deux visualisations
plt.show()

