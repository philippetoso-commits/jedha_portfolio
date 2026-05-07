#!/usr/bin/env python3
"""
Generate fake personal data for license plates.
"""

import csv
import random
from pathlib import Path

# Listes de données françaises
PRENOMS = [
    "Jean", "Marie", "Pierre", "Sophie", "Luc", "Anne", "Michel", "Isabelle",
    "François", "Nathalie", "Philippe", "Catherine", "Alain", "Sylvie", "Bernard",
    "Martine", "Jacques", "Christine", "Daniel", "Monique", "Claude", "Françoise",
    "Gérard", "Nicole", "André", "Brigitte", "René", "Jacqueline", "Paul", "Chantal",
    "Laurent", "Sandrine", "Olivier", "Valérie", "Stéphane", "Céline", "Thierry",
    "Véronique", "Patrick", "Corinne", "Nicolas", "Émilie", "Julien", "Aurélie",
    "Thomas", "Julie", "Alexandre", "Marion", "Maxime", "Camille"
]

NOMS = [
    "Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand",
    "Leroy", "Moreau", "Simon", "Laurent", "Lefebvre", "Michel", "Garcia", "David",
    "Bertrand", "Roux", "Vincent", "Fournier", "Morel", "Girard", "André", "Lefevre",
    "Mercier", "Dupont", "Lambert", "Bonnet", "François", "Martinez", "Legrand",
    "Garnier", "Faure", "Rousseau", "Blanc", "Guerin", "Muller", "Henry", "Roussel",
    "Nicolas", "Perrin", "Morin", "Mathieu", "Clement", "Gauthier", "Dumont", "Lopez",
    "Fontaine", "Chevalier", "Robin"
]

RUES = [
    "Rue de la République", "Avenue des Champs-Élysées", "Boulevard Saint-Michel",
    "Rue Victor Hugo", "Avenue Jean Jaurès", "Rue Gambetta", "Place de la Liberté",
    "Rue du Général de Gaulle", "Avenue de la Paix", "Rue Pasteur", "Boulevard Voltaire",
    "Rue Molière", "Avenue Foch", "Rue Lafayette", "Boulevard Haussmann", "Rue Balzac",
    "Avenue Montaigne", "Rue Racine", "Boulevard Diderot", "Rue Corneille",
    "Avenue Carnot", "Rue Thiers", "Boulevard Clemenceau", "Rue Zola", "Avenue Joffre"
]

VILLES = [
    ("Paris", "75001"), ("Lyon", "69001"), ("Marseille", "13001"), ("Toulouse", "31000"),
    ("Nice", "06000"), ("Nantes", "44000"), ("Strasbourg", "67000"), ("Montpellier", "34000"),
    ("Bordeaux", "33000"), ("Lille", "59000"), ("Rennes", "35000"), ("Reims", "51100"),
    ("Le Havre", "76600"), ("Saint-Étienne", "42000"), ("Toulon", "83000"), ("Grenoble", "38000"),
    ("Dijon", "21000"), ("Angers", "49000"), ("Nîmes", "30000"), ("Villeurbanne", "69100"),
    ("Le Mans", "72000"), ("Aix-en-Provence", "13100"), ("Clermont-Ferrand", "63000"),
    ("Brest", "29200"), ("Tours", "37000"), ("Amiens", "80000"), ("Limoges", "87000"),
    ("Annecy", "74000"), ("Perpignan", "66000"), ("Boulogne-Billancourt", "92100")
]

MARQUES = [
    "Renault", "Peugeot", "Citroën", "Volkswagen", "Dacia", "Toyota", "Ford", 
    "BMW", "Mercedes-Benz", "Audi", "Fiat", "Opel", "Nissan", "Kia", 
    "Hyundai", "Seat", "Skoda", "Volvo", "Mini", "Suzuki"
]

def generate_phone():
    """Génère un numéro de téléphone français valide."""
    prefixes = ["06", "07"]  # Mobile
    prefix = random.choice(prefixes)
    number = ''.join([str(random.randint(0, 9)) for _ in range(8)])
    # Format: 06 12 34 56 78
    return f"{prefix} {number[0:2]} {number[2:4]} {number[4:6]} {number[6:8]}"

def generate_age():
    """Génère un âge réaliste pour un conducteur."""
    # Distribution réaliste: plus de conducteurs entre 25-65 ans
    return random.choices(
        range(18, 85),
        weights=[1]*7 + [3]*10 + [5]*30 + [3]*15 + [1]*5
    )[0]

def generate_address():
    """Génère une adresse complète."""
    numero = random.randint(1, 250)
    rue = random.choice(RUES)
    ville, code_postal = random.choice(VILLES)
    
    return {
        'adresse': f"{numero} {rue}",
        'ville': ville,
        'code_postal': code_postal
    }

def main():
    """Génère le fichier CSV enrichi."""
    
    print("=" * 80)
    print("🎲 Génération de données fictives pour les plaques")
    print("=" * 80)
    
    # Résoudre les chemins relatifs au script
    base_dir = Path(__file__).parent
    input_file = base_dir / "plaques_extraites.csv"
    output_file = base_dir / "plaques_avec_donnees.csv"
    
    print(f"\n📖 Lecture de {input_file}...")
    
    if not input_file.exists():
        print(f"❌ Erreur: {input_file} n'existe pas.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        plates = list(reader)
    
    print(f"✅ {len(plates)} plaques trouvées")
    print(f"\n🎲 Génération de données aléatoires...")
    
    # Générer les données enrichies
    enriched_data = []
    
    for plate in plates:
        prenom = random.choice(PRENOMS)
        nom = random.choice(NOMS)
        marque = random.choice(MARQUES)
        age = generate_age()
        telephone = generate_phone()
        address = generate_address()
        abonnement = random.choice(["oui", "non"])
        acces = random.choice(["oui", "non"])
        
        enriched_data.append({
            'plaque': plate['plate_number'],
            'marque': marque,
            'nom': nom,
            'prenom': prenom,
            'age': age,
            'telephone': telephone,
            'adresse': address['adresse'],
            'ville': address['ville'],
            'code_postal': address['code_postal'],
            'abonnement': abonnement,
            'acces': acces
        })
    
    # Écrire le nouveau CSV
    print(f"\n💾 Écriture dans {output_file}...")
    
    fieldnames = ['plaque', 'marque', 'nom', 'prenom', 'age', 'telephone', 'adresse', 
                  'ville', 'code_postal', 'abonnement', 'acces']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_data)
    
    # Statistiques
    print(f"\n{'='*80}")
    print("📊 STATISTIQUES")
    print(f"{'='*80}")
    print(f"Total d'entrées: {len(enriched_data)}")
    print(f"Marques représentées: {len(set(d['marque'] for d in enriched_data))}")
    print(f"Avec abonnement: {sum(1 for d in enriched_data if d['abonnement'] == 'oui')}")
    print(f"Avec accès: {sum(1 for d in enriched_data if d['acces'] == 'oui')}")
    print(f"Âge moyen: {sum(d['age'] for d in enriched_data) / len(enriched_data):.1f} ans")
    print(f"\n✅ Fichier créé: {output_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
