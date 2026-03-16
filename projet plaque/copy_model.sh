#!/bin/bash

# Script pour copier le modèle entraîné vers le répertoire demo

echo "🔍 Recherche du modèle entraîné best.pt..."

# Chemins possibles du modèle
PATHS=(
    "/mnt/c/DEV/JEDHA/FULLSTACK_WSL/PROJETS/PROJET_FINAL/2_IMMAT_PLAQUES/YOLO8-n/ROBOFLOW_universe/runs/detect/LP_roboflow/weights/best.pt"
    "../runs/detect/LP_roboflow/weights/best.pt"
    "runs/detect/LP_roboflow/weights/best.pt"
    "./runs/detect/LP_roboflow/weights/best.pt"
)

TARGET="demo/models/best.pt"

# Créer le répertoire si nécessaire
mkdir -p demo/models

for path in "${PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "✅ Modèle trouvé : $path"
        cp "$path" "$TARGET"
        echo "✅ Modèle copié vers $TARGET"
        echo ""
        echo "📊 Informations du modèle :"
        ls -lh "$TARGET"
        echo ""
        echo "🎉 Succès ! Redémarrez l'application pour utiliser le modèle fine-tuné."
        exit 0
    fi
done

echo "❌ Modèle non trouvé dans les emplacements attendus :"
for path in "${PATHS[@]}"; do
    echo "   - $path"
done
echo ""
echo "📝 Solutions :"
echo "   1. Copiez manuellement best.pt vers demo/models/"
echo "   2. Ré-entraînez le modèle avec Connect_roboflow.ipynb"
echo "   3. Continuez avec le modèle de base yolov8n.pt (performances réduites)"
echo ""
echo "📖 Consultez MODEL_INTEGRATION.md pour plus de détails"
exit 1
