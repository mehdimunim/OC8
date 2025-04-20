import os
import requests
import zipfile

def download_and_extract(url, extract_to):
    """Télécharge un fichier ZIP depuis une URL et l'extrait dans un dossier."""

    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(extract_to, exist_ok=True)

    # Extraire le nom du fichier depuis l'URL
    filename = url.split("/")[-1]
    filepath = os.path.join(extract_to, filename)

    # Télécharger le fichier
    print(f"Téléchargement de {url} vers {filepath}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Lève une exception pour les erreurs HTTP

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Téléchargement terminé.")

        # Extraire le fichier ZIP
        print(f"Extraction de {filepath} vers {extract_to}...")
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        print(f"Extraction terminée.")

        # Supprimer le fichier ZIP téléchargé
        os.remove(filepath)
        print(f"Fichier ZIP supprimé : {filepath}")

    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du téléchargement : {e}")
    except zipfile.BadZipFile as e:
        print(f"Erreur lors de l'extraction du fichier ZIP : {e}")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {e}")

if __name__ == "__main__":
    # Configuration
    data_urls = [
        "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Participez+%C3%A0+la+conception+d'une+voiture+autonome/P8_Cityscapes_gtFine_trainvaltest.zip",
        "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Participez+%C3%A0+la+conception+d'une+voiture+autonome/P8_Cityscapes_leftImg8bit_trainvaltest.zip"
    ]

    extract_to_dir = "data/raw"

    # Télécharger et extraire les données
    for url in data_urls:
        download_and_extract(url, extract_to_dir)

    print("Script terminé.")

