import json
from pathlib import Path
from typing import Any


def load_hardware_tree_from_file(file_path: str) -> dict[str, Any]:
    """Lädt die hardware_tree.json Datei und gibt sie als Dictionary zurück."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    return data


def main():
    file_path = "export/hardware_tree.json"  # Passe den Pfad bei Bedarf an
    try:
        tree = load_hardware_tree_from_file(file_path)
        print("Hardware-Tree erfolgreich geladen.")
        hostname = list(tree.keys())[0]
        # Beispielzugriffe:
        linux_info = tree[hostname]["os"]
        print("Hostname:", linux_info.get("hostname"))
        cpus = tree[hostname]["cpu"]
        print("Anzahl CPUs:", len(cpus))
        gpus = tree[hostname]["gpu"]
        print("Anzahl GPUs:", len(gpus))
    except Exception as e:
        print("Fehler beim Einlesen:", e)


if __name__ == "__main__":
    main()
