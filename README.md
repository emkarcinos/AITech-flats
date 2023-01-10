# AITech-flats

Projekt magisterski AITech - klasyfikacja stylu wnętrz mieszkań i transfer stylu

## Instrukcja

1. Sklonuj repozytorium.
2. Przejdź do lokalizacji źródłowej projektu.
3. Utwórz środowisko wirtualne z Python=3.10
4. Zainstaluj zależności z paczki requirements.txt lub requirements_mps.txt (Wsparcie dla GPU na procesorze Apple M1/M2).

```
cd {root repository path}
conda create -n ENVNAME python=3.10
conda activate ENVNAME

# Domyślna wersja
pip install -r requirements.txt

# Wersja dla procesora M1
pip install -r requirements_mps.txt
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

```
