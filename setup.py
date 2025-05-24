#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# Lecture du README pour la description longue
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    # Ces paramètres sont maintenant dans pyproject.toml
    # Ils sont conservés ici pour la rétrocompatibilité
    name="neuro-ai",
    version="0.1.0",
    author="Votre Nom",
    author_email="votre.email@example.com",
    description="Une architecture d'IA universelle modulaire et évolutive",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/votre-utilisateur/neuro",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    # Les dépendances sont gérées par pyproject.toml
)