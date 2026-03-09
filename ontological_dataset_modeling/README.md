
![Build Status](https://github.com/durmaz07/MatchAnything-Evaluation-on-AmalgaMatch-Dataset/actions/workflows/qc.yml/badge.svg)

# ImageTransformation: Application-level ontology for image transformation

## Introduction

ImageTransformation is an application level ontology based on the third version of the [Platform Material Digital Core Ontology (PMDco)](https://github.com/materialdigital/core-ontology). It aims to provice a semantic description of image transformation processes described in the [AmalgaMatch Dataset](https://github.com/durmaz07/MatchAnything-Evaluation-on-AmalgaMatch-Dataset). An examplary design pattern for the semantic description is provided in the accompanying the article [Foundation Models for Multimodal Image Data Fusion in Materials Science]().


## File Structure

This folder provides the modular implementation of ImageTransformation, developed and maintained using the [Ontology Development Kit (ODK)](https://github.com/INCATools/ontology-development-kit).

### Directories
* **src/:** Main development folder generated and managed through ODK.
  * **ontology/components/:** – Modular ontology components (general entities, microscopy, transformation).
  * **ontology/imt-edit.owl:** – Primary editable ontology file used during development (ontology editors' version).

### Ontology versions
* **imt-full.owl/ttl:** Complete ontology with all imports and full axiomatization.
* **imt-base.owl/ttl:** Core entities without extended imports.
* **imt-simple.owl/ttl:** Simplified version with basic subclass and existential axioms.
* **imt.owl/ttl:** Main ontology file contains the full version.

### Other files
* README.md, LICENSE.txt, CONTRIBUTING.md – Project overview, license, and contribution guidelines.
