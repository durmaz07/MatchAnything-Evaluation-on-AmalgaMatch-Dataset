## Customize Makefile settings for imt
## 
## If you need to customize your Makefile, make
## changes here rather than in the main Makefile

PMDCO_DISJOINTNESS_REMOVAL_TERMS = $(IMPORTDIR)/pmdco_remove_disjoint.txt
IAO_TO_REMOVE = $(IMPORTDIR)/iao_to_remove.txt
PMDCO_CLASSES_TO_REMOVE = $(IMPORTDIR)/pmdco_classes_to_remove.txt


$(IMPORTDIR)/pmdco_import.owl: $(MIRRORDIR)/pmdco.owl $(IMPORTDIR)/pmdco_terms.txt
	@echo "Generating Application Module from pmdco..."
	if [ $(IMP) = true ]; then $(ROBOT) \
	  query -i $< --update ../sparql/preprocess-module.ru \
	  extract --term-file $(IMPORTDIR)/pmdco_terms.txt \
	          --force true \
	          --copy-ontology-annotations true \
	          --intermediates all \
	          --method BOT \
	  \
	  query --update ../sparql/inject-subset-declaration.ru \
	        --update ../sparql/inject-synonymtype-declaration.ru \
	        --update ../sparql/postprocess-module.ru \
	  \
	  remove --term http://purl.obolibrary.org/obo/IAO_0000412 \
             --select annotation \
	  \
	  remove --term-file $(PMDCO_DISJOINTNESS_REMOVAL_TERMS) \
			 --axioms DisjointClasses \
	  remove --term-file $(PMDCO_CLASSES_TO_REMOVE) \
			 --select "classes"\
	  remove --term-file $(IAO_TO_REMOVE) \
			 --select "individuals classes"\
	  $(ANNOTATE_CONVERT_FILE); \
	fi

$(IMPORTDIR)/ro_import.owl: $(MIRRORDIR)/ro.owl $(IMPORTDIR)/ro_terms.txt \
			   $(IMPORTSEED) | all_robot_plugins
	$(ROBOT) annotate --input $< --remove-annotations \
	     remove --select "RO:*" --select complement --select "classes"  --axioms annotation \
		 odk:normalize --add-source true \
		 extract --term-file $(IMPORTDIR)/ro_terms.txt  \
		         --force true --copy-ontology-annotations true \
		         --individuals exclude \
		         --method SUBSET \
		 remove $(foreach p, $(ANNOTATION_PROPERTIES), --term $(p)) \
		        --term-file $(IMPORTDIR)/ro_terms.txt \
		        --select complement --select annotation-properties \
		 odk:normalize --base-iri https://w3id.org/pmd \
		               --subset-decls true --synonym-decls true \
		 $(ANNOTATE_CONVERT_FILE)


$(IMPORTDIR)/bfo_import.owl: $(MIRRORDIR)/bfo.owl $(IMPORTDIR)/bfo_terms.txt \
			   $(IMPORTSEED) | all_robot_plugins
	$(ROBOT) annotate --input $< --remove-annotations \
	     remove --select "BFO:*" --select complement --select "classes"  --axioms annotation \
		 odk:normalize --add-source true \
		 extract --term-file $(IMPORTDIR)/bfo_terms.txt  \
		         --force true --copy-ontology-annotations true \
		         --individuals exclude \
		         --method SUBSET \
		 remove $(foreach p, $(ANNOTATION_PROPERTIES), --term $(p)) \
		        --term-file $(IMPORTDIR)/bfo_terms.txt \
		        --select complement --select annotation-properties \
		 odk:normalize --base-iri https://w3id.org/pmd \
		               --subset-decls true --synonym-decls true \
		 $(ANNOTATE_CONVERT_FILE)

$(IMPORTDIR)/iao_import.owl: $(MIRRORDIR)/iao.owl $(IMPORTDIR)/iao_terms.txt
	if [ $(IMP) = true ]; then $(ROBOT) query -i $< --update ../sparql/preprocess-module.ru \
		remove --select "IAO:*" --select complement --select "classes object-properties data-properties"  --axioms annotation \
		extract --term-file $(IMPORTDIR)/iao_terms.txt  --force true --copy-ontology-annotations true --individuals exclude --intermediates none --method BOT \
		query --update ../sparql/inject-subset-declaration.ru --update ../sparql/inject-synonymtype-declaration.ru --update ../sparql/postprocess-module.ru \
 		remove --term IAO:0000032 --axioms subclass \
 		remove $(foreach p, $(ANNOTATION_PROPERTIES), --term $(p)) \
		      --select complement --select annotation-properties \
		$(ANNOTATE_CONVERT_FILE); fi



$(ONT)-base.owl: $(EDIT_PREPROCESSED) $(OTHER_SRC) $(IMPORT_FILES)
	$(ROBOT_RELEASE_IMPORT_MODE) \
	reason --reasoner ELK --equivalent-classes-allowed asserted-only --exclude-tautologies structural --annotate-inferred-axioms False \
	relax \
	reduce -r ELK \
	remove --base-iri $(URIBASE)/ --axioms external --preserve-structure false --trim false \
	$(SHARED_ROBOT_COMMANDS) \
	annotate --link-annotation http://purl.org/dc/elements/1.1/type http://purl.obolibrary.org/obo/IAO_8000001 \
		--ontology-iri $(ONTBASE)/$@ $(ANNOTATE_ONTOLOGY_VERSION) \
		--output $@.tmp.owl && mv $@.tmp.owl $@


CITATION=imt: Image Transformation Ontology. Version $(VERSION), https://w3id.org/pmd/imt/

ALL_ANNOTATIONS=--ontology-iri https://w3id.org/pmd/imt/ -V https://w3id.org/pmd/imt/$(VERSION) \
	--annotation http://purl.org/dc/terms/created "$(TODAY)" \
	--annotation owl:versionInfo "$(VERSION)" \
	--annotation http://purl.org/dc/terms/bibliographicCitation "$(CITATION)" \
	--link-annotation owl:priorVersion https://w3id.org/pmd/imt/$(PRIOR_VERSION)

update-ontology-annotations: 
	$(ROBOT) annotate --input imt.owl $(ALL_ANNOTATIONS) --output ../../imt.owl
	$(ROBOT) annotate --input imt.ttl $(ALL_ANNOTATIONS) --output ../../imt.ttl
	$(ROBOT) annotate --input imt-full.owl $(ALL_ANNOTATIONS) --output ../../imt-full.owl
	$(ROBOT) annotate --input imt-full.ttl $(ALL_ANNOTATIONS) --output ../../imt-full.ttl
	$(ROBOT) annotate --input imt-base.owl $(ALL_ANNOTATIONS) --output ../../imt-base.owl
	$(ROBOT) annotate --input imt-base.ttl $(ALL_ANNOTATIONS) --output ../../imt-base.ttl

all_assets: update-ontology-annotations
