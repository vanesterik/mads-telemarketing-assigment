.PHONY: \
	bootstrap \
	help \
	install \
	reports


# SYSTEM DEPENDENCIES ##########################################################

## Bootstrap system dependencies
bootstrap:
	@which pdm || brew install pdm
	@which pandoc || brew install pandoc
	@which pdflatex || brew install --cask mactex


# PROJECT DEPENDENCIES #########################################################

## Install required project dependencies
install:
	make bootstrap
	@pdm install


# REPORTS ######################################################################

## Define various variables for reports
REPORTS_DIR := reports
CITATIONS_DIR := $(REPORTS_DIR)/citations
FIGURES_DIR := $(REPORTS_DIR)/figures
PRESENTATION_PATH := $(REPORTS_DIR)/20250629-kve-presentation
REPORTS_PATHS := $(filter-out $(PRESENTATION_PATH), $(basename $(wildcard $(REPORTS_DIR)/*.md)))

## Run Pandoc to generate presentation
reports:

	@for report in $(REPORTS_PATHS); do \
		pandoc \
			$${report}.md \
			--bibliography=${CITATIONS_DIR}/mads-telemarketing-assignment.bib \
			--citeproc \
			--csl=${CITATIONS_DIR}/ieee.xml \
			--output=$${report}.pdf \
			--resource-path=${FIGURES_DIR}; \
	done

	@echo "✅ Reports generated successfully"

	@pandoc \
		${PRESENTATION_PATH}.md \
		-t beamer \
		--output=${PRESENTATION_PATH}.pdf \
		--resource-path=${FIGURES_DIR} \
		--slide-level=1 \
		--variable aspectratio:169 \
		--variable theme:Pittsburgh \
		--variable colortheme:default \
		--variable fonttheme:serif \

	@echo "✅ Presentation generated successfully"



# TEST #########################################################################

## Run tests


# LINT #########################################################################

## Lint code

## Format code



################################################################################
# Self Documenting Commands                                                    #
################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
