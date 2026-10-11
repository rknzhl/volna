PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
SOURCE_DIR := $(PROJECT_ROOT)/research
BUILD_DIR := $(PROJECT_ROOT)/output/pdf
SOURCES := $(wildcard $(SOURCE_DIR)/*.tex)
PDFS := $(SOURCES:.tex=.pdf)

.PHONY: all method
all: $(PDFS)

$(SOURCE_DIR)/%.pdf: $(SOURCE_DIR)/%.tex
	mkdir -p $(BUILD_DIR)
	latexmk -silent -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(BUILD_DIR) $<
	cp $(BUILD_DIR)/$(notdir $@) $@

method: $(SOURCE_DIR)/phase_ssa_method.pdf
