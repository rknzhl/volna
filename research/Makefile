SOURCES := $(wildcard *.tex)
PDFS := $(SOURCES:.tex=.pdf)

.PHONY: all
all: $(PDFS)

%.pdf: %.tex
	latexmk -silent -xelatex -interaction=nonstopmode -halt-on-error -file-line-error $<
