LATEX=pdflatex
BIBTEX=bibtex
TARGET=report

all: $(TARGET).pdf

view:
	open $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex references.bbl
	$(LATEX) -interaction=nonstopmode $(TARGET).tex
	$(LATEX) -interaction=nonstopmode $(TARGET).tex

references.bbl: $(TARGET).aux references.bib
	$(BIBTEX) $(TARGET)

$(TARGET).aux: $(TARGET).tex
	$(LATEX) -interaction=nonstopmode $(TARGET).tex

clean:
	rm -f *.aux *.log *.bbl *.blg *.pdf *.out
