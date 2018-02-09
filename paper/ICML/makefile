paper.pdf: *.tex
	pdflatex paper.tex
	bibtex paper.aux
	pdflatex paper.tex
	pdflatex paper.tex

clean:
	rm *.aux *.bbl *.log *.blg paper.pdf
