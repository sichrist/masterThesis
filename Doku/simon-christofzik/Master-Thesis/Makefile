#################################
#            Defines            #
#################################


FILES2CLEAN=$(shell find ./ -name "*.log" -o -name "*.aux" -o -name "*.toc" -o -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.lot" -o -name "*.xml" -o -name "*.lof" -o -name "*.bcf" -o -name "*.out")
CMD_RM		= rm -rf
CMD_LATEX	= pdflatex
CMD_BIB     = biber
FILE2COMPILE = ./thesis.tex
FILE4BIB     = thesis

.PHONY: all bib compile 
all: compile bib compile clean


compile:
	$(CMD_LATEX) $(FILE2COMPILE)

bib:
	$(CMD_LATEX) $(FILE2COMPILE)
	$(CMD_BIB) $(FILE4BIB)


clean:
	$(CMD_RM) $(FILES2CLEAN)




